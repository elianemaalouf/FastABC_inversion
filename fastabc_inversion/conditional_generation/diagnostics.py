"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Base class for diagnostics on the trained models and SuS inference for conditional generation experiments.
"""
from torch.utils.data import Subset, DataLoader
import torch
import os
import numpy as np

from fastabc_inversion.utils.utilities import compute_stats
from fastabc_inversion.utils.evaluation.mmd import MMD2 as mmd, two_sample_mmd_test
from fastabc_inversion.utils.torch_distances import rmse_torch
import fastabc_inversion.conditional_generation.utils.plotting as plot

def compare_labels(target, predicted):
    """
    Compare two label tensors and return the accuracy.
    :param target: tensor of true labels
    :param predicted: tensor of predicted labels
    :return: accuracy (float)
    """
    # check that both target and predicted are tensors of same shape
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.long)
    if not isinstance(predicted, torch.Tensor):
        predicted = torch.tensor(predicted, dtype=torch.long)
    if target.shape != predicted.shape:
        raise ValueError("Target and predicted tensors must have the same shape.")

    correct = (target == predicted).sum().item()
    total = target.size(0)
    accuracy = correct / total
    return accuracy

class Diagnostics:

    def __init__(self, experiment_obj, epoch, round_digits=3):

        self.mmd_kernel_params = {"metric": "rbf"}
        self.mmd_est_params = {'x': None, 'y': None, 'z': None} # TODO: make these parameters configurable from outside

        self.AVAILABLE_DIAGS = {
            "prior_vs_gen_dist_distance": self.make_priorvsgen_dists_distances_stats,
            "recons_stats": self.make_recons_stats,
            "resim_stats": self.assess_forward_learning,
            "latent_dist": self.inspect_latent_distribution,
            "orig_data_embed_viz": self.plot_x_y_umap_tsne_scatters_2,
            "assess_frechet_scores": self.assess_frechet_scores,
            "class_proportions": self.assess_class_proportions,
            }
        self.two_sample_test_params = {"kernel_params":self.mmd_kernel_params.copy(), "unbiased": True, "alpha": 0.05,
                                  "iterations": 1000} # TODO : make these parameters configurable from outside

        self.mmd_params = {"kernel_params": self.mmd_kernel_params.copy(), "unbiased": False} # TODO : make these parameters configurable from outside

        self.DEFAULT_DIAG_PARAMS = {
            "prior_vs_gen_dist_distance": {"mmd_params": self.mmd_params,
                                           "sample_size": 300, "repeats": 100,
                                           "two_sample_test_params": self.two_sample_test_params},
                                           #"compute_only_with_rand_proj": False, "rand_proj_dims":30}, # TODO : delete these parameters
            "recons_stats": {"build_y_stats": True, "make_plots": True},
            "resim_stats": {"make_plots": True},
            "latent_dist": {"make_plots": True, "n_neighbors": 5, "min_dist": 0.1, "fit_prior_only":False, "denseMAP":True,
                            "mmd_params": self.mmd_params, "two_sample_test_params": self.two_sample_test_params,
                            "test_repeats": 100, "test_sample_size": 300},
            "orig_data_embed_viz":{"fit_prior_only":False, "denseMAP":True, "n_neighbors": 15, "min_dist": 0.1},
            "assess_frechet_scores": {"pca_components": .85, "sample_size": 500, "repeats": 100},
            "class_proportions": {},

            }
        self.NO_RESULTS_DIAGS = ["orig_data_embed_viz", "class_proportions"]

        self.experiment = experiment_obj
        self.round_digits = round_digits
        self.epoch = epoch

        self.classifier = None

        # set experiment to model to eval mode
        self.experiment.model.eval()

        # data objects for diagnostics
        self.training_data = None # training subset dataloader
        self.train_x = None
        self.train_y = None
        self.train_y_label = None

        self.validation_data = None # validation subset dataloader
        self.val_x = None
        self.val_y = None
        self.val_y_label = None

        self.reconstructed_train_x = None
        self.reconstructed_train_y = None
        self.reconstructed_train_y_label = None
        self.resimulated_recon_train_x = None  # classified reconstructed_train_x
        self.resimulated_recon_train_x_label = None

        self.reconstructed_val_x = None
        self.reconstructed_val_y = None
        self.reconstructed_val_y_label = None
        self.resimulated_recon_val_x = None # classified reconstructed_val_x
        self.resimulated_recon_val_x_label = None

        self.generated_x = None
        self.generated_y = None
        self.generated_y_label = None
        self.resimulated_generated_x = None # classified generated_x
        self.resimulated_generated_x_label = None

        self.latent_vector = None
        self.latent_train_codes = None
        self.latent_val_codes = None

        # setup logging file in parent directory
        self.logging_dir = None
        self.all_exp_logging_file = None
        self.logging_string = None
        self.logging_headers = None
        self.exp_diag_stats_file = None
        self.hyperparams_string = None
        self.hpyperparams_headers = None
        self.results_vecs = {}

    def load_classifier(self, classifier, append_softmax=False):
        """
        Load a pretrained classifier to be used for diagnostics.
        :param append_softmax: If True, append a softmax layer to the classifier.
        :param classifier: The pretrained classifier model.
        """
        if not append_softmax:
            self.classifier = classifier
        else:
            self.classifier = torch.nn.Sequential(
                classifier,
                torch.nn.Softmax(dim=1)
            )

        # set to eval mode
        self.classifier.eval()


    def build_diagnostics_data(self, sample_size=None):
        """
        Make reconstruction and generated data for diagnostics
        """

        netG = self.experiment.model.netG
        netD = self.experiment.model.netD

        z_dist = self.experiment.latent_dist
        z_dist_params = self.experiment.latent_dist_params_list
        latent_dim = self.experiment.latent_dim
        device = self.experiment.device

        self.sample_size = self.experiment.val_size if sample_size is None else sample_size

        # make random laten_vector :
        self.latent_vector = (
            z_dist(
                torch.tensor(z_dist_params[0], dtype=torch.float32),
                torch.tensor(z_dist_params[1], dtype=torch.float32),
            )
            .sample((self.sample_size, latent_dim))
            .to(device)
        )

        # get prior samples from training data: randomly select samples from different bartches of experiment.train_loader
        train_random_indices = torch.randperm(self.experiment.train_size)[:self.sample_size]
        self.training_data = DataLoader(
            Subset(self.experiment.train_loader.dataset, train_random_indices),
            batch_size=300,
            shuffle=False,  # No need to shuffle a fixed assessment set
            num_workers=0  # Can keep this low or 0 for a simple assessment
        )

        all_train_samples = [self.training_data.dataset[i] for i in range(len(self.training_data.dataset))]
        self.train_x = torch.stack([sample[0] for sample in all_train_samples])
        self.train_y = torch.stack([sample[1] for sample in all_train_samples])
        self.train_y_label = self.experiment.label_transform.simplex_vec_to_label(self.train_y)

        val_random_indices = torch.randperm(self.experiment.val_size)[:self.sample_size]
        self.validation_data = DataLoader(
            Subset(self.experiment.val_loader.dataset, val_random_indices),
            batch_size=300,
            shuffle=False,  # No need to shuffle a fixed assessment set
            num_workers=0  # Can keep this low or 0 for a simple assessment
        )

        all_val_samples = [self.validation_data.dataset[i] for i in range(len(self.validation_data.dataset))]
        self.val_x = torch.stack([sample[0] for sample in all_val_samples])
        self.val_y = torch.stack([sample[1] for sample in all_val_samples])
        self.val_y_label = self.experiment.label_transform.simplex_vec_to_label(self.val_y)

        del all_train_samples, all_val_samples

        with torch.no_grad():

            latent_train_codes_list = []
            reconstructed_train_x_list = []
            reconstructed_train_y_list = []
            resimulated_recon_train_x_list = []

            for batch in self.training_data:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                latent_codes = netD(images, labels)
                latent_train_codes_list.append(latent_codes.cpu())
                reconstructed_train = netG(latent_codes)
                reconstructed_train_x_list.append(reconstructed_train[0].cpu())
                reconstructed_train_y_list.append(reconstructed_train[1].cpu())
                # classify reconstructed_train[0] and store in resimulated_recon_train_x_list
                if self.classifier is not None:
                    classified_recon_x = self.classifier(reconstructed_train[0].to(device))
                    resimulated_recon_train_x_list.append(classified_recon_x.cpu())
                else:
                    # warning message if no classifier is loaded
                    print("Warning: No classifier loaded. resimulated_recon_train_x will not be computed.")

            self.latent_train_codes = torch.cat(latent_train_codes_list, dim=0)
            self.reconstructed_train_x = torch.cat(reconstructed_train_x_list, dim=0)
            self.reconstructed_train_y = torch.cat(reconstructed_train_y_list, dim=0)
            self.reconstructed_train_y_label = self.experiment.label_transform.simplex_vec_to_label(self.reconstructed_train_y)

            if self.classifier is not None:
                self.resimulated_recon_train_x = torch.cat(resimulated_recon_train_x_list, dim=0)
                self.resimulated_recon_train_x_label = self.experiment.label_transform.simplex_vec_to_label(self.resimulated_recon_train_x)

            del latent_train_codes_list, reconstructed_train_x_list, reconstructed_train_y_list, resimulated_recon_train_x_list

            latent_val_codes_list = []
            reconstructed_val_x_list = []
            reconstructed_val_y_list = []
            resimulated_recon_val_x_list = []

            for batch in self.validation_data:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                latent_codes = netD(images, labels)
                latent_val_codes_list.append(latent_codes.cpu())
                reconstructed_val = netG(latent_codes)
                reconstructed_val_x_list.append(reconstructed_val[0].cpu())
                reconstructed_val_y_list.append(reconstructed_val[1].cpu())
                # classify reconstructed_val[0] and store in resimulated_recon_val_x_list
                if self.classifier is not None:
                    classified_recon_x = self.classifier(reconstructed_val[0].to(device))
                    resimulated_recon_val_x_list.append(classified_recon_x.cpu())
                else:
                    # warning message if no classifier is loaded
                    print("Warning: No classifier loaded. resimulated_recon_val_x_list will not be computed.")

            self.latent_val_codes = torch.cat(latent_val_codes_list, dim=0)
            self.reconstructed_val_x = torch.cat(reconstructed_val_x_list, dim=0)
            self.reconstructed_val_y = torch.cat(reconstructed_val_y_list, dim=0)
            self.reconstructed_val_y_label = self.experiment.label_transform.simplex_vec_to_label(self.reconstructed_val_y)
            if self.classifier is not None:
                self.resimulated_recon_val_x = torch.cat(resimulated_recon_val_x_list, dim=0)
                self.resimulated_recon_val_x_label = self.experiment.label_transform.simplex_vec_to_label(self.resimulated_recon_val_x)

            del latent_val_codes_list, reconstructed_val_x_list, reconstructed_val_y_list, resimulated_recon_val_x_list

            # make generated data from random latent_vector
            generated = netG(self.latent_vector)
            self.generated_x = generated[0].cpu()
            self.generated_y = generated[1].cpu()
            self.latent_vector = self.latent_vector.cpu()
            self.generated_y_label = self.experiment.label_transform.simplex_vec_to_label(self.generated_y)
            # classify generated_x and store in resimulated_generated_x
            if self.classifier is not None:
                self.resimulated_generated_x = self.classifier(self.generated_x.to(device)).cpu()
                self.resimulated_generated_x_label = self.experiment.label_transform.simplex_vec_to_label(self.resimulated_generated_x)
            else:
                # warning message if no classifier is loaded
                print("Warning: No classifier loaded. resimulated_generated_x will not be computed.")

    def setup_logging(self, logging_comment=""):
        """
        Setup logging directory and files
        :param logging_comment: comment to add to the logging string, after the hyperparameters string
        """

        def make_logging_strings(hpyer_params_dicts_list):
            """
            Make logging strings for hyperparameters
            :param hpyer_params_dicts_list: list of dictionaries of hyperparameters to log. Each dictionary should have
                                        the hyperparameter name as key (=header) and the hyperparameter value as value.
            """
            logging_string = ""
            logging_headers = ""
            for hyper_params_dict in hpyer_params_dicts_list:
                for key, value in hyper_params_dict.items():
                    logging_string = f"{logging_string}:{value}"
                    logging_headers = f"{logging_headers}:{key}"
            return logging_string, logging_headers

        self.logging_dir = self.experiment.experiment_dir + f"/diagnostics_epoch_{self.epoch}"
        self.reconstructions_diag_dir = self.logging_dir + "/reconstructions"
        self.resimulations_diag_dir = self.logging_dir + "/resimulations"
        self.latent_space_diag_dir = self.logging_dir + "/latent_space"

        dirs_to_create = [self.logging_dir, self.reconstructions_diag_dir, self.resimulations_diag_dir,
                          self.latent_space_diag_dir]

        # create directories if they don't exist
        for dir in dirs_to_create:
            os.makedirs(dir, exist_ok=True)

        self.all_exp_logging_file = self.experiment.experiment_rootdir + "/jGNNdiag.csv"
        self.logging_string = f"{self.experiment.latent_dim}:{self.experiment.model_log_ref}:"
        self.logging_headers = "latent_dim:model_log_ref:"
        self.exp_diag_stats_file = self.logging_dir + "/stats.txt"

        # write hyperparameters to string
        # read the hyperparameters from the experiment object
        seed = {"seed": self.experiment.seed}
        train_size = {"train_size":self.experiment.train_size}
        diagnostics_sample_size = {"diagnostics_sample_size":self.sample_size}
        optimizer_name = {"optimizer_name":type (self.experiment.optimizer).__name__ if self.experiment.optimizer
                                                                                         is not None else 'N/A'}
        decoder_model_selector = {"decoder_model_selector": self.experiment.model_selector}
        decoder_smooth = {"decoder_smoothness": self.experiment.nn_params['netG_apply_spectral_norm']}
        latent_dimension = {"latent_dimension":self.experiment.latent_dim}
        latent_distribution = {"latent_distribution":self.experiment.latent_dist_name}
        latent_distribution_params = {"latent_distribution_params":self.experiment.latent_dist_params_list}
        norm_power_p = {"norm_power_p":self.experiment.norms_params['l_norm_p_x']}
        batch_size = {"batch_size":self.experiment.model_training_params['batch_size']}
        nb_epochs = {"nb_epochs":self.experiment.model_training_params['nb_epochs']}
        sink_lambda = {"sink_lambda":self.experiment.sinkhorn_lambda_scheduling_params['sink_lambda']} # final value at the end of training
        sink_eps = {"sink_eps":self.experiment.sinkhorn_params['epsilon']}
        sink_p = {"sink_p":self.experiment.sinkhorn_params['p']}
        lambda_sink_scheduling = {"lambda_sink_scheduling": self.experiment.sinkhorn_lambda_scheduling_params["sink_lambda_scheduler_factor"]}
        lr_scheduling = {"lr_scheduling": self.experiment.lr_scheduling}
        lr_scheduler = {"lr_scheduler": self.experiment.optim_params.get('lr_scheduler', None)}
        lr = {"lr": self.experiment.optim_params['lr']}
        beta1 = {"beta1": self.experiment.optim_params['betas'][0]}
        beta2 = {"beta2": self.experiment.optim_params['betas'][1]}
        inflate_recon_y = {"inflate_recon_y": self.experiment.model_training_params['inflate_recon_y']}
        logging_comment = {"logging_comment":logging_comment}

        hyperparams_to_log = [seed, train_size, diagnostics_sample_size, nb_epochs, optimizer_name, decoder_model_selector, decoder_smooth,
                              latent_dimension, latent_distribution, latent_distribution_params, norm_power_p, batch_size,
                              sink_lambda, sink_eps, sink_p, lambda_sink_scheduling, lr_scheduling, lr_scheduler,
                              lr, beta1, beta2, inflate_recon_y, logging_comment]

        self.hyperparams_string, self.hpyperparams_headers = make_logging_strings(hyperparams_to_log)

    def estimate_mmd_params(self, which_variable, max_sample_size = 10000):
        """
        Estimate MMD kernel parameter using median heuristic.
        :param which_variable: string indicating which variable to use for estimation. Options are 'x', 'y', 'z'.
        :param max_sample_size: maximum number of samples to use for estimation.
        """
        print("Estimating median squared euclidean distance for MMD kernel parameter...")

        from fastabc_inversion.utils.evaluation.mmd import estimate_median_pairwise_dists

        if which_variable not in ['x', 'y', 'z']:
            raise ValueError("which_variable must be one of 'x', 'y', 'z'.")

        max_sample_size = max_sample_size

        if which_variable in ['x', 'y']:
            large_sample_size = min(max_sample_size, self.experiment.train_size)
            # get large sample from training data
            random_indices = torch.randperm(self.experiment.train_size)[:large_sample_size]
            data_loader = DataLoader(
                Subset(self.experiment.train_loader.dataset, random_indices),
                batch_size=300,
                shuffle=False,
                num_workers=0
            )
            all_train_samples = [data_loader.dataset[i] for i in range(len(data_loader.dataset))]
            # extract x or y samples
            all_train_x = torch.stack([sample[0] for sample in all_train_samples])
            all_train_y = torch.stack([sample[1] for sample in all_train_samples])
            del all_train_samples

            # make large generated samples
            large_latent_vector = self.experiment.latent_dist(
                torch.tensor(self.experiment.latent_dist_params_list[0], dtype=torch.float32),
                torch.tensor(self.experiment.latent_dist_params_list[1], dtype=torch.float32),
            ).sample((large_sample_size, self.experiment.latent_dim)).to(self.experiment.device)

            with torch.no_grad():
                gen_x, gen_y = self.experiment.netG(large_latent_vector)
            del large_latent_vector

            if which_variable == 'x':
                train_samples = all_train_x.numpy()
                gen_samples = gen_x.cpu().numpy()
                # stack train_samples and gen_samples for median heuristic
                all_data = np.vstack((train_samples, gen_samples))
                median_x = estimate_median_pairwise_dists(all_data, sample_ratio= 1.0, chunk_size=300)
                self.mmd_est_params['x'] = 1/median_x
                del all_data, train_samples, gen_samples, all_train_x, gen_x

            if which_variable == 'y':
                train_samples = all_train_y.numpy()
                gen_samples = gen_y.cpu().numpy()
                # stack train_samples and gen_samples for median heuristic
                all_data = np.vstack((train_samples, gen_samples))
                median_y = estimate_median_pairwise_dists(all_data, sample_ratio= 1.0, chunk_size=300)
                self.mmd_est_params['y'] = 1/median_y
                del all_data, train_samples, gen_samples, all_train_y, gen_y

        else: # which_variable == 'z'
            large_sample_size = min(max_sample_size, self.experiment.val_size)

            # get large sample from validation data
            random_indices = torch.randperm(self.experiment.val_size)[:large_sample_size]
            data_loader = DataLoader(
                Subset(self.experiment.val_loader.dataset, random_indices),
                batch_size=300,
                shuffle=False,
                num_workers=0
            )
            all_val_samples = [data_loader.dataset[i] for i in range(len(data_loader.dataset))]
            # compute latent codes for all samples
            latent_codes_list = []
            for sample in all_val_samples:
                image = sample[0].to(self.experiment.device).unsqueeze(0)
                label = sample[1].to(self.experiment.device).unsqueeze(0)
                with torch.no_grad():
                    latent_code = self.experiment.netD(image, label)
                latent_codes_list.append(latent_code.cpu())
            all_val_latent_codes = torch.cat(latent_codes_list, dim=0).numpy()
            del all_val_samples, latent_codes_list, latent_code

            # make large generated samples
            large_latent_vector = self.experiment.latent_dist(
                torch.tensor(self.experiment.latent_dist_params_list[0], dtype=torch.float32),
                torch.tensor(self.experiment.latent_dist_params_list[1], dtype=torch.float32),
            ).sample((large_sample_size, self.experiment.latent_dim)).numpy()

            all_data = np.vstack((all_val_latent_codes, large_latent_vector))
            median_z = estimate_median_pairwise_dists(all_data, sample_ratio=1.0, chunk_size=300)
            self.mmd_est_params['z'] = 1/median_z
            del all_data, all_val_latent_codes, large_latent_vector

        # store estimated parameter on disk
        with open(self.logging_dir + f"/mmd_{which_variable}_kernel_param.txt", "w") as f:
            f.write(f"median squared euclidean : {str(self.mmd_est_params[which_variable])}")


    def run_diagnostics(self, sample_size=2000, diagnostics_to_run=[], diags_params = {}, logging_comment='', write_to_file=True):
        """
        Run all diagnostics and log to files
        :param diagnostics_to_run: list of diagnostics to run, if empty run all listed in AVAILABLE_DIAGS
        :param diags_params: dictionary of parameters for each diagnostic. If empty, use DEFAULT_DIAG_PARAMS
        :param logging_comment: comment to add to the logging string, after the hyperparameters string
        :param write_to_file: if True, write results to file
        """

        # build diagnostics data
        self.build_diagnostics_data(sample_size=sample_size)

        # setup logging directory and files
        self.setup_logging(logging_comment=logging_comment)

        if 'inspect_latent_distribution' in diagnostics_to_run and (self.mmd_est_params['z'] is None):
            self.estimate_mmd_params(which_variable='z', max_sample_size=10000)
        if 'prior_vs_gen_dist_distance' in diagnostics_to_run and (self.mmd_est_params['x'] is None):
            self.estimate_mmd_params(which_variable='x', max_sample_size=10000)

        # run diagnostics
        if len(diagnostics_to_run) == 0:
            diagnostics_to_run = self.AVAILABLE_DIAGS.keys()
        if len(diags_params) == 0:
            diags_params = self.DEFAULT_DIAG_PARAMS

        for diag in diagnostics_to_run:
            if diag not in self.NO_RESULTS_DIAGS:
                results_dict = self.AVAILABLE_DIAGS[diag](**diags_params[diag])
                if write_to_file:
                    self.write_results(results_dict)
            else:
                self.AVAILABLE_DIAGS[diag](**diags_params[diag])

        # write logging string to file
        if write_to_file:
            # append hyperparameters string to logging string
            self.append_logging_string(self.hyperparams_string, self.hpyperparams_headers)
            self.write_logging_string_to_file()


    def append_logging_string(self, text_to_append, header_to_append):

        self.logging_headers = self.logging_headers + ":" + header_to_append
        self.logging_string = self.logging_string + ":" + text_to_append


    def write_logging_string_to_file(self):

        if os.path.exists(self.all_exp_logging_file):
            file_exists = True
        else:
            file_exists = False

        with open(self.all_exp_logging_file, "a+") as f:
            f.write("\n")
            if not file_exists:
                f.write(self.logging_headers)
                f.write("\n")
            f.write(self.logging_string)

    def write_results(self, results_dict):
        """
        Write results dictionary to stats file and to logging_string
        :param results_dict: dictionary of results to write
        """
        from datetime import datetime
        # write to logging_string
        data = results_dict["data"]
        data_keys = list(data.keys())
        headers = results_dict["headers"]
        headers_keys = list(headers.keys())
        text_to_file = f"\n {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} \n" #add date and time

        for i in range(len(data_keys)):
            self.logging_string = f"{self.logging_string}:{data[data_keys[i]][0]}"
            self.logging_headers = f"{self.logging_headers}:{headers[headers_keys[i]]}"
            # append to text_to_file
            text_to_file = f"{text_to_file}{data[data_keys[i]][1]}: {data[data_keys[i]][0]} \n"

        # write to stats file
        with open(self.exp_diag_stats_file, "a+") as f:
            f.write(text_to_file)
            f.write("\n")


    def compute_array_stats(self, np_array = None):

        return compute_stats(np_array)

    def make_array_stats_string(self, stats, precision):
        """
        Prepare string for array statistics.
        :param stats: dictionnary with keys "mean", "std", "median", "q25", "q75", "q025", "q975"
        """
        mean = f"{stats['mean']:.{precision}f}" if stats['mean'] is not None else "N/A"
        std = f"{stats['std']:.{precision}f}" if stats['std'] is not None else "N/A"
        median = f"{stats['median']:.{precision}f}" if stats['median'] is not None else "N/A"
        q25 = f"{stats['q25']:.{precision}f}" if stats['q25'] is not None else "N/A"
        q75 = f"{stats['q75']:.{precision}f}" if stats['q75'] is not None else "N/A"
        q025 = f"{stats['q025']:.{precision}f}" if stats['q025'] is not None else "N/A"
        q975 = f"{stats['q975']:.{precision}f}" if stats['q975'] is not None else "N/A"

        return (f"{mean}, {std}, {median},([{q25},{q75}],[{q025},{q975}])")

    def make_priorvsgen_dists_distances_stats(self, mmd_params=None, sample_size=300,
                                              repeats=100, two_sample_test_params=None):
        """
        Make statistics for distances between generated and training data using MMD two-sample test.
        :param mmd_params: dictionary of parameters for MMD function. It should contain the following keys:
                            "kernel_params", "unbiased". kernel_params should contain the following
                            keys: "metric" and other keys depending on the metric used. Check MMD function for more details.
                            unbiased should be a boolean (True to use unbiased estimator of MMD, False otherwise).
        :param sample_size: number of samples to use in the empirical estimations of the distances.
        :param repeats: number of times to repeat the computation of the distances to estimate means and stds.
        :param two_sample_test_params: if not None, it should be a dictionary containing the two-sample MMD test parameters.
        """
        print("Making prior vs generated distributions distance statistics...")
        precision = self.round_digits
        compute_mmd = True if mmd_params is not None else False

        two_samples_test_p_values = [] # store all p-value from two-sample tests repetitions
        two_samples_test_rejections = [] # store all rejections results from two-sample tests repetitions
        mmd_x_dists = []
        mmd_x_refs = []

        # set kernel parameter
        if self.mmd_est_params['x'] is None:
            self.estimate_mmd_params(which_variable='x', max_sample_size=10000)
        self.two_sample_test_params['kernel_params']['gamma'] = self.mmd_est_params['x']
        self.mmd_params['kernel_params']['gamma'] = self.mmd_est_params['x']
        print(f"Using estimated MMD X kernel parameter gamma: {self.mmd_est_params['x']}")

        for i in range(repeats):
            print(f"Repeat {i+1}/{repeats}...")

            # get sample from self.experiment.train_loader of size sample_size
            train_loader_sample = DataLoader(
                Subset(self.experiment.train_loader.dataset,
                       torch.randperm(self.experiment.train_size)[:sample_size*2]),
                batch_size=sample_size*2,
                shuffle=False,
                num_workers=0
            )
            all_train_samples = [train_loader_sample.dataset[i] for i in range(len(train_loader_sample.dataset))]
            all_train_samples = torch.stack([sample[0] for sample in all_train_samples]).view(-1, self.experiment.dim_x).numpy()
            train_sample_1 = all_train_samples[:sample_size, :]
            train_sample_2 = all_train_samples[sample_size:, :]
            del all_train_samples

            # generate a random sample from latent distribution
            latent_vector = self.experiment.latent_dist(
                torch.tensor(self.experiment.latent_dist_params_list[0], dtype=torch.float32),
                torch.tensor(self.experiment.latent_dist_params_list[1], dtype=torch.float32),
            ).sample((sample_size, self.experiment.latent_dim)).to(self.experiment.device)
            with torch.no_grad():
                gen_x, _ = self.experiment.netG(latent_vector)
            generated_sample = gen_x.cpu().view(-1, self.experiment.dim_x).numpy()
            del latent_vector, gen_x

            if compute_mmd:
                print("computing MMD distances on X data...")
                mmd_x_dists.append(mmd(train_sample_1, generated_sample, **mmd_params)[0])
                mmd_x_refs.append(mmd(train_sample_1, train_sample_2, **mmd_params)[0])
            else:
                mmd_x_dists = mmd_x_refs = None

            if two_sample_test_params is not None:
                print("computing two-sample MMD test on X data...")
                # store p-value
                test_result = two_sample_mmd_test(train_sample_1, generated_sample, **two_sample_test_params)
                two_samples_test_p_values.append(test_result[1])
                two_samples_test_rejections.append(test_result[2])
            else:
                two_samples_test_p_values = two_samples_test_rejections = None

        # if testing was done, combine p-values in chi-square statistic
        if two_samples_test_p_values is not None:
            combined_stat = -2 * np.sum(np.log(two_samples_test_p_values))
            from scipy.stats import chi2
            combined_p_value = 1 - chi2.cdf(combined_stat, 2 * repeats)

            rejection_proportion = np.mean(np.array(two_samples_test_rejections))
        else:
            combined_p_value = rejection_proportion = None

        mmd_x_dists = np.array(mmd_x_dists) if mmd_x_dists is not None else None
        mmd_x_refs = np.array(mmd_x_refs) if mmd_x_refs is not None else None

        self.results_vecs["mmd_x_dists"] = mmd_x_dists
        self.results_vecs["two_sample_tests_x_p_values"] = two_samples_test_p_values
        self.results_vecs["two_sample_tests_x_rejections"] = two_samples_test_rejections

        write_combined_fisher_pvalue_two_sample_x = f"{combined_p_value:.{precision}f}" \
            if combined_p_value is not None else None
        write_combined_fisher_pvalue_two_sample_x_txt = f"Combined p-value from Fisher (chi-2) distrbution for {repeats} repetitions of two-sample MMD test on x data (train vs gen):"
        write_header_combined_fisher_pvalue_two_sample_x = "combined_pvalue_two_sample_x"

        write_rejection_prop_two_sample_x = f"{rejection_proportion:.{precision}f}" \
            if rejection_proportion is not None else None
        write_rejection_prop_two_sample_x_txt = f"H0 rejection proportions for {repeats} repetitions of two-sample MMD test on x data (train vs gen):"
        write_headerrejection_prop_two_sample_x = "rejection_proportion_sample_x"

        # run stats & prepare logging strings
        mmd_x_dists_stats = self.compute_array_stats(mmd_x_dists)
        mmd_x_refs_stats = self.compute_array_stats(mmd_x_refs)

        write_data_mmd_x = self.make_array_stats_string(mmd_x_dists_stats, precision)
        write_data_mmd_x_txt = f"MMD distances between training and generated x data (est. with sample size={sample_size * repeats}):"
        write_header_mmd_x = "mmd_x_mean, mmd_x_median, [(mmd_x_q25, mmd_x_q75), (mmd_x_q025, mmd_x_q975)]"

        write_data_mmd_x_ref = self.make_array_stats_string(mmd_x_refs_stats, precision)
        write_data_mmd_x_ref_txt = f"Ref-MMD distances between two training x data samples (est. with sample size={sample_size * repeats}):"
        write_header_mmd_x_ref = f"mmd_x_ref_mean, mmd_x_ref_median, [(mmd_x_ref_q25, mmd_x_ref_q75), (mmd_x_ref_q025, mmd_x_ref_q975)]"

        return {
            "data": {
                "write_data_mmd_x": (write_data_mmd_x, write_data_mmd_x_txt),
                "write_data_mmd_x_ref": (write_data_mmd_x_ref, write_data_mmd_x_ref_txt),
                "write_combined_fisher_pvalue_two_sample_x": (write_combined_fisher_pvalue_two_sample_x, write_combined_fisher_pvalue_two_sample_x_txt),
                "write_rejection_prop_two_sample_x": (write_rejection_prop_two_sample_x, write_rejection_prop_two_sample_x_txt),
            },
            "headers": {
                "write_header_mmd_x": write_header_mmd_x,
                "write_header_mmd_x_ref": write_header_mmd_x_ref,
                "write_header_combined_fisher_pvalue_two_sample_x": write_header_combined_fisher_pvalue_two_sample_x,
                "write_headerrejection_prop_two_sample_x": write_headerrejection_prop_two_sample_x,
            }
        }

    def make_priorvsgen_dists_distances_stats_2(self, mmd_params=None, sample_size=300,
                                              repeats=100, two_sample_test_params=None,
                                              compute_only_with_rand_proj=True, rand_proj_dims=30):
        # TODO : delete
        """
        Make statistics for distances between generated and training data using MMD two-sample test.

        :param mmd_params: dictionary of parameters for MMD function. It should contain the following keys:
                            "kernel_params", "unbiased". kernel_params should contain the following
                            keys: "metric" and other keys depending on the metric used. Check MMD function for more details.
                            unbiased should be a boolean (True to use unbiased estimator of MMD, False otherwise).
        :param sample_size: number of samples to use in the empirical estimations of the distances.
        :param repeats: number of times to repeat the computation of the distances to estimate means and stds.
        :param two_sample_test_params: if not None, it should be a dictionary containing the two-sample MMD test parameters.
        :param compute_only_with_rand_proj_x: if True, compute MMD distances only with random projections of the data.
        :param rand_proj_dims: number of dimensions for the random projections.
        """
        print("Making prior vs generated distributions distance statistics for IMAGE data only...")
        precision = self.round_digits

        from sklearn.random_projection import SparseRandomProjection

        compute_mmd = True if mmd_params is not None else False

        train_x = self.train_x.view(-1, self.experiment.dim_x).numpy()
        train_size = train_x.shape[0]

        generated_x = self.generated_x.view(-1, self.experiment.dim_x).numpy()
        gen_size = generated_x.shape[0]

        mmd_x_dists = []
        mmd_proj_x_dists = []
        mmd_x_refs = []
        mmd_proj_x_refs = []
        two_sample_tests_x = []
        two_sample_tests_proj_x = []

        if compute_mmd or two_sample_test_params is not None:
            random_proj_x = SparseRandomProjection(n_components= rand_proj_dims)
            proj_train_x = random_proj_x.fit_transform(train_x.reshape(-1, self.experiment.dim_x))

        for i in range(repeats):
            print(f"Repeat {i+1}/{repeats}...")
            # sample from training data
            idx_rnd = np.random.randint(0, train_size, size=sample_size)
            idx_rnd_2 = np.random.randint(0, train_size, size=sample_size)
            train_x_sample = train_x[idx_rnd, :, :, :].reshape(-1, self.experiment.dim_x) \
                if len(train_x.shape) > 2 else train_x[idx_rnd, :].reshape(-1, self.experiment.dim_x)

            train_x_sample_2 = train_x[idx_rnd_2, :, :, :].reshape(-1, self.experiment.dim_x) \
                if len(train_x.shape) > 2 else train_x[idx_rnd_2, :].reshape(-1, self.experiment.dim_x)

            # sample from generated data
            idx_rnd_gen = np.random.randint(0, gen_size, size=sample_size)
            generated_x_sample = generated_x[idx_rnd_gen, :]

            if compute_mmd or two_sample_test_params is not None:
                proj_train_x_sample = proj_train_x[idx_rnd, :]
                proj_train_x_sample_2 = proj_train_x[idx_rnd_2, :]
                proj_gen_x_sample = random_proj_x.transform(generated_x_sample)

            if compute_mmd:
                print("computing MMD distances on X data...")

                if not compute_only_with_rand_proj:
                    #compute reference
                    mmd_x_refs.append(mmd(train_x_sample, train_x_sample_2, **mmd_params)[0])
                    mmd_x_dists.append(mmd(train_x_sample, generated_x_sample, **mmd_params)[0])
                else:
                    mmd_x_refs = mmd_x_dists = None

                mmd_proj_x_refs.append(mmd(proj_train_x_sample, proj_train_x_sample_2, **mmd_params)[0])
                mmd_proj_x_dists.append(mmd(proj_train_x_sample, proj_gen_x_sample, **mmd_params)[0])

            else:
                mmd_x_dists = mmd_proj_x_dists = mmd_x_refs = mmd_proj_x_refs = None

            if two_sample_test_params is not None:
                print("computing two-sample MMD test on X data...")

                if not compute_only_with_rand_proj:
                    # store H0 rejection decision
                    two_sample_tests_x.append(two_sample_mmd_test(train_x_sample, generated_x_sample, **two_sample_test_params)[2])
                else:
                    two_sample_tests_x = None

                two_sample_tests_proj_x.append(
                    two_sample_mmd_test(proj_train_x_sample, proj_gen_x_sample, **two_sample_test_params)[2])

            else:
                two_sample_tests_x = two_sample_tests_proj_x = None

        mmd_x_dists = np.array(mmd_x_dists) if mmd_x_dists is not None else None
        mmd_proj_x_dists = np.array(mmd_proj_x_dists) if mmd_proj_x_dists is not None else None

        mmd_x_refs = np.array(mmd_x_refs) if mmd_x_refs is not None else None
        mmd_proj_x_refs = np.array(mmd_proj_x_refs) if mmd_proj_x_refs is not None else None

        two_sample_tests_x = np.array(two_sample_tests_x) if two_sample_tests_x is not None else None
        two_sample_tests_proj_x = np.array(two_sample_tests_proj_x) if two_sample_tests_proj_x is not None else None

        # store in results_vecs
        self.results_vecs["mmd_x_dists"] = mmd_x_dists
        self.results_vecs["mmd_proj_x_dists"] = mmd_proj_x_dists
        self.results_vecs["two_sample_tests_x"] = two_sample_tests_x
        self.results_vecs["two_sample_tests_proj_x"] = two_sample_tests_proj_x

        # compute proportion of H0 rejection decisions
        if two_sample_test_params is not None:
            prop_reject_two_sample_x = np.mean(two_sample_tests_x) if two_sample_tests_x is not None else None
            prop_reject_two_sample_proj_x = np.mean(two_sample_tests_proj_x)
        else:
            prop_reject_two_sample_x = prop_reject_two_sample_proj_x = None

        write_prop_reject_two_sample_x = f"{prop_reject_two_sample_x:.{precision}f}" \
            if prop_reject_two_sample_x is not None else None
        write_prop_reject_two_sample_x_txt = "Proportion of H0 rejections (samples from same dist.) for two-sample MMD test on x data (train vs gen):"
        write_header_prop_reject_two_sample_x = "prop_reject_two_sample_x"

        write_prop_reject_two_sample_proj_x = f"{prop_reject_two_sample_proj_x:.{precision}f}" \
            if prop_reject_two_sample_proj_x is not None else None
        write_prop_reject_two_sample_proj_x_txt = "Proportion of H0 rejections (samples from same dist.) for two-sample MMD test on Rand. Proj. x data (train vs gen):"
        write_header_prop_reject_two_sample_proj_x = "prop_reject_two_sample_proj_x"

        # run stats & prepare logging strings
        mmd_x_dists_stats = self.compute_array_stats(mmd_x_dists)
        mmd_proj_x_dists_stats = self.compute_array_stats(mmd_proj_x_dists)

        mmd_x_refs_stats = self.compute_array_stats(mmd_x_refs)
        mmd_proj_x_refs_stats = self.compute_array_stats(mmd_proj_x_refs)

        write_data_mmd_x = self.make_array_stats_string(mmd_x_dists_stats, precision)
        write_data_mmd_x_txt = f"MMD distances between training and generated x data (est. with sample size={sample_size}):"
        write_header_mmd_x = "mmd_x_mean, mmd_x_median, [(mmd_x_q25, mmd_x_q75), (mmd_x_q025, mmd_x_q975)]"

        write_data_mmd_proj_x = self.make_array_stats_string(mmd_proj_x_dists_stats, precision)
        write_data_mmd_proj_x_txt = f"MMD distances between training and generated, Rand. projected, x data (est. with sample size={sample_size}):"
        write_header_mmd_proj_x = "mmd_proj_x_mean, mmd_proj_x_median, [(mmd_proj_x_q25, mmd_proj_x_q75), (mmd_proj_x_q025, mmd_proj_x_q975)]"

        write_data_mmd_x_ref = self.make_array_stats_string(mmd_x_refs_stats, precision)
        write_data_mmd_x_ref_txt = f"Ref-MMD distances between two training x data samples (est. with sample size={sample_size}):"
        write_header_mmd_x_ref = f"mmd_x_ref_mean, mmd_x_ref_median, [(mmd_x_ref_q25, mmd_x_ref_q75), (mmd_x_ref_q025, mmd_x_ref_q975)]"

        write_data_mmd_proj_x_ref = self.make_array_stats_string(mmd_proj_x_refs_stats, precision)
        write_data_mmd_proj_x_ref_txt = f"Ref-MMD distances between two training, Rand. projected, x data samples (est. with sample size={sample_size}):"
        write_header_mmd_proj_x_ref = f"mmd_proj_x_ref_mean, mmd_proj_x_ref_median, [(mmd_proj_x_ref_q25, mmd_proj_x_ref_q75), (mmd_proj_x_ref_q025, mmd_proj_x_ref_q975)]"

        return {
            "data": {
                "write_data_mmd_x": (write_data_mmd_x, write_data_mmd_x_txt),
                "write_data_mmd_proj_x": (write_data_mmd_proj_x, write_data_mmd_proj_x_txt),
                "write_prop_reject_two_sample_x": (write_prop_reject_two_sample_x, write_prop_reject_two_sample_x_txt),
                "write_prop_reject_two_sample_proj_x": (write_prop_reject_two_sample_proj_x, write_prop_reject_two_sample_proj_x_txt),
                "write_data_mmd_x_ref": (write_data_mmd_x_ref, write_data_mmd_x_ref_txt),
                "write_data_mmd_proj_x_ref": (write_data_mmd_proj_x_ref, write_data_mmd_proj_x_ref_txt),
            },
            "headers": {
                "write_header_mmd_x": write_header_mmd_x,
                "write_header_mmd_proj_x": write_header_mmd_proj_x,
                "write_header_prop_reject_two_sample_x": write_header_prop_reject_two_sample_x,
                "write_header_prop_reject_two_sample_proj_x": write_header_prop_reject_two_sample_proj_x,
                "write_header_mmd_x_ref": write_header_mmd_x_ref,
                "write_header_mmd_proj_x_ref": write_header_mmd_proj_x_ref,
            }
        }

    def make_recons_stats(self, build_y_stats = True, make_plots = True):
        """
        Make mean, std and iqr statistics for all data.

        :param build_y_stats: if True, build mean, std and iqr statistics for y data
        :param make_plots: if True, make plots of the statistics
        :return: dictionary of statistics formatted to be logged by write_results()
        """
        print("Making reconstruction statistics...")

        dim_x = self.experiment.dim_x

        precision = self.round_digits

        if self.train_x is not None and self.train_y is not None:

            recon_train_x_rmse_vec = rmse_torch(self.reconstructed_train_x.view(-1, dim_x), self.train_x.view(-1, dim_x)).numpy()
            self.results_vecs["recon_train_x_rmse_vec"] = recon_train_x_rmse_vec

            recon_train_x_rmse_vec_stats = self.compute_array_stats(recon_train_x_rmse_vec)

            write_data_recon_train_x = self.make_array_stats_string(recon_train_x_rmse_vec_stats, precision)
            write_data_recon_train_x_txt = "Reconstructed training models RMSE vs training models: "
            write_header_recon_train_x = "mean_recon_train_x_rmse,median_recon_train_x_rmse,(q25_recon_train_x_rmse,q75_recon_train_x_rmse),(q025_recon_train_x_rmse,q975_recon_train_x_rmse)"

            if build_y_stats:
                # compute proportion of concordence between reconstructed_train_y_label and train_y_label
                recon_y_label_accuracy = compare_labels(self.train_y_label, self.reconstructed_train_y_label)
            else:
                recon_y_label_accuracy = None

            self.results_vecs["recon_y_label_accuracy"] = recon_y_label_accuracy

            write_data_recon_train_y = f'{self.results_vecs["recon_y_label_accuracy"]}'
            write_data_recon_train_y_txt = "Reconstructed training label vs training label accuracy: "
            write_header_recon_train_y = "recon_train_y_label_accuracy"

        else:
            raise ValueError("No training data to compute mean and std statistics."
                             "Load data into the experiment object first.")

        if self.val_x is not None and self.val_y is not None:

            recon_val_x_rmse_vec = rmse_torch(self.reconstructed_val_x.view(-1, dim_x), self.val_x.view(-1, dim_x)).numpy()
            self.results_vecs["recon_val_x_rmse_vec"] = recon_val_x_rmse_vec

            recon_val_x_rmse_vec_stats = self.compute_array_stats(recon_val_x_rmse_vec)

            write_data_recon_val_x = self.make_array_stats_string(recon_val_x_rmse_vec_stats, precision)
            write_data_recon_val_x_txt = "Reconstructed validation models RMSE vs validation models: "
            write_header_recon_val_x = "mean_recon_val_x_rmse,median_recon_val_x_rmse,(q25_recon_val_x_rmse,q75_recon_val_x_rmse),(q025_recon_val_x_rmse,q975_recon_val_x_rmse)"

            if build_y_stats:
                # compute proportion of concordence between reconstructed_val_y_label and val_y_label
                recon_val_y_label_accuracy = compare_labels(self.val_y_label, self.reconstructed_val_y_label)
            else:
                recon_val_y_label_accuracy = None

            self.results_vecs["recon_val_y_label_accuracy"] = recon_val_y_label_accuracy

            write_data_recon_val_y = f'{self.results_vecs["recon_val_y_label_accuracy"]}'
            write_data_recon_val_y_txt = "Reconstructed validation label vs validation label accuracy:"
            write_header_recon_val_y = "recon_val_y_label_accuracy"

        else:
            raise ValueError("No validation data to compute mean and std statistics."
                             "Load data into the experiment object first.")

        if make_plots:
            self.plot_true_vs_recon(data_type='train')
            self.plot_true_vs_recon(data_type='val')
            self.plot_recon_rmse_hists(joint_output= False)

        return {
            "data": {
                "write_data_recon_train_x": (write_data_recon_train_x, write_data_recon_train_x_txt),
                "write_data_recon_val_x": (write_data_recon_val_x, write_data_recon_val_x_txt),
                "write_data_recon_train_y": (write_data_recon_train_y, write_data_recon_train_y_txt),
                "write_data_recon_val_y": (write_data_recon_val_y, write_data_recon_val_y_txt),
                },
            "headers": {
                "write_header_recon_train_x": write_header_recon_train_x,
                "write_header_recon_val_x": write_header_recon_val_x,
                "write_header_recon_train_y": write_header_recon_train_y,
                "write_header_recon_val_y": write_header_recon_val_y,
                }
        }

    def plot_recon_rmse_hists(self, joint_output = True, dpi = 600, show = False, **kwargs):
        """
        Plot histograms of reconstruction rmse.
        :param joint_output: whether the model being diagnosed is a joint distribution model. If true, both x and y data will be plotted.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to plot_histograms()
        """
        print("Plotting reconstruction rmse histograms...")

        colors = ['lightseagreen', 'darkgreen']
        labels = ['Train', 'Validation']

        x_recon_rmse = np.concatenate([self.results_vecs["recon_train_x_rmse_vec"].reshape(1, -1),
                                     self.results_vecs["recon_val_x_rmse_vec"].reshape(1, -1)], axis=0)

        plot.plot_histograms(x_recon_rmse, labels, colors = colors, plot_title= 'Reconstruction RMSE on models (ns/m)',
                             save_location= f"{self.reconstructions_diag_dir}/models_recon_rmse.pdf", dpi = dpi, show=show, **kwargs)

        if joint_output:
            y_recon_rmse = np.concatenate([self.results_vecs["recon_train_y_rmse_vec"].reshape(1, -1),
                                         self.results_vecs["recon_val_y_rmse_vec"].reshape(1, -1)], axis=0)

            plot.plot_histograms(y_recon_rmse, labels, colors = colors, plot_title= 'Reconstruction RMSE on travel times (ns)',
                                 save_location= f"{self.reconstructions_diag_dir}/TT_recon_rmse.pdf", dpi = dpi, show=show, **kwargs)

    def plot_true_vs_recon(self, data_type='train', grid_size = 4, dpi=600, show=False):
        """
        Select random samples and plot true vs reconstructed data on a grid. Label each image with its reconstruction RMSE and true label vs reconstructed label.
        :param data_type: 'train' or 'val' to select training or validation data
        :param grid_size: size of the grid (grid_size x grid_size). Total number of samples plotted = grid_size^2
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        """
        grid_size = grid_size if grid_size is not None else 4

        num_samples = grid_size * grid_size
        total_sample_size = self.train_x.shape[0] if data_type == 'train' else self.val_x.shape[0]
        random_indices = torch.randperm(total_sample_size)[:num_samples]
        if data_type == 'train':
            x_data = self.train_x[random_indices]
            recon_x_data = self.reconstructed_train_x[random_indices]
            recon_y_label = self.reconstructed_train_y_label[random_indices]
            true_y_label = self.train_y_label[random_indices]
        else:
            x_data = self.val_x[random_indices]
            recon_x_data = self.reconstructed_val_x[random_indices]
            recon_y_label = self.reconstructed_val_y_label[random_indices]
            true_y_label = self.val_y_label[random_indices]

        dim_x = self.experiment.dim_x
        rmse_x = rmse_torch(recon_x_data.view(-1, dim_x), x_data.view(-1, dim_x)).numpy()

        x_data = x_data.squeeze()
        recon_x_data = recon_x_data.squeeze()
        true_y_label = true_y_label.unsqueeze(1)
        recon_y_label = recon_y_label.unsqueeze(1)

        # combine x_data and recon_x_data to form tensor of shape (num_samples *2, C, H, W) for plotting
        x_plot_data = torch.cat([x_data, recon_x_data], dim=0).numpy()

        # combine true_y_label and recon_y_label to form tensor of shape (num_samples *2, 1) for plotting
        y_plot_data = torch.cat([true_y_label, recon_y_label], dim=0).numpy()

        plot.plot_true_vs_recon_grid(x_plot_data, rmse_x, y_plot_data, num_samples,
                                     save_location=f"{self.reconstructions_diag_dir}/recon_examples.pdf",
                                     dpi=dpi, show=show)

    def assess_forward_learning(self, make_plots = True, dpi=600, show = False):
        """
        Assess how well the forward model is learned by comparing generated Gen_X passed through the forward model F(Gen_X) vs associated generated labels Gen_Y by the decoder,
        as well as reconstructed Recon_X passed through the forward model F(Recon_X) vs reconstructed labels Recon_Y by the decoder.
        Plot examples of generated data Gen_X and their generated Gen_Y vs real forward outputs F(Gen_X),
        as well as reconstructed data Recon_X and their reconstructed Recon_Y vs real forward outputs F(Recon_X).
        In the case of image data (e.g., MNIST), the forward is the classifier used to label the data.
        :param make_plots: whether to make plots of the assessments.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :return: dictionary of statistics formatted to be logged by write_results()
        """

        print("Assessing forward model learning...")

        dim_x = self.experiment.dim_x

        if self.train_x is not None and self.train_y is not None:
            resimulated_recon_train_x_label_accuracy = compare_labels(self.resimulated_recon_train_x_label, self.reconstructed_train_y_label)
            self.results_vecs["resimulated_recon_train_x_label_accuracy"] = resimulated_recon_train_x_label_accuracy

            write_data_resimulated_recon_train_x = f'{self.results_vecs["resimulated_recon_train_x_label_accuracy"]}'
            write_data_resimulated_recon_train_x_txt = "(Train) Resimulated reconstructed image labels vs reconstructed label accuracy: "
            write_header_resimulated_recon_train_x = "resimulated_recon_train_x_label_accuracy"

        else:
            raise ValueError("No training data to compute forward model learning assessment."
                             "Load data into the experiment object first.")

        if self.val_x is not None and self.val_y is not None:
            resimulated_recon_val_x_label_accuracy = compare_labels(self.resimulated_recon_val_x_label, self.reconstructed_val_y_label)
            self.results_vecs["resimulated_recon_val_x_label_accuracy"] = resimulated_recon_val_x_label_accuracy

            write_data_resimulated_recon_val_x = f'{self.results_vecs["resimulated_recon_val_x_label_accuracy"]}'
            write_data_resimulated_recon_val_x_txt = "(Val) Resimulated reconstructed image labels vs reconstructed label accuracy: "
            write_header_resimulated_recon_val_x = "resimulated_recon_val_x_label_accuracy"

        else:
            raise ValueError("No validation data to compute forward model learning assessment."
                             "Load data into the experiment object first.")

        # compare resimulated_generated_x_label to train_y_label
        resimulated_generated_x_label_accuracy = compare_labels(self.resimulated_generated_x_label, self.generated_y_label)
        self.results_vecs["resimulated_generated_x_label_accuracy"] = resimulated_generated_x_label_accuracy

        write_data_resimulated_generated_x = f'{self.results_vecs["resimulated_generated_x_label_accuracy"]}'
        write_data_resimulated_generated_x_txt = "Resimulated generated image vs generated label accuracy: "
        write_header_resimulated_generated_x = "resimulated_generated_x_label_accuracy"

        # plot examples of generated data and their generated forward outputs vs real forward outputs
        # as well as reconstructed data (train and val) and their reconstructed forward outputs vs real forward outputs
        num_samples = 16
        total_train_size = self.train_x.shape[0]
        total_val_size = self.val_x.shape[0]
        random_indices_train = torch.randperm(total_train_size)[:num_samples]
        random_indices_val = torch.randperm(total_val_size)[:num_samples]
        random_indices_gen = torch.randperm(self.generated_x.shape[0])[:num_samples]

        gen_x_data = self.generated_x[random_indices_gen].squeeze() # generated data
        gen_y_data_label = self.generated_y_label[random_indices_gen].unsqueeze(1) # generated data labels
        resim_gen_x_data_label = self.resimulated_generated_x_label[random_indices_gen].unsqueeze(1) # resimulated generated data labels

        train_x_data = self.train_x[random_indices_train].squeeze() # true train data
        recon_train_x_data = self.reconstructed_train_x[random_indices_train].squeeze() # reconstructed train data
        true_train_y_label = self.train_y_label[random_indices_train].unsqueeze(1) # true train data labels
        resim_recon_train_x_data_label = self.resimulated_recon_train_x_label[random_indices_train].unsqueeze(1) # resimulated reconstructed train data labels
        rmse_train_x = rmse_torch(recon_train_x_data.view(-1, dim_x), train_x_data.view(-1, dim_x)).numpy()

        val_x_data = self.val_x[random_indices_val].squeeze() # true val data
        recon_val_x_data = self.reconstructed_val_x[random_indices_val].squeeze() # reconstructed val data
        true_val_y_label = self.val_y_label[random_indices_val].unsqueeze(1) # true val data labels
        resim_recon_val_x_data_label = self.resimulated_recon_val_x_label[random_indices_val].unsqueeze(1) # resimulated reconstructed val data labels
        rmse_val_x = rmse_torch(recon_val_x_data.view(-1, dim_x), val_x_data.view(-1, dim_x)).numpy()

        # concatenate true and reconstruction data for plotting
        train_plot_x_data = torch.cat([train_x_data, recon_train_x_data], dim=0).numpy()
        val_plot_x_data = torch.cat([val_x_data, recon_val_x_data], dim=0).numpy()
        gen_plot_x_data = gen_x_data.numpy()

        # concatenate labels for plotting
        train_plot_y_data = torch.cat([true_train_y_label, resim_recon_train_x_data_label], dim=0).numpy()
        val_plot_y_data = torch.cat([true_val_y_label, resim_recon_val_x_data_label], dim=0).numpy()
        gen_plot_y_data = torch.cat([gen_y_data_label, resim_gen_x_data_label], dim=1).numpy()

        if make_plots:
            plot.plot_true_vs_recon_grid(train_plot_x_data, rmse_train_x, train_plot_y_data, num_samples,
                                         save_location=f"{self.resimulations_diag_dir}/Resim_recon_train_examples.pdf",
                                         dpi=dpi, show=show)
            plot.plot_true_vs_recon_grid(val_plot_x_data, rmse_val_x, val_plot_y_data, num_samples,
                                         save_location=f"{self.resimulations_diag_dir}/Resim_recon_val_examples.pdf",
                                         dpi=dpi, show=show)

            plot.plot_samples_grid(gen_plot_x_data, gen_plot_y_data, num_samples,
                                   save_location=f"{self.resimulations_diag_dir}/Resim_generated_examples.pdf",
                                   dpi=dpi, show=show)

        return {
            "data": {
                "write_data_resimulated_recon_train_x": (write_data_resimulated_recon_train_x, write_data_resimulated_recon_train_x_txt),
                "write_data_resimulated_recon_val_x": (write_data_resimulated_recon_val_x, write_data_resimulated_recon_val_x_txt),
                "write_data_resimulated_generated_x": (write_data_resimulated_generated_x, write_data_resimulated_generated_x_txt),
            },
            "headers": {
                "write_header_resimulated_recon_train_x": write_header_resimulated_recon_train_x,
                "write_header_resimulated_recon_val_x": write_header_resimulated_recon_val_x,
                "write_header_resimulated_generated_x": write_header_resimulated_generated_x,
            }
        }

    def inspect_latent_distribution(self, make_plots = True, n_neighbors = 100,  min_dist = 0.3, fit_prior_only = False, denseMAP = False,
                                    mmd_params=None, two_sample_test_params = None, test_repeats = 100, test_sample_size = 300):
        """
        Inspect latent space distribution.
        :param make_plots: if True, make plots of the latent distribution inspections
        :param n_neighbors: number of neighbors to consider for UMAP embedding
        :param min_dist: minimum distance between points in UMAP embedding
        :param two_sample_test_params: parameters for the two sample MMD test
        :param test_repeats: number of repeats for the two sample MMD test
        :param test_sample_size: sample size for the two sample MMD test
        :return: dictionary of statistics formatted to be logged by write_results()
        """
        print("Inspecting latent space distribution...")

        precision = self.round_digits

        z_dist = self.experiment.latent_dist
        z_dist_params = self.experiment.latent_dist_params_list
        latent_dim = self.experiment.latent_dim

        sample_size = test_sample_size

        self.latent_prior_mean = np.mean(self.latent_vector.numpy(), axis=0)
        self.latent_train_codes_mean = np.mean(self.latent_train_codes.numpy(), axis=0)
        self.latent_val_codes_mean = np.mean(self.latent_val_codes.numpy(), axis=0)

        self.latent_prior_covm = np.cov(self.latent_vector.numpy(), rowvar=False)
        self.latent_train_codes_covm = np.cov(self.latent_train_codes.numpy(), rowvar=False)
        self.latent_val_codes_covm = np.cov(self.latent_val_codes.numpy(), rowvar=False)

        self.covm_diff_prior_train = np.abs(self.latent_prior_covm - self.latent_train_codes_covm)
        self.covm_diff_prior_val = np.abs(self.latent_prior_covm - self.latent_val_codes_covm)


        # multivariate normality test
        if self.experiment.latent_dist_name == "standardnormal" or self.experiment.latent_dist_name == "normal":
            # test multivariate normality (Henze-Zirkler test)
            import pingouin as pg
            latent_train_mvn_test = pg.multivariate_normality(self.latent_train_codes, alpha=0.05)
            latent_val_mvn_test = pg.multivariate_normality(self.latent_val_codes, alpha=0.05)
        else:
            latent_train_mvn_test = latent_val_mvn_test = None

        # two sample MMD test
        if two_sample_test_params is not None:
            # set kernel parameter
            if self.mmd_est_params['z'] is None:
                self.estimate_mmd_params(which_variable='z', max_sample_size=10000)
            self.two_sample_test_params['kernel_params']['gamma'] = self.mmd_est_params['z']
            self.mmd_params['kernel_params']['gamma'] = self.mmd_est_params['z']
            print(f"Using estimated MMD Z kernel parameter gamma: {self.mmd_est_params['z']}")

            latent_train_mmd_two_sample_tests_pvalues = []
            latent_train_mmd_two_sample_tests_rejections = []
            latent_val_mmd_two_sample_tests_pvalues = []
            latent_val_mmd_two_sample_tests_rejections = []

            latent_train_mmd_values_vec = []
            latent_val_mmd_values_vec = []

            latent_mmd_ref = []

            for i in range(test_repeats):
                print(f"Repeat {i + 1}/{test_repeats}...")

                latent_vector = (
                    z_dist(
                        torch.tensor(z_dist_params[0], dtype=torch.float32),
                        torch.tensor(z_dist_params[1], dtype=torch.float32),
                    )
                    .sample((sample_size, latent_dim))
                    .cpu()
                ).numpy()

                latent_vector_2 = (
                    z_dist(
                        torch.tensor(z_dist_params[0], dtype=torch.float32),
                        torch.tensor(z_dist_params[1], dtype=torch.float32),
                    )
                    .sample((sample_size, latent_dim))
                    .cpu()
                ).numpy()

                idx_train = np.random.choice(self.latent_train_codes.shape[0], sample_size, replace=False)
                latent_train_codes = self.latent_train_codes[idx_train, :].numpy()
                idx_val = np.random.choice(self.latent_val_codes.shape[0], sample_size, replace=False)
                latent_val_codes = self.latent_val_codes[idx_val, :].numpy()

                train_test = two_sample_mmd_test(latent_vector, latent_train_codes,
                                                               **two_sample_test_params)
                val_test = two_sample_mmd_test(latent_vector, latent_val_codes,
                                                             **two_sample_test_params)

                latent_train_mmd_two_sample_tests_pvalues.append(train_test[1])
                latent_train_mmd_two_sample_tests_rejections.append(train_test[2])

                latent_val_mmd_two_sample_tests_pvalues.append(val_test[1])
                latent_val_mmd_two_sample_tests_rejections.append(val_test[2])

                if mmd_params is None:
                    latent_train_mmd_values_vec.append(train_test[0])
                    latent_val_mmd_values_vec.append(val_test[0])
                    mmd_params = {"kernel_params": self.two_sample_test_params['kernel_params'].copy(), "unbiased": False}
                else:
                    latent_train_mmd_values_vec.append(mmd(latent_vector, latent_train_codes, **mmd_params)[0])
                    latent_val_mmd_values_vec.append(mmd(latent_vector, latent_val_codes, **mmd_params)[0])

                latent_mmd_ref.append(mmd(latent_vector, latent_vector_2, **mmd_params)[0])

            latent_train_mmd_values_vec = np.array(latent_train_mmd_values_vec)
            latent_val_mmd_values_vec = np.array(latent_val_mmd_values_vec)
            latent_mmd_ref = np.array(latent_mmd_ref)
        else:
            latent_train_mmd_two_sample_tests_pvalues = latent_val_mmd_two_sample_tests_pvalues = \
                latent_train_mmd_values_vec = latent_train_mmd_two_sample_tests_rejections = latent_val_mmd_two_sample_tests_rejections =\
                latent_val_mmd_values_vec = latent_mmd_ref = None

        self.results_vecs["latent_train_mmd_two_sample_tests_pvalues"] = latent_train_mmd_two_sample_tests_pvalues
        self.results_vecs["latent_train_mmd_two_sample_tests_rejections"] = latent_train_mmd_two_sample_tests_rejections
        self.results_vecs["latent_val_mmd_two_sample_tests_pvalues"] = latent_val_mmd_two_sample_tests_pvalues
        self.results_vecs["latent_val_mmd_two_sample_tests_rejections"] = latent_val_mmd_two_sample_tests_rejections
        self.results_vecs["latent_train_mmd_values"] = latent_train_mmd_values_vec
        self.results_vecs["latent_val_mmd_values"] = latent_val_mmd_values_vec

        write_data_train_mvn_test = str(latent_train_mvn_test) if latent_train_mvn_test is not None else 'N/A' # ['hz', 'pval', 'normal'?]
        write_data_train_mvn_test_txt = "'Multivariate normal test - training latent codes : "
        write_header_train_mvn_test = "latent_train_mvn_test"

        write_data_val_mvn_test = str(latent_val_mvn_test) if latent_val_mvn_test is not None else 'N/A' # ['hz', 'pval', 'normal'?]
        write_data_val_mvn_test_txt = "'Multivariate normal test - validation latent codes : "
        write_header_val_mvn_test = "latent_val_mvn_test"

        # compute combined p-value
        # if testing was done, combine p-values in chi-square statistic
        from scipy.stats import chi2
        if latent_train_mmd_two_sample_tests_pvalues is not None:
            combined_stat_train = -2 * np.sum(np.log(latent_train_mmd_two_sample_tests_pvalues))
            combined_p_value_train = 1 - chi2.cdf(combined_stat_train, 2 * test_repeats)

            rejection_prop_train = np.mean(np.array(latent_train_mmd_two_sample_tests_rejections))

            combined_stat_val = -2 * np.sum(np.log(latent_val_mmd_two_sample_tests_pvalues))
            combined_p_value_val = 1 - chi2.cdf(combined_stat_val, 2 * test_repeats)

            rejection_prop_val = np.mean(np.array(latent_val_mmd_two_sample_tests_rejections))
        else:
            combined_p_value_train = combined_p_value_val = rejection_prop_train = rejection_prop_val = None

        write_combined_pvalue_train_mmd_two_sample_test = f"{combined_p_value_train:.{precision}f}" \
            if combined_p_value_train is not None else None
        write_combined_pvalue_train_mmd_two_sample_test_txt = f"Combined p-value with Fisher (chi-2) with {test_repeats} of two sample MMD test - training latent codes: "
        write_header_combined_pvalue_train_mmd_two_sample_test = "combined_pvalue_train_mmd_two_sample_test"

        write_rejection_prop_train_mmd_two_sample_test = f"{rejection_prop_train:.{precision}f}" \
            if rejection_prop_train is not None else None
        write_rejection_prop_train_mmd_two_sample_test_txt = f"H0 rejection proportion of {test_repeats} of two sample MMD test - training latent codes: "
        write_header_rejection_prop_train_mmd_two_sample_test = "rejection_prop_train_mmd_two_sample_test"

        write_combined_pvalue_val_mmd_two_sample_test = f"{combined_p_value_val:.{precision}f}" \
            if combined_p_value_val is not None else None
        write_combined_pvalue_val_mmd_two_sample_test_txt = f"Combined p-value with Fisher (chi-2) with {test_repeats} of two sample MMD test - validation latent codes: "
        write_header_combined_pvalue_val_mmd_two_sample_test = "combined_pvalue_val_mmd_two_sample_test"

        write_rejection_prop_val_mmd_two_sample_test = f"{rejection_prop_val:.{precision}f}" \
            if rejection_prop_val is not None else None
        write_rejection_prop_val_mmd_two_sample_test_txt = f"H0 rejection proportion of {test_repeats} of two sample MMD test - validation latent codes: "
        write_header_rejection_prop_val_mmd_two_sample_test = "rejection_prop_val_mmd_two_sample_test"

        latent_train_mmd_values_stats = self.compute_array_stats(latent_train_mmd_values_vec)
        write_data_train_mmd_values = self.make_array_stats_string(latent_train_mmd_values_stats, precision)
        write_data_train_mmd_values_txt = f"MMD values - training latent codes estimated with {test_sample_size * test_repeats}: "
        write_header_train_mmd_values = ("mean_latent_train_mmd_values,median_latent_train_mmd_values,"
                                            "[(q25_latent_train_mmd_values,q75_latent_train_mmd_values),"
                                            "(q025_latent_train_mmd_values,q975_latent_train_mmd_values)]")

        latent_val_mmd_values_stats = self.compute_array_stats(latent_val_mmd_values_vec)
        write_data_val_mmd_values = self.make_array_stats_string(latent_val_mmd_values_stats, precision)
        write_data_val_mmd_values_txt = f"MMD values - validation latent codes {test_sample_size * test_repeats}: "
        write_header_val_mmd_values = ("mean_latent_val_mmd_values,median_latent_val_mmd_values,"
                                        "[(q25_latent_val_mmd_values,q75_latent_val_mmd_values),"
                                        "(q025_latent_val_mmd_values,q975_latent_val_mmd_values)]")

        latent_mmd_ref_values_stats = self.compute_array_stats(latent_mmd_ref)
        write_data_mmd_ref_values = self.make_array_stats_string(latent_mmd_ref_values_stats, precision)
        write_data_mmd_ref_values_txt = "Ref - MMD values latent space: "
        write_header_mmd_ref_values = ("mean_mmd_ref_values,median_mmd_ref_values,"
                                        "[(q25_mmd_ref_values,q75_mmd_ref_values),"
                                        "(q025_mmd_ref_values,q975_mmd_ref_values)]")

        if make_plots:
            self.plot_latent_cov_matrices()
            self.plot_latent_histograms()
            self.plot_latent_umap_tsne_scatters(n_neighbors=n_neighbors, min_dist=min_dist, denseMAP=denseMAP, fit_prior_only=fit_prior_only)

        return {
            "data":{
                "write_data_train_mvn_test": (write_data_train_mvn_test, write_data_train_mvn_test_txt),
                "write_data_val_mvn_test": (write_data_val_mvn_test, write_data_val_mvn_test_txt),
                "write_combined_pvalue_train_mmd_two_sample_test": (write_combined_pvalue_train_mmd_two_sample_test, write_combined_pvalue_train_mmd_two_sample_test_txt),
                "write_rejection_prop_train_mmd_two_sample_test":(write_rejection_prop_train_mmd_two_sample_test, write_rejection_prop_train_mmd_two_sample_test_txt),
                "write_data_train_mmd_values": (write_data_train_mmd_values, write_data_train_mmd_values_txt),
                "write_combined_pvalue_val_mmd_two_sample_test": (write_combined_pvalue_val_mmd_two_sample_test, write_combined_pvalue_val_mmd_two_sample_test_txt),
                "write_rejection_prop_val_mmd_two_sample_test":(write_rejection_prop_val_mmd_two_sample_test, write_rejection_prop_val_mmd_two_sample_test_txt),
                "write_data_val_mmd_values": (write_data_val_mmd_values, write_data_val_mmd_values_txt),
                "write_data_latent_mmd_ref_values": (write_data_mmd_ref_values, write_data_mmd_ref_values_txt),
            },
            "headers": {
                "write_header_train_mvn_test": write_header_train_mvn_test,
                "write_header_val_mvn_test": write_header_val_mvn_test,
                "write_header_combined_pvalue_train_mmd_two_sample_test": write_header_combined_pvalue_train_mmd_two_sample_test,
                "write_header_rejection_prop_train_mmd_two_sample_test": write_header_rejection_prop_train_mmd_two_sample_test,
                "write_header_train_mmd_values": write_header_train_mmd_values,
                "write_header_combined_pvalue_val_mmd_two_sample_test": write_header_combined_pvalue_val_mmd_two_sample_test,
                "write_header_rejection_prop_val_mmd_two_sample_test": write_header_rejection_prop_val_mmd_two_sample_test,
                "write_header_val_mmd_values": write_header_val_mmd_values,
                "write_header_mmd_ref_values": write_header_mmd_ref_values,
            }
        }

    def plot_latent_cov_matrices(self, dpi = 600, show = False, **kwargs):
        """
        Plot covariance matrices of latent codes.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to base_config()
        """
        print("Plotting latent covariance matrices...")

        vmin = min(np.min(self.latent_prior_covm), np.min(self.latent_train_codes_covm), np.min(self.latent_val_codes_covm))
        vmax = max(np.max(self.latent_prior_covm), np.max(self.latent_train_codes_covm), np.max(self.latent_val_codes_covm))

        plot.plot_cov(self.latent_prior_covm, plot_title='Latent prior covariance matrix', vmin_vmax=(vmin, vmax),
                      save_location=f"{self.latent_space_diag_dir}/covm_latent_prior.pdf", dpi=dpi, show=show, **kwargs)

        plot.plot_cov(self.latent_train_codes_covm, plot_title = 'Training latent codes covariance matrix', vmin_vmax=(vmin, vmax),
                      save_location=f"{self.latent_space_diag_dir}/covm_latent_train_codes.pdf", dpi=dpi, show=show, **kwargs)

        plot.plot_cov(self.latent_val_codes_covm, plot_title ='Validation latent codes covariance matrix', vmin_vmax=(vmin, vmax),
                      save_location=f"{self.latent_space_diag_dir}/covm_latent_val_codes.pdf", dpi=dpi, show=show, **kwargs)

        plot.plot_cov(self.covm_diff_prior_train, plot_title='Prior covm - Train latent codes covm',
                      save_location=f"{self.latent_space_diag_dir}/diff_priorcovm_train_codes.pdf", dpi=dpi, show=show, **kwargs)

        plot.plot_cov(self.covm_diff_prior_val, plot_title='Prior covm - Val latent codes covm',
                      save_location=f"{self.latent_space_diag_dir}/diff_priorcovm_val_codes.pdf", dpi=dpi, show=show, **kwargs)

    def plot_latent_histograms(self, nl = 6, nco = 5, dpi = 600, show = False, **kwargs):
        print("Plotting latent histograms...")

        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config(mpl, **kwargs)

        def plot_hist(codes, file_name_key, save_location_dir):
            latent_dim = codes.shape[1]

            nb_img = int(np.ceil(latent_dim / 30))

            nline = min(6, nl) if nl is not None else 6
            ncol = min(5, nco) if nco is not None else 5

            for id in range(nb_img):
                fig, axes = plt.subplots(nline, ncol, figsize=(20, 20))
                for i in range(id * 30, (id + 1) * 30):
                    if i >= latent_dim:
                        break
                    ref = (i-id*30)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].hist(codes[:, i].reshape(-1).numpy(),
                                                                       bins=30, rwidth=0.8, color='grey', density=True)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].spines['top'].set_visible(False)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].spines['right'].set_visible(False)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].spines['bottom'].set_visible(False)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].spines['left'].set_visible(False)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].set_xlabel(f'z_{i}')
                plt.tight_layout()
                if show:
                    plt.show()
                else:
                    plt.savefig(f"{save_location_dir}/{file_name_key}_histograms_{id}.pdf", dpi=dpi, bbox_inches="tight")
                plt.close()

        #plot latent_train_codes histograms
        plot_hist(self.latent_train_codes, 'latent_train_codes', self.latent_space_diag_dir)

        #plot latent_val_codes histograms
        plot_hist(self.latent_val_codes, 'latent_val_codes', self.latent_space_diag_dir)

        #plot latent prior samples histograms
        plot_hist(self.latent_vector, 'latent_prior_samples', self.latent_space_diag_dir)

    def plot_latent_umap_tsne_scatters(self, n_neighbors = 100,  min_dist = 0.3, dpi= 600, show = False,
                                       fit_prior_only = False, denseMAP = True, **kwargs):
        """
        Plot UMAP scatters of training and validation latent codes.
        :param n_neighbors: number of neighbors to use to compute the UMAP embedding
        :param min_dist: minimum distance between points in the UMAP embedding
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param fit_prior_only: whether to fit UMAP based only on samples from the prior
        :param kwargs: additional arguments to be passed to plot_scatters()
        """
        print("Plotting UMAP & TSNE scatters of training and validation latent codes...")

        # give warning when fit_prior_only is True and denseMAP is True for umap
        if fit_prior_only and denseMAP:
            if type == 'umap':
                print("Warning: denseMAP cannot be used when fit_prior_only is True. Setting denseMAP to False.")
                denseMAP = False

        import umap
        from sklearn.manifold import TSNE

        def make_dim_redux(type):
            colors = ['grey', 'cornflowerblue', 'mediumvioletred']
            labels = ['Prior', 'Training', 'Validation']

            if fit_prior_only:
                all_data = self.latent_vector.numpy()
            else:
                all_data = np.vstack(
                    [self.latent_vector.numpy(), self.latent_train_codes.numpy(), self.latent_val_codes.numpy()])

            if type.lower() == 'umap':
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=self.experiment.seed,
                                  n_components=2, densmap=denseMAP)
            elif type.lower() == 'tsne':
                if not fit_prior_only:
                    reducer = TSNE(perplexity=n_neighbors, n_components=2, random_state=self.experiment.seed)
                else:
                    raise ValueError("TSNE cannot be fitted on prior samples only. Use UMAP instead.")
            else:
                raise ValueError("Unknown dimensionality reduction method.")

            embeddings = reducer.fit_transform(all_data)

            # read embedding for each data type separately. For each, concatenate the 1st and 2nd components of the embedding
            if not fit_prior_only:
                latent_id_start = 0
                latent_id_end = self.latent_vector.shape[0]
                latent_train_id_start = latent_id_end
                latent_train_id_end = latent_train_id_start + self.latent_train_codes.shape[0]
                latent_val_id_start = latent_train_id_end
                latent_val_id_end = latent_val_id_start + self.latent_val_codes.shape[0]

                embeddings_latent = np.expand_dims(np.hstack((np.expand_dims(embeddings[latent_id_start:latent_id_end, 0], axis = 1),
                                                                np.expand_dims(embeddings[latent_id_start:latent_id_end, 1], axis = 1))), axis = 0)
                embeddings_latent_train = np.expand_dims(np.hstack((np.expand_dims(embeddings[latent_train_id_start:latent_train_id_end, 0], axis = 1),
                                    np.expand_dims(embeddings[latent_train_id_start:latent_train_id_end, 1], axis = 1))), axis = 0)
                embeddings_latent_val = np.expand_dims(np.hstack((np.expand_dims(embeddings[latent_val_id_start:latent_val_id_end, 0], axis = 1),
                                    np.expand_dims(embeddings[latent_val_id_start:latent_val_id_end, 1], axis = 1))), axis = 0)
            else:
                embeddings_latent = np.expand_dims(embeddings, axis = 0)
                embeddings_latent_train = np.expand_dims(reducer.transform(self.latent_train_codes.numpy()), axis = 0)
                embeddings_latent_val = np.expand_dims(reducer.transform(self.latent_val_codes.numpy()), axis = 0)

            embedding_data = np.concatenate(
            [embeddings_latent, embeddings_latent_train, embeddings_latent_val], axis=0)

            optional_string = ''
            if type.lower() == 'tsne':
                optional_string = f"_KL = {reducer.kl_divergence_:.3f}"

            extra_string_comment = f"_fit_prior_only{fit_prior_only}_denseMAP{denseMAP}" if type == 'umap' else f"_fit_prior_only{fit_prior_only}"
            plot.plot_scatters(
                data=embedding_data,
                labels=labels,
                plot_title=f'{type} embedding scatters of latent prior samples, training and validation latent codes{optional_string}',
                colors=colors,
                save_location=f"{self.latent_space_diag_dir}/latent_{type}{extra_string_comment}_scatter.pdf",
                dpi=dpi,
                show=show,
                **kwargs
            )

        def make_dim_redux_2(type):
            import matplotlib.pyplot as plt

            # Define colors for each digit class (0-9)
            digit_colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 distinct colors

            # Define markers for data types
            markers = {'Prior': 'o', 'Training': '*', 'Validation': '^'}
            marker_sizes = {'Prior': 20, 'Training': 30, 'Validation': 20}

            if fit_prior_only:
                all_data = self.latent_vector.numpy()
            else:
                all_data = np.vstack(
                    [self.latent_vector.numpy(), self.latent_train_codes.numpy(), self.latent_val_codes.numpy()])

            if type.lower() == 'umap':
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=self.experiment.seed,
                                    n_components=2, densmap=denseMAP)
            elif type.lower() == 'tsne':
                if not fit_prior_only:
                    reducer = TSNE(perplexity=n_neighbors, n_components=2, random_state=self.experiment.seed)
                else:
                    raise ValueError("TSNE cannot be fitted on prior samples only. Use UMAP instead.")
            else:
                raise ValueError("Unknown dimensionality reduction method.")

            embeddings = reducer.fit_transform(all_data)

            # Get labels for each data type
            prior_labels = self.generated_y_label.numpy()
            train_labels = self.train_y_label.numpy()
            val_labels = self.val_y_label.numpy()

            # Create figure
            mpl, plt, make_axes_locatable, tick = plot.plots_imports()
            plot.base_config(mpl, **kwargs)
            fig, ax = plt.subplots(figsize=(12, 10))

            if not fit_prior_only:
                latent_id_start = 0
                latent_id_end = self.latent_vector.shape[0]
                latent_train_id_start = latent_id_end
                latent_train_id_end = latent_train_id_start + self.latent_train_codes.shape[0]
                latent_val_id_start = latent_train_id_end
                latent_val_id_end = latent_val_id_start + self.latent_val_codes.shape[0]

                embeddings_latent = embeddings[latent_id_start:latent_id_end]
                embeddings_latent_train = embeddings[latent_train_id_start:latent_train_id_end]
                embeddings_latent_val = embeddings[latent_val_id_start:latent_val_id_end]
            else:
                embeddings_latent = embeddings
                embeddings_latent_train = reducer.transform(self.latent_train_codes.numpy())
                embeddings_latent_val = reducer.transform(self.latent_val_codes.numpy())

            # Plot Prior samples (all grey)
            ax.scatter(embeddings_latent[:, 0], embeddings_latent[:, 1],
                       c='grey', marker=markers['Prior'], s=marker_sizes['Prior'],
                       alpha=0.3, label='Prior', edgecolors='none')

            # Plot Training samples (colored by digit)
            for digit in range(10):
                mask = train_labels == digit
                if np.any(mask):
                    ax.scatter(embeddings_latent_train[mask, 0], embeddings_latent_train[mask, 1],
                               c=[digit_colors[digit]], marker=markers['Training'],
                               s=marker_sizes['Training'], alpha=0.7,
                               label=f'Train-{digit}', edgecolors='white', linewidths=0.5)

            # Plot Validation samples (colored by digit)
            for digit in range(10):
                mask = val_labels == digit
                if np.any(mask):
                    ax.scatter(embeddings_latent_val[mask, 0], embeddings_latent_val[mask, 1],
                               c=[digit_colors[digit]], marker=markers['Validation'],
                               s=marker_sizes['Validation'], alpha=0.7,
                               label=f'Val-{digit}', edgecolors='white', linewidths=0.5)

            optional_string = ''
            if type.lower() == 'tsne':
                optional_string = f" (KL = {reducer.kl_divergence_:.3f})"

            ax.set_title(f'{type.upper()} embedding of latent codes{optional_string}', fontsize=14)
            ax.set_xlabel(f'{type.upper()} 1', fontsize=12)
            ax.set_ylabel(f'{type.upper()} 2', fontsize=12)

            # Create custom legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=9)

            plt.tight_layout()

            extra_string_comment = f"_fit_prior_only{fit_prior_only}_denseMAP{denseMAP}" if type == 'umap' else f"_fit_prior_only{fit_prior_only}"

            if show:
                plt.show()
            else:
                plt.savefig(f"{self.latent_space_diag_dir}/latent_{type}{extra_string_comment}_scatter.pdf",
                            dpi=dpi, bbox_inches="tight")
            plt.close()

        make_dim_redux_2(type='tsne')
        make_dim_redux_2(type='umap')

    def plot_x_y_umap_tsne_scatters(self, n_neighbors = 100, min_dist = 0.3, dpi = 600, show = False, fit_prior_only = False,
                                    denseMAP = True, **kwargs):
        """
        Plot UMAP & TSNE scatters of training, validation and generated data
        :param n_neighbors: number of neighbors to use to compute the UMAP embedding
        :param min_dist: minimum distance between points in the UMAP embedding
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param fit_prior_only: whether to fit UMAP based only on samples from the prior
        :param kwargs: additional arguments to be passed to plot_umap_scatter()
        """
        print("Plotting UMAP & TSNE scatters of training, validation and generated X & Y...")

        import umap
        from sklearn.manifold import TSNE

        def make_dim_redux(data, title_string, type):
            """
            data is a list of tensors [train, recon_train, recon_val, generated]
            """
            colors = ['grey', 'cornflowerblue', 'lightseagreen', 'lightcoral']
            labels = ['Train', 'Recon.-Train', 'Recon.-Val', 'Generated']
            markers = ['o', '.', 'x', '*']

            if fit_prior_only:
                # only train
                all_data = data[0].numpy()
            else:
                # concatenate train, reconstructions and generated
                all_data = np.vstack([data[0].numpy(), data[1].numpy(), data[2].numpy(), data[3].numpy()])

            if type.lower() == 'umap':
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=self.experiment.seed,
                                  n_components=2, densmap = denseMAP)

            elif type.lower() == 'tsne':
                if not fit_prior_only:
                    reducer = TSNE(perplexity=n_neighbors, n_components=2, random_state=self.experiment.seed)
                else:
                    raise ValueError("TSNE cannot be fitted on prior samples only. Use UMAP instead.")
            else:
                raise ValueError("Unknown dimensionality reduction method.")

            embeddings = reducer.fit_transform(all_data)

            # read embedding for each data type separately. For each, concatenate the 1st and 2nd components of the embedding
            if not fit_prior_only:
                train_id_start = 0
                train_id_end = data[0].shape[0]
                recon_train_id_start = train_id_end
                recon_train_id_end = recon_train_id_start + data[1].shape[0]
                recon_val_id_start = recon_train_id_end
                recon_val_id_end = recon_val_id_start + data[2].shape[0]
                gen_id_start = recon_val_id_end
                gen_id_end = gen_id_start + data[3].shape[0]

                embeddings_train = np.expand_dims(np.hstack((np.expand_dims(embeddings[train_id_start:train_id_end, 0], axis = 1),
                                 np.expand_dims(embeddings[train_id_start:train_id_end, 1], axis = 1))), axis = 0)
                embeddings_recon_train = np.expand_dims(np.hstack((np.expand_dims(embeddings[recon_train_id_start:recon_train_id_end, 0], axis = 1),
                                            np.expand_dims(embeddings[recon_train_id_start:recon_train_id_end, 1], axis = 1))), axis = 0)
                embeddings_recon_val = np.expand_dims(np.hstack((np.expand_dims(embeddings[recon_val_id_start:recon_val_id_end, 0], axis = 1),
                                            np.expand_dims(embeddings[recon_val_id_start:recon_val_id_end, 1], axis = 1))), axis = 0)
                embeddings_gen = np.expand_dims(np.hstack((np.expand_dims(embeddings[gen_id_start:gen_id_end, 0], axis=1),
                                                           np.expand_dims(embeddings[gen_id_start:gen_id_end, 1], axis=1))), axis=0)
            else:
                embeddings_train = np.expand_dims(embeddings, axis = 0)
                embeddings_recon_train = np.expand_dims(reducer.transform(data[1].numpy()), axis = 0)
                embeddings_recon_val = np.expand_dims(reducer.transform(data[2].numpy()), axis = 0)
                embeddings_gen = np.expand_dims(reducer.transform(data[3].numpy()), axis = 0)

            embeddings_data = np.concatenate(
                [embeddings_train, embeddings_recon_train, embeddings_recon_val, embeddings_gen], axis=0)

            optional_string = ''
            if type.lower() == 'tsne':
                optional_string = f"_KL = {reducer.kl_divergence_:.3f}"

            plot.plot_scatters(
                data=embeddings_data,
                labels=labels,
                plot_title=f'{type} embedding scatters of training, validation and generated {title_string}{optional_string}',
                colors=colors,
                markers = markers,
                save_location=f"{self.logging_dir}/{type}_scatter_{title_string}.pdf",
                dpi=dpi,
                show=show,
                **kwargs
            )

        if self.train_x is None or self.val_x is None or self.train_y is None or self.val_y is None:
            raise ValueError("No training or validation data to plot. Build diagnostics data first.")

        if (self.generated_x is None or self.reconstructed_train_x is None or self.reconstructed_val_x is None
                or self.reconstructed_train_y is None or self.reconstructed_val_y is None):
            raise ValueError("No generated or reconstructed data to plot. "
                             "Build diagnostics data first.")

        dim_x = self.experiment.dim_x
        train_x = self.train_x.view(-1, dim_x)
        generated_x = self.generated_x.view(-1, dim_x)
        recon_train_x = self.reconstructed_train_x.view(-1, dim_x)
        recon_val_x = self.reconstructed_val_x.view(-1, dim_x)

        make_dim_redux([train_x, recon_train_x, recon_val_x, generated_x], 'X', type='umap')
        make_dim_redux([train_x, recon_train_x, recon_val_x, generated_x], 'X', type='tsne')


    def plot_x_y_umap_tsne_scatters_2(self, n_neighbors=100, min_dist=0.3, dpi=600, show=False, fit_prior_only=False,
                                denseMAP=True, **kwargs):
        """
        Plot UMAP & TSNE scatters of training, validation and generated data
        :param n_neighbors: number of neighbors to use to compute the UMAP embedding
        :param min_dist: minimum distance between points in the UMAP embedding
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param fit_prior_only: whether to fit UMAP based only on samples from the prior
        :param denseMAP: whether to use densMAP for UMAP
        :param kwargs: additional arguments to be passed to plot_umap_scatter()
        """
        print("Plotting UMAP & TSNE scatters of training, validation and generated X & Y...")

        import umap
        from sklearn.manifold import TSNE


        def make_dim_redux(data, labels_list, title_string, data_types, type):
            """
            data is a list of tensors [train, recon_train, recon_val, generated]
            labels_list is a list of label tensors [train_labels, recon_train_labels, recon_val_labels, gen_labels]
            """

            # Define markers for data types
            markers_all = {'Train': 'o', 'Recon.-Train': 's', 'Recon.-Val': '^', 'Generated': '*'}
            marker_sizes_all = {'Train': 20, 'Recon.-Train': 20, 'Recon.-Val': 25, 'Generated': 30}
            data_type_names_all = ['Train', 'Recon.-Train', 'Recon.-Val', 'Generated']

            # select markers and sizes based on data_types input
            markers = {dt: markers_all[dt] for dt in data_types}
            marker_sizes = {dt: marker_sizes_all[dt] for dt in data_types}
            data_type_names = [dt for dt in data_type_names_all if dt in data_types]

            if fit_prior_only:
                all_data = data[0].numpy()
            else:
                all_data = np.vstack([d.numpy() for d in data])

            if type.lower() == 'umap':
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=self.experiment.seed,
                                  n_components=2, densmap=denseMAP)
            elif type.lower() == 'tsne':
                if not fit_prior_only:
                    reducer = TSNE(perplexity=n_neighbors, n_components=2, random_state=self.experiment.seed)
                else:
                    raise ValueError("TSNE cannot be fitted on prior samples only. Use UMAP instead.")
            else:
                raise ValueError("Unknown dimensionality reduction method.")

            embeddings = reducer.fit_transform(all_data)

            # Create figure
            mpl, plt, make_axes_locatable, tick = plot.plots_imports()
            plot.base_config(mpl, **kwargs)
            fig, ax = plt.subplots(figsize=(12, 10))

            # Define colors for each digit class (0-9)
            digit_colors = plt.cm.tab10(np.linspace(0, 1, 10))

            # Split embeddings back into separate datasets
            if not fit_prior_only:
                idx = 0
                embeddings_list = []
                for d in data:
                    size = d.shape[0]
                    embeddings_list.append(embeddings[idx:idx+size])
                    idx += size
            else:
                embeddings_list = [embeddings]
                for i in range(1, len(data)):
                    if denseMAP or type.lower() == 'tsne':
                        embeddings_list.append(reducer.fit_transform(data[i].numpy()))
                    else:
                        embeddings_list.append(reducer.transform(data[i].numpy()))

            # Plot each dataset type with digit-based coloring
            for data_idx, (emb, labels, data_type) in enumerate(zip(embeddings_list, labels_list, data_type_names)):
                for digit in range(10):
                    mask = labels.numpy() == digit
                    if np.any(mask):
                        ax.scatter(emb[mask, 0], emb[mask, 1],
                                 c=[digit_colors[digit]],
                                 marker=markers[data_type],
                                 s=marker_sizes[data_type],
                                 alpha=0.6,
                                 label=f'{data_type}-{digit}', #if data_idx == 0 else None,
                                 edgecolors= 'none', #'black' if data_type != 'Train' else 'none',
                                 linewidths=0.5)

            optional_string = ''
            if type.lower() == 'tsne':
                optional_string = f" (KL = {reducer.kl_divergence_:.3f})"

            ax.set_title(f'{type.upper()} embedding of {title_string}{optional_string}', fontsize=14)
            ax.set_xlabel(f'{type.upper()} 1', fontsize=12)
            ax.set_ylabel(f'{type.upper()} 2', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)

            plt.tight_layout()

            extra_string_comment = f"_fit_prior_only{fit_prior_only}_denseMAP{denseMAP}" if type == 'umap' else f"_fit_prior_only{fit_prior_only}"

            if show:
                plt.show()
            else:
                plt.savefig(f"{self.logging_dir}/{type}_scatter_{title_string}{extra_string_comment}.pdf",
                           dpi=dpi, bbox_inches='tight')
            plt.close()

        if self.train_x is None or self.val_x is None or self.train_y is None or self.val_y is None:
            raise ValueError("No training or validation data to plot. Build diagnostics data first.")

        if (self.generated_x is None or self.reconstructed_train_x is None or self.reconstructed_val_x is None
                or self.reconstructed_train_y is None or self.reconstructed_val_y is None):
            raise ValueError("No generated or reconstructed data to plot. Build diagnostics data first.")

        dim_x = self.experiment.dim_x
        train_x = self.train_x.view(-1, dim_x)
        generated_x = self.generated_x.view(-1, dim_x)
        recon_train_x = self.reconstructed_train_x.view(-1, dim_x)
        recon_val_x = self.reconstructed_val_x.view(-1, dim_x)

        # Prepare labels
        train_labels = self.train_y_label
        recon_train_labels = self.reconstructed_train_y_label
        recon_val_labels = self.reconstructed_val_y_label
        gen_labels = self.generated_y_label

        make_dim_redux([train_x, generated_x],
                       [train_labels, gen_labels], data_types=['Train', 'Generated'],
                       title_string='train_vs_generated', type='umap')
        make_dim_redux([train_x, generated_x],
                       [train_labels, gen_labels], data_types=['Train', 'Generated'],
                       title_string='train_vs_generated', type='tsne')

        make_dim_redux([train_x, recon_val_x],
                       [train_labels, recon_val_labels], data_types=['Train', 'Recon.-Train'],
                       title_string='train_vs_recons_train', type='umap')
        make_dim_redux([train_x, recon_val_x],
                       [train_labels, recon_val_labels], data_types=['Train', 'Recon.-Train'],
                       title_string='train_vs_recons_train', type='tsne')

        make_dim_redux([train_x, recon_val_x],
                       [train_labels, recon_val_labels], data_types=['Train', 'Recon.-Val'],
                       title_string='train_vs_recons_val', type='umap')
        make_dim_redux([train_x, recon_val_x],
                       [train_labels, recon_val_labels],data_types=['Train', 'Recon.-Val'],
                       title_string='train_vs_recons_val', type='tsne')

    def assess_class_proportions(self, **kwargs):
        """
        Assess class proportions in original training data and class proportions in generated data.
            Compare them using Chi-squared test and print to console.
        :return:
        """
        print("Assessing class proportions in training and generated data...")

        from fastabc_inversion.conditional_generation.utils.plotting import plot_samples_inspections

        from scipy.stats import chisquare

        # run through the whole training set self.experiment.train_loader to get class counts
        train_class_counts = np.zeros(self.experiment.dim_y)
        for _, onehot in self.experiment.train_loader:
            labels = self.experiment.label_transform.simplex_vec_to_label(onehot)

            for c in range(self.experiment.dim_y):
                train_class_counts[c] += torch.sum(labels == c).item()
        train_class_proportions = train_class_counts / np.sum(train_class_counts)

        # get class counts in generated data
        sample_size = self.experiment.train_size
        z_dist = self.experiment.latent_dist
        z_dist_params = self.experiment.latent_dist_params_list
        latent_dim = self.experiment.latent_dim
        device = self.experiment.device

        gen_class_counts = np.zeros(self.experiment.dim_y)
        classified_gen_class_counts = np.zeros(self.experiment.dim_y)

        # generate a sample of size sample_size
        with torch.no_grad():
            latent_vector = z_dist(
                torch.tensor(z_dist_params[0], dtype=torch.float32),
                torch.tensor(z_dist_params[1], dtype=torch.float32),
            ).sample((sample_size, latent_dim)).to(device)

            generated_x, generated_y = self.experiment.netG(latent_vector)
            del latent_vector

            gen_labels = self.experiment.label_transform.simplex_vec_to_label(generated_y)

            classified_generated_x = self.classifier(generated_x)
            classified_labels = self.experiment.label_transform.simplex_vec_to_label(classified_generated_x)

            plot_samples_inspections(generated_x.cpu(), gen_labels, labels_2=classified_labels, random_samples=False,
                                     grid_size=10,
                                     save_fig_path=f"{self.logging_dir}/generated_samples_by_true_labels.pdf", dpi=600)

            del generated_x
            del generated_y

            for c in range(self.experiment.dim_y):
                gen_class_counts[c] += torch.sum(gen_labels == c).item()
                classified_gen_class_counts[c] += torch.sum(classified_labels == c).item()

        gen_class_proportions = gen_class_counts / np.sum(gen_class_counts)
        classified_gen_class_proportions = classified_gen_class_counts / np.sum(classified_gen_class_counts)

        # perform Chi-squared test
        chi2_stat, p_value = chisquare(gen_class_counts, f_exp=train_class_counts)
        chi2_stat_classified, p_value_classified = chisquare(classified_gen_class_counts, f_exp=train_class_counts)

        print("Class Proportions Assessment:")
        print(f"Training data class proportions: {train_class_proportions}")
        print(f"Generated data class proportions: {gen_class_proportions}")
        print(f"Classified Generated data class proportions: {classified_gen_class_proportions}")
        print(f"Chi-squared statistic: {chi2_stat:.4f}, p-value: {p_value:.4f}")
        print(f"Classified Chi-squared statistic: {chi2_stat_classified:.4f}, p-value: {p_value_classified:.4f}")

        # plot bar chart of class proportions
        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config(mpl, **kwargs)

        x = np.arange(self.experiment.dim_y)
        width = 0.3  # narrower bars
        gap = 0 #0.08  # extra gap between bars
        offsets = np.array([-1, 0, 1]) * (width + gap)

        fig, ax = plt.subplots()
        # plot three bars per class: train, generated, classified generated
        bars1 = ax.bar(x + offsets[0], train_class_proportions, width, label='Training Data', color='cornflowerblue')
        bars2 = ax.bar(x + offsets[1], gen_class_proportions, width, label='Generated Data', color='lightcoral')
        bars3 = ax.bar(x + offsets[2], classified_gen_class_proportions, width, label='Classified Generated Data', color='mediumseagreen')

        ax.set_xlabel('Class')
        ax.set_ylabel('Proportion')
        ax.set_title('Class Proportions in Training and Generated Data')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(self.experiment.dim_y)])
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.logging_dir}/class_proportions_comparison.pdf", dpi=600)
        plt.close()

    def assess_frechet_scores(self, pca_components =.85, sample_size = 300, repeats= 100):
        """
        Assess Frechet distance between training data and generated data in PCA-reduced space.
        :param pca_components: float, number of PCA components to keep (if <1, fraction of variance; if >=1, number of components)
        """

        print("Assessing Frechet distances between training and generated data...")

        from fastabc_inversion.conditional_generation.utils.frechet_pca_distance import compute_fpd_pca

        precision = self.round_digits

        train_x = self.train_x.view(-1, self.experiment.dim_x).numpy()
        train_size = train_x.shape[0]

        generated_x = self.generated_x.view(-1, self.experiment.dim_x).numpy()
        gen_size = generated_x.shape[0]

        # make a PCA object and fit it on the whole training data from self.experiment.train_loader
        from sklearn.decomposition import PCA
        all_train_dataset = self.experiment.train_loader.dataset
        all_train_dataset = DataLoader(all_train_dataset, batch_size=self.experiment.train_size, shuffle=False)
        all_train_x, _ = next(iter(all_train_dataset))
        del all_train_dataset
        all_train_x = all_train_x.view(-1, self.experiment.dim_x).numpy()
        pca = PCA(n_components=pca_components)
        pca.fit(all_train_x)
        print(f'PCA fitted on the full training data with {pca.n_components_} components.')
        del all_train_x

        fpd_scores = []
        fpd_scores_refs = []

        for r in range(repeats):
            # sample random subset of training data
            train_indices = np.random.choice(train_size, size=sample_size, replace=False)
            train_indices_2 = np.random.choice(train_size, size=sample_size, replace=False)
            train_subset = train_x[train_indices]
            train_subset_2 = train_x[train_indices_2]

            # sample random subset of generated data
            gen_indices = np.random.choice(gen_size, size=sample_size, replace=False)
            gen_subset = generated_x[gen_indices]

            # compute FPD
            fpd = compute_fpd_pca(train_subset, gen_subset, fitted_pca=pca)
            fpd_ref = compute_fpd_pca(train_subset, train_subset_2, fitted_pca=pca)

            fpd_scores.append(fpd)
            fpd_scores_refs.append(fpd_ref)
            print(f"Repeat {r+1}/{repeats}: FPD = {fpd:.4f}")

        fpd_scores = np.array(fpd_scores)
        fpd_scores_refs = np.array(fpd_scores_refs)

        fpd_scores_stats = self.compute_array_stats(fpd_scores)
        fpd_scores_refs_stats = self.compute_array_stats(fpd_scores_refs)

        write_data_frechet_scores_gen_x = self.make_array_stats_string(fpd_scores_stats, precision)
        write_data_frechet_scores_gen_x_txt = "Frechet distance in PCA space between training and generated images: "
        write_header_rechet_scores_gen_x = "mean_frechet_gen_x, std_frechet_gen_x, median_frechet_gen_x,(q25_frechet_gen_x,q75_frechet_gen_x),(q025_frechet_gen_x,q975_frechet_gen_x)"

        write_data_frechet_scores_gen_x_refs = self.make_array_stats_string(fpd_scores_refs_stats, precision)
        write_data_frechet_scores_gen_x_txt_refs = "(Ref) Frechet distance in PCA space between two random subsets of training images: "
        write_header_rechet_scores_gen_x_refs = "mean_frechet_train_x_refs, std_frechet_train_x_refs, median_frechet_train_x_refs,(q25_frechet_train_x_refs,q75_frechet_train_x_refs),(q025_frechet_train_x_refs,q975_frechet_train_x_refs)"

        return {
            "data": {
                "write_data_frechet_scores_gen_x": (write_data_frechet_scores_gen_x, write_data_frechet_scores_gen_x_txt),
                "write_data_frechet_scores_gen_x_refs": (write_data_frechet_scores_gen_x_refs, write_data_frechet_scores_gen_x_txt_refs),
                },
            "headers": {
                "write_header_frechet_scores_gen_x": write_header_rechet_scores_gen_x,
                "write_header_frechet_scores_gen_x_refs": write_header_rechet_scores_gen_x_refs,
                }

        }




















