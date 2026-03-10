"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Base class for diagnostics on the trained models and to compare different models performances.
Not used for inference results.
"""
# TODO: expose the plots functions parameters to the user via the make_XYZ functions parameters

import os

import fastabc_inversion.geo_problems.utils.visualization.plotting_tools as plot
import numpy as np
import pingouin as pg
import torch
import umap
from fastabc_inversion.geo_problems.linear.analytical_inversion import (
    compute_rmse_grdTruth, resimulate)
from fastabc_inversion.geo_problems.utils.evaluation.mmd import MMD2 as mmd
from fastabc_inversion.geo_problems.utils.evaluation.mmd import \
    two_sample_mmd_test
from fastabc_inversion.geo_problems.utils.torch_data_prep import un_normalize
from fastabc_inversion.geo_problems.utils.torch_distances import rmse_torch
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


class Diagnostics:
    def __init__(self, experiment_obj, epoch, round_digits=3):
        """
        Neural Networks training diagnostics class
        :param experiment_obj: the experiment object containing all the experiment parameters and data.
        :param epoch: the epoch number for which the diagnostics are run.
        :param round_digits: number of digits to round the results to.
        """
        self.mmd_kernel_params = {"metric": "rbf"}
        self.mmd_est_params = {
            "x": None,
            "y": None,
            "z": None,
        }  # TODO: make these parameters configurable from outside

        self.AVAILABLE_DIAGS = {
            "min_max": self.make_min_max_stats,
            "pixel_stats": self.make_pixel_stats,
            "recons_stats": self.make_recons_stats,
            "resim_stats": self.make_resim_stats,
            "variance_loss": self.make_variance_loss_stats,
            "latent_dist": self.inspect_latent_distribution,
            "variograms": self.plot_variograms,
            "structural_similarity": self.make_ssim_stats,
            "cosine_similariry": self.make_cosine_sim_stats,
            "prior_vs_gen_dist_distance": self.make_priorvsgen_dists_distances_stats,
        }
        self.two_sample_test_params = {
            "kernel_params": self.mmd_kernel_params.copy(),
            "unbiased": True,
            "alpha": 0.05,
            "iterations": 1000,
        }  # TODO : make these parameters configurable from outside
        self.mmd_params = {
            "kernel_params": self.mmd_kernel_params.copy(),
            "unbiased": False,
        }  # TODO : make these parameters configurable from outside

        self.DEFAULT_DIAG_PARAMS = {
            "min_max": {"build_y_stats": True, "make_plots": True},
            "pixel_stats": {
                "build_y_stats": True,
                "joint_output": True,
                "make_plots": True,
                "n_neighbors": 15,
                "min_dist": 0.1,
                "denseMAP": True,
            },
            "recons_stats": {
                "build_y_stats": True,
                "joint_output": True,
                "make_plots": True,
            },
            "resim_stats": {"make_plots": True},
            "variance_loss": {
                "build_y_stats": True,
                "joint_output": True,
                "make_plots": True,
            },
            "latent_dist": {
                "make_plots": True,
                "n_neighbors": 5,
                "min_dist": 0.1,
                "mmd_params": self.mmd_params,
                "two_sample_test_params": self.two_sample_test_params,
                "test_sample_size": 300,
                "denseMAP": True,
            },
            "variograms": {
                "detailed_varios": [0, 0, 0, 0, 0],
                "add_legend": True,
                "datasets_selection": ["train", "recon_val", "gen"],
            },
            "structural_similarity": {
                "ssim_kwargs": {"gaussian_weights": True, "full": False}
            },
            "cosine_similariry": {"build_y_stats": True},
            "prior_vs_gen_dist_distance": {
                "mmd_params": self.mmd_params,
                "two_sample_test_params": self.two_sample_test_params,
                "sample_size": 300,
            },
        }
        self.NO_RESULTS_DIAGS = ["variograms"]
        self.experiment = experiment_obj
        self.round_digits = round_digits

        # setup logging file in parent directory
        self.logging_dir = None
        self.all_exp_logging_file = None
        self.logging_string = None
        self.logging_headers = None
        self.exp_diag_stats_file = None

        self.latent_vector = None
        self.latent_train_codes = None
        self.latent_val_codes = None

        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None

        self.generated_data = None
        self.generated_x = None
        self.generated_y = None
        self.resimulated_generated_x = None

        self.reconstructed_train_data = None
        self.reconstructed_train_x = None
        self.reconstructed_train_y = None
        self.resimulated_recon_train_x = None

        self.reconstructed_val_data = None
        self.reconstructed_val_x = None
        self.reconstructed_val_y = None
        self.resimulated_recon_val_x = None

        self.hyperparams_string = None
        self.hpyperparams_headers = None

        self.results_vecs = {}
        self.epoch = epoch

    def build_diagnostics_data(self):
        """
        Make reconstruction and generated data for diagnostics
        """
        model = self.experiment.model
        model.eval()

        netG = model.netG
        netD = model.netD

        device = self.experiment.device

        dim_x = self.experiment.dim_x
        dim_y = self.experiment.dim_y

        z_dist = self.experiment.latent_dist
        z_dist_params = self.experiment.latent_dist_params_list
        latent_dim = self.experiment.latent_dim

        sample_size = self.experiment.val_size

        if sample_size < self.experiment.train_size:
            # select random training data to read for diagnostics comparisons
            idx_rnd = np.random.randint(0, self.experiment.train_size, size=sample_size)

            self.train_x = (
                self.experiment.training_x[idx_rnd, :, :, :]
                if len(self.experiment.training_x.shape) > 2
                else self.experiment.training_x[idx_rnd, :]
            )
            self.train_y = self.experiment.training_y[idx_rnd, :]
        else:
            self.train_x = self.experiment.training_x
            self.train_y = self.experiment.training_y

        self.val_x = self.experiment.validation_x
        self.val_y = self.experiment.validation_y

        train_data = torch.cat(
            (
                self.train_x.view(sample_size, dim_x),
                self.train_y.view(sample_size, dim_y),
            ),
            1,
        ).to(device)
        val_data = torch.cat(
            (self.val_x.view(sample_size, dim_x), self.val_y.view(sample_size, dim_y)),
            1,
        ).to(device)

        latent_vector = (
            z_dist(
                torch.tensor(z_dist_params[0], dtype=torch.float32),
                torch.tensor(z_dist_params[1], dtype=torch.float32),
            )
            .sample((sample_size, latent_dim))
            .to(device)
        )

        with torch.no_grad():
            self.generated_data = netG(latent_vector).cpu()
            self.generated_x = self.generated_data[:, 0:dim_x]
            self.generated_y = self.generated_data[:, dim_x:]

            train_latent_codes, _, __ = netD(train_data)
            self.reconstructed_train_data = netG(train_latent_codes).cpu()
            self.reconstructed_train_x = self.reconstructed_train_data[:, 0:dim_x]
            self.reconstructed_train_y = self.reconstructed_train_data[:, dim_x:]
            self.latent_train_codes = train_latent_codes.cpu()

            val_latent_codes, _, __ = netD(val_data)
            self.reconstructed_val_data = netG(val_latent_codes).cpu()
            self.reconstructed_val_x = self.reconstructed_val_data[:, 0:dim_x]
            self.reconstructed_val_y = self.reconstructed_val_data[:, dim_x:]
            self.latent_val_codes = val_latent_codes.cpu()

            self.latent_vector = latent_vector.cpu()

        if self.experiment.normalize:
            self.train_x = un_normalize(
                self.train_x.view(sample_size, dim_x),
                self.experiment.normalization_dict_x,
            )
            self.train_y = un_normalize(
                self.train_y, self.experiment.normalization_dict_y
            )
            self.val_x = un_normalize(self.val_x, self.experiment.normalization_dict_x)
            self.val_y = un_normalize(self.val_y, self.experiment.normalization_dict_y)
            self.generated_x = un_normalize(
                self.generated_x, self.experiment.normalization_dict_x
            )
            self.generated_y = un_normalize(
                self.generated_y, self.experiment.normalization_dict_y
            )
            self.reconstructed_train_x = un_normalize(
                self.reconstructed_train_x, self.experiment.normalization_dict_x
            )
            self.reconstructed_train_y = un_normalize(
                self.reconstructed_train_y, self.experiment.normalization_dict_y
            )
            self.reconstructed_val_x = un_normalize(
                self.reconstructed_val_x, self.experiment.normalization_dict_x
            )
            self.reconstructed_val_y = un_normalize(
                self.reconstructed_val_y, self.experiment.normalization_dict_y
            )

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

        self.logging_dir = (
            self.experiment.experiment_dir + f"/diagnostics_epoch_{self.epoch}"
        )
        self.reconstructions_diag_dir = self.logging_dir + "/reconstructions"
        self.resimulations_diag_dir = self.logging_dir + "/resimulations"
        self.latent_space_diag_dir = self.logging_dir + "/latent_space"

        dirs_to_create = [
            self.logging_dir,
            self.reconstructions_diag_dir,
            self.resimulations_diag_dir,
            self.latent_space_diag_dir,
        ]

        # create directories if they don't exist
        for dir in dirs_to_create:
            os.makedirs(dir, exist_ok=True)

        self.all_exp_logging_file = self.experiment.data_rootdir + "/jGNNdiag.csv"
        self.logging_string = (
            f"{self.experiment.latent_dim}:{self.experiment.model_log_ref}:"
        )
        self.logging_headers = "latent_dim:model_log_ref:"
        self.exp_diag_stats_file = self.logging_dir + "/stats.txt"

        # write hyperparameters to string
        # read the hyperparameters from the experiment object
        seed = {"seed": self.experiment.seed}
        train_size = {"train_size": self.experiment.train_size}
        val_size = {"val_size": self.experiment.val_size}
        optimizer_name = {
            "optimizer_name": type(self.experiment.optimizer).__name__
            if self.experiment.optimizer is not None
            else "N/A"
        }
        encoder_smoothness = {
            "encoder_smoothness": self.experiment.nn_params["spectral_norm_encoder"]
        }
        encoder_activation = {
            "encoder_activation": "/".join(
                [
                    str(v)
                    for v in self.experiment.nn_params[
                        "activation_dict_encoder"
                    ].values()
                ]
            )
        }
        decoder_activation = {
            "decoder_activation": "/".join(
                [
                    str(v)
                    for v in self.experiment.nn_params[
                        "activation_dict_decoder"
                    ].values()
                ]
            )
        }
        latent_dimension = {"latent_dimension": self.experiment.latent_dim}
        latent_distribution = {"latent_distribution": self.experiment.latent_dist_name}
        latent_distribution_params = {
            "latent_distribution_params": self.experiment.latent_dist_params_list
        }
        norm_power_p = {"norm_power_p": self.experiment.norms_params["l_norm_p_x"]}
        batch_size = {"batch_size": self.experiment.model_training_params["batch_size"]}
        # nb_epochs = {"nb_epochs":self.experiment.model_training_params['nb_epochs']}
        sink_lambda = {
            "sink_lambda": self.experiment.sinkhorn_lambda_scheduling_params[
                "sink_lambda"
            ]
        }
        sink_eps = {"sink_eps": self.experiment.sinkhorn_params["epsilon"]}
        # sink_niter = {"sink_niter":self.experiment.sinkhorn_params['niter']}
        sink_p = {"sink_p": self.experiment.sinkhorn_params["p"]}
        lambda_sink_scheduling = {
            "lambda_sink_scheduling": True
            if self.experiment.sinkhorn_lambda_scheduling_params[
                "sink_lambda_scheduler_factor"
            ]
            > 1
            else False
        }
        lr_scheduling = {"lr_scheduling": self.experiment.lr_scheduling}
        lr_scheduler = {
            "lr_scheduler": self.experiment.optim_params.get("lr_scheduler", None)
        }
        lr = {"lr": self.experiment.optim_params["lr"]}
        beta1 = {"beta1": self.experiment.optim_params["betas"][0]}
        beta2 = {"beta2": self.experiment.optim_params["betas"][1]}
        # lr_factor = {"lr_factor":self.experiment.optim_params['lr_factor'] if lr_scheduling else 'N/A'}
        # lr_patience = {"lr_patience":self.experiment.optim_params['lr_patience'] if lr_scheduling else 'N/A'}
        # lr_threshold = {"lr_threshold":self.experiment.optim_params['lr_threshold'] if lr_scheduling else 'N/A'}
        recon_loss_scale_by_dim = {
            "recon_loss_scale_by_dim": True
            if self.experiment.norms_params["norm_fct_type_x"] == "mse"
            else False
        }
        sink_scale_by_dim = {
            "sink_scale_by_dim": self.experiment.sinkhorn_params["scale_by_dim"]
        }
        data_normalizer = {
            "data_normalizer": self.experiment.normalization_dict_x["function"]
            if self.experiment.normalize
            else None
        }
        logging_comment = {"logging_comment": logging_comment}

        hyperparams_to_log = [
            seed,
            train_size,
            val_size,
            optimizer_name,
            encoder_smoothness,
            encoder_activation,
            decoder_activation,
            latent_dimension,
            latent_distribution,
            latent_distribution_params,
            norm_power_p,
            batch_size,
            sink_lambda,
            sink_eps,
            sink_p,
            lambda_sink_scheduling,
            lr_scheduling,
            lr_scheduler,
            lr,
            beta1,
            beta2,
            recon_loss_scale_by_dim,
            sink_scale_by_dim,
            data_normalizer,
            logging_comment,
        ]

        self.hyperparams_string, self.hpyperparams_headers = make_logging_strings(
            hyperparams_to_log
        )

    def run_diagnostics(
        self, diagnostics_to_run, diags_params, logging_comment="", write_to_file=True
    ):
        """
        Run all diagnostics and log to files
        :param diagnostics_to_run: list of diagnostics to run, if empty run all. Available diagnostics are:
                                   "min_max", "pixel_stats", "recons_stats", "resim_stats", "variance_loss",
                                   "latent_dist", "variograms".
        :param build_y_stats: if True, build statistics for y data
        :param make_plots: if True, make plots of the statistics
        :param write_to_file: if True, write results to file
        """

        # build diagnostics data
        if diagnostics_to_run == ["prior_vs_gen_dist_distance"]:
            pass
        else:
            self.build_diagnostics_data()  # do not run when 'prior_vs_gen_dist_distance' only.

        # setup logging directory and files
        self.setup_logging(logging_comment=logging_comment)

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
            self.append_logging_string(
                self.hyperparams_string, self.hpyperparams_headers
            )
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
        text_to_file = (
            f"\n {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} \n"  # add date and time
        )

        for i in range(len(data_keys)):
            self.logging_string = f"{self.logging_string}:{data[data_keys[i]][0]}"
            self.logging_headers = f"{self.logging_headers}:{headers[headers_keys[i]]}"
            # append to text_to_file
            text_to_file = (
                f"{text_to_file}{data[data_keys[i]][1]}: {data[data_keys[i]][0]} \n"
            )

        # write to stats file
        with open(self.exp_diag_stats_file, "a+") as f:
            f.write(text_to_file)
            f.write("\n")

        # backup Diagnostics class instance

    def compute_array_stats(self, np_array=None):
        """
        Compute statistics for an array.
        Statistics computed are mean, median, 25th and 75th percentiles, 2.5th and 97.5th percentiles.
        :param np_array: 1D numpy array
        """
        if np_array is not None:
            mean = np.mean(np_array)
            median = np.median(np_array)
            std = np.std(np_array)
            q25 = np.quantile(np_array, q=0.25)
            q75 = np.quantile(np_array, q=0.75)
            q025 = np.quantile(np_array, q=0.025)
            q975 = np.quantile(np_array, q=0.975)

            return {
                "mean": mean,
                "std": std,
                "median": median,
                "q25": q25,
                "q75": q75,
                "q025": q025,
                "q975": q975,
            }
        else:
            return {
                "mean": None,
                "std": None,
                "median": None,
                "q25": None,
                "q75": None,
                "q025": None,
                "q975": None,
            }

    def make_array_stats_string(self, stats, precision):
        """
        Prepare string for array statistics.
        :param stats: dictionnary with keys "mean", "median", "q25", "q75", "q025", "q975"
        """
        mean = f"{stats['mean']:.{precision}f}" if stats["mean"] is not None else "N/A"
        median = (
            f"{stats['median']:.{precision}f}" if stats["median"] is not None else "N/A"
        )
        q25 = f"{stats['q25']:.{precision}f}" if stats["q25"] is not None else "N/A"
        q75 = f"{stats['q75']:.{precision}f}" if stats["q75"] is not None else "N/A"
        q025 = f"{stats['q025']:.{precision}f}" if stats["q025"] is not None else "N/A"
        q975 = f"{stats['q975']:.{precision}f}" if stats["q975"] is not None else "N/A"

        return f"{mean},{median},([{q25},{q75}],[{q025},{q975}])"

    def estimate_mmd_kernel_gamma(self, data):
        """
        Estimate MMD kernel parameter using median heuristic on the provided data.
        :param data: data to use for estimation.
        :return: estimated MMD kernel parameter gamma.
        """
        from fastabc_inversion.geo_problems.utils.evaluation.mmd import \
            estimate_median_pairwise_dists

        sq_median = estimate_median_pairwise_dists(
            data, sample_ratio=1.0, chunk_size=300
        )  # squared euclidean
        return 1 / sq_median

    def estimate_mmd_params(self, which_variable, max_sample_size=10000):
        """
        Estimate MMD kernel parameter using median heuristic.
        :param which_variable: string indicating which variable to use for estimation. Options are 'x', 'y', 'z'.
        :param max_sample_size: maximum number of samples to use for estimation.
        """
        print(
            f"Estimating overall median squared euclidean distance for MMD kernel parameter for variable {which_variable}..."
        )

        from fastabc_inversion.geo_problems.utils.evaluation.mmd import \
            estimate_median_pairwise_dists

        if which_variable not in ["x", "y", "z"]:
            raise ValueError("which_variable must be one of 'x', 'y', 'z'.")

        if which_variable in ["x", "y"]:
            large_sample_size = min(max_sample_size, self.experiment.train_size)
            # get large sample from training data
            random_indices = torch.randperm(self.experiment.train_size)[
                :large_sample_size
            ]

            all_train_x = (
                self.experiment.training_x[random_indices, :, :, :]
                if len(self.experiment.training_x.shape) > 2
                else self.experiment.training_x[random_indices, :]
            )
            all_train_y = self.experiment.training_y[random_indices, :]

            # make large generated samples
            large_latent_vector = (
                self.experiment.latent_dist(
                    torch.tensor(
                        self.experiment.latent_dist_params_list[0], dtype=torch.float32
                    ),
                    torch.tensor(
                        self.experiment.latent_dist_params_list[1], dtype=torch.float32
                    ),
                )
                .sample((large_sample_size, self.experiment.latent_dim))
                .to(self.experiment.device)
            )

            with torch.no_grad():
                gen_data = self.experiment.netG(large_latent_vector).cpu()
                gen_x = gen_data[:, 0 : self.experiment.dim_x]
                gen_y = gen_data[:, self.experiment.dim_x :]
            del large_latent_vector, gen_data

            if which_variable == "x":
                train_samples = all_train_x.numpy()
                gen_samples = gen_x.cpu().numpy()
                # stack train_samples and gen_samples for median heuristic
                all_data = np.vstack((train_samples, gen_samples))
                median_x = estimate_median_pairwise_dists(
                    all_data, sample_ratio=1.0, chunk_size=300
                )  # squared euclidean
                self.mmd_est_params["x"] = 1 / median_x
                del all_data, train_samples, gen_samples, all_train_x, gen_x

            if which_variable == "y":
                train_samples = all_train_y.numpy()
                gen_samples = gen_y.cpu().numpy()
                # stack train_samples and gen_samples for median heuristic
                all_data = np.vstack((train_samples, gen_samples))
                median_y = estimate_median_pairwise_dists(
                    all_data, sample_ratio=1.0, chunk_size=300
                )  # squared euclidean
                self.mmd_est_params["y"] = 1 / median_y
                del all_data, train_samples, gen_samples, all_train_y, gen_y

        else:  # which_variable == 'z'
            large_sample_size = min(max_sample_size, self.experiment.val_size)

            # get large sample from validation data
            random_indices = torch.randperm(self.experiment.val_size)[
                :large_sample_size
            ]

            all_train_latent_code = self.latent_train_codes[random_indices, :]
            all_val_latent_codes = self.latent_val_codes[random_indices, :]

            # make large generated samples
            large_latent_vector = (
                self.experiment.latent_dist(
                    torch.tensor(
                        self.experiment.latent_dist_params_list[0], dtype=torch.float32
                    ),
                    torch.tensor(
                        self.experiment.latent_dist_params_list[1], dtype=torch.float32
                    ),
                )
                .sample((large_sample_size, self.experiment.latent_dim))
                .numpy()
            )

            all_data = np.vstack(
                (all_train_latent_code, all_val_latent_codes, large_latent_vector)
            )
            median_z = estimate_median_pairwise_dists(
                all_data, sample_ratio=1.0, chunk_size=300
            )  # squared euclidean
            self.mmd_est_params["z"] = 1 / median_z
            del (
                all_data,
                all_train_latent_code,
                all_val_latent_codes,
                large_latent_vector,
            )

        # store estimated parameter on disk
        with open(
            self.logging_dir + f"/mmd_{which_variable}_kernel_param.txt", "w"
        ) as f:
            f.write(
                f"median squared euclidean : {str(self.mmd_est_params[which_variable])}"
            )

    def compute_ssim(self, image_subset_1, image_subset_2, dims, ssim_kwargs):
        """
        Compute the structural similarity index between two image subsets.
        :param image_subset_1: 2D numpy array of shape (number of observations, HxW)
        :param image_subset_2: 2D numpy array of same shape as image_subset_1. Assumes pairing with image_subset_1.
        :param dims: tuple of image dimensions (H, W)
        :param ssim_kwargs: dictionary of kwargs for the ssim function
        """
        n = image_subset_1.shape[0]
        h = dims[0]
        w = dims[1]
        ssim_values = np.zeros(n)

        for i in range(n):
            ssim_values[i] = ssim(
                image_subset_1[i].reshape((h, w), order="C"),
                image_subset_2[i].reshape((h, w), order="C"),
                **ssim_kwargs,
            )

        return ssim_values

    def compute_cosine_sim(self, subset_1, subset_2):
        """
        Compute the cosine similarity between two subsets of vectors.
        :param subset_1: 2D numpy array of shape (number of observations, vector dimension)
        :param subset_2: 2D numpy array of same shape as subset_1. Assumes pairing with subset_1.
        """
        n = subset_1.shape[0]
        cos_sim_values = np.zeros(n)

        for i in range(n):
            cos_sim_values[i] = cosine_similarity(
                subset_1[i, :].reshape(1, -1), subset_2[i, :].reshape(1, -1)
            )[0][0]

        return cos_sim_values

    def compare_random_projections(
        self, sample1, sample2, num_projections=100, ks_test_alpha=0.05
    ):
        """
        Compute 1d random projections of the samples and compare the distributions using the Kolmogorov-Smirnov test.
        :param sample1: first data sample of shape (number of observations, dimension of data)
        :param sample2: second data sample of shape (number of observations, dimension of data). Same shape as sample1.
        :param num_projections: number of random projections to generate
        :return:
        """
        from scipy.stats import ks_2samp
        from sklearn.random_projection import GaussianRandomProjection

        # n_dimensions = sample1.shape[1]
        differences = []  # final size = num_projections
        ks_tests_rejections = []  # final size = num_projections

        for _ in range(num_projections):
            # Generate a random projection
            # projection = np.random.randn(n_dimensions)
            # projection /= np.linalg.norm(projection)
            transformer = GaussianRandomProjection(n_components=1)

            # Project the samples
            s1_proj = transformer.fit_transform(sample1).reshape(-1)
            s2_proj = transformer.transform(sample2).reshape(-1)
            # s1_proj = sample1.dot(projection)
            # s2_proj = sample2.dot(projection)

            # Compare using KS test

            # H0: the two samples are drawn from the same distribution
            stat, p_value = ks_2samp(s1_proj, s2_proj)
            differences.append(stat)
            ks_tests_rejections.append(p_value < ks_test_alpha)  # H0 rejection decision

        differences = np.array(differences)
        ks_tests_rejections = np.array(ks_tests_rejections)

        return differences, ks_tests_rejections

    def make_priorvsgen_dists_distances_stats(
        self, mmd_params=None, two_sample_test_params=None, sample_size=300
    ):
        """
        Generates and computes statistical distances between prior (training) and generated data distributions.
        This function performs Maximum Mean Discrepancy (MMD) two-sample tests on the given data
        to evaluate the similarity between training and generated data distributions. It calculates
        summary statistics for MMD distances and p-values from the two-sample tests for both input (x)
        and output (y) data samples.

        :param mmd_params: Dictionary containing parameters for the MMD computation.
        :param two_sample_test_params: Dictionary containing parameters for two-sample testing.
        :param sample_size: The size of the sample subsets used for computations.
        :return: dictionary containing the results of the statistics to log.
        """
        print("Making prior vs generated distributions distance statistics...")
        precision = self.round_digits

        import copy

        mmd_params_x = copy.deepcopy(mmd_params)
        mmd_params_y = copy.deepcopy(mmd_params)

        two_sample_test_params_x = copy.deepcopy(two_sample_test_params)
        two_sample_test_params_y = copy.deepcopy(two_sample_test_params)

        # define large sample
        train_x = self.experiment.training_x
        train_y = self.experiment.training_y
        train_size = self.experiment.train_size

        # generate from jGNN
        latent_vector = (
            self.experiment.latent_dist(
                torch.tensor(
                    self.experiment.latent_dist_params_list[0], dtype=torch.float32
                ),
                torch.tensor(
                    self.experiment.latent_dist_params_list[1], dtype=torch.float32
                ),
            )
            .sample((self.experiment.train_size, self.experiment.latent_dim))
            .to(self.experiment.device)
        )

        with torch.no_grad():
            generated_data = self.experiment.netG(latent_vector).cpu()
            generated_x = generated_data[:, 0 : self.experiment.dim_x]
            generated_y = generated_data[:, self.experiment.dim_x :]
            del generated_data

        # for X
        mmd_x_dists = []
        mmd_x_refs = []
        two_sample_tests_x_pvalues = []

        all_x = np.vstack((train_x, generated_x))
        gamma_x = self.estimate_mmd_kernel_gamma(all_x)
        mmd_params_x["kernel_params"]["gamma"] = gamma_x
        two_sample_test_params_x["kernel_params"]["gamma"] = gamma_x
        self.mmd_est_params["x"] = gamma_x
        print(f"Using estimated MMD x kernel parameter gamma: {gamma_x}")
        del all_x

        # for Y
        mmd_y_dists = []
        mmd_y_refs = []
        two_sample_tests_y_pvalues = []

        all_y = np.vstack((train_y, generated_y))
        gamma_y = self.estimate_mmd_kernel_gamma(all_y)
        mmd_params_y["kernel_params"]["gamma"] = gamma_y
        two_sample_test_params_y["kernel_params"]["gamma"] = gamma_y
        self.mmd_est_params["y"] = gamma_y
        print(f"Using estimated MMD y kernel parameter gamma: {gamma_y}")
        del all_y

        # compute MMD and two-sample tests on data subsets of size sample_size
        num_repeats = train_size // sample_size
        for i in range(num_repeats):
            idx_start_i = i * sample_size
            idx_end_i = (i + 1) * sample_size

            train_x_sample = train_x[idx_start_i:idx_end_i, :]
            train_y_sample = train_y[idx_start_i:idx_end_i, :]

            print(f"Repeat {i + 1}/{num_repeats}...")
            for j in range(num_repeats):
                print(f"  Sub-repeat {j + 1}/{num_repeats}...")

                idx_start_j = j * sample_size
                idx_end_j = (j + 1) * sample_size

                print(
                    f"    Computing MMD and two-sample test for x data...gammas: "
                    f"{mmd_params_x['kernel_params']['gamma']}, {two_sample_test_params_x['kernel_params']['gamma']}"
                )
                generated_x_sample = generated_x[idx_start_j:idx_end_j, :]
                mmd_x_dists.append(
                    mmd(train_x_sample, generated_x_sample, **mmd_params_x)[0]
                )
                two_sample_tests_x_pvalues.append(
                    two_sample_mmd_test(
                        train_x_sample, generated_x_sample, **two_sample_test_params_x
                    )[1]
                )

                print(
                    f"    Computing MMD and two-sample test for y data..."
                    f"gammas: {mmd_params_y['kernel_params']['gamma']}, {two_sample_test_params_y['kernel_params']['gamma']}"
                )
                generated_y_sample = generated_y[idx_start_j:idx_end_j, :]
                mmd_y_dists.append(
                    mmd(train_y_sample, generated_y_sample, **mmd_params_y)[0]
                )
                two_sample_tests_y_pvalues.append(
                    two_sample_mmd_test(
                        train_y_sample, generated_y_sample, **two_sample_test_params_y
                    )[1]
                )

                # Compute reference MMD between different training chunks
                if j != i:
                    train_x_sample_2 = train_x[idx_start_j:idx_end_j, :]
                    train_y_sample_2 = train_y[idx_start_j:idx_end_j, :]

                    mmd_x_refs.append(
                        mmd(train_x_sample, train_x_sample_2, **mmd_params_x)[0]
                    )
                    mmd_y_refs.append(
                        mmd(train_y_sample, train_y_sample_2, **mmd_params_y)[0]
                    )

        mmd_x_dists = np.array(mmd_x_dists)
        mmd_x_refs = np.array(mmd_x_refs)
        mmd_y_dists = np.array(mmd_y_dists)
        mmd_y_refs = np.array(mmd_y_refs)
        two_sample_tests_x_pvalues = np.array(two_sample_tests_x_pvalues)
        two_sample_tests_y_pvalues = np.array(two_sample_tests_y_pvalues)

        self.results_vecs["mmd_x_dists"] = mmd_x_dists
        self.results_vecs["mmd_x_refs"] = mmd_x_refs
        self.results_vecs["mmd_y_dists"] = mmd_y_dists
        self.results_vecs["mmd_y_refs"] = mmd_y_refs
        self.results_vecs["two_sample_tests_x_pvalues"] = two_sample_tests_x_pvalues
        self.results_vecs["two_sample_tests_y_pvalues"] = two_sample_tests_y_pvalues

        # compute statistics
        mmd_x_dists_stats = self.compute_array_stats(mmd_x_dists)
        mmd_x_refs_stats = self.compute_array_stats(mmd_x_refs)
        mmd_y_dists_stats = self.compute_array_stats(mmd_y_dists)
        mmd_y_refs_stats = self.compute_array_stats(mmd_y_refs)

        # combined p_values
        from scipy.stats import chi2

        total_test_repeats = len(two_sample_tests_x_pvalues)

        combined_stat_x = -2 * np.sum(np.log(two_sample_tests_x_pvalues))
        combined_p_value_x = 1 - chi2.cdf(combined_stat_x, 2 * total_test_repeats)

        combined_stat_y = -2 * np.sum(np.log(two_sample_tests_y_pvalues))
        combined_p_value_y = 1 - chi2.cdf(combined_stat_y, 2 * total_test_repeats)

        # prepare data for logging
        write_combined_pvalue_two_sample_x = (
            f"{combined_p_value_x:.{precision}f}"
            if combined_p_value_x is not None
            else None
        )
        write_combined_pvalue_two_sample_x_txt = f"Combined p-value from two-sample MMD {total_test_repeats} test repetitions on x data (train vs gen)- sample size = {sample_size}:"
        write_header_combined_pvalue_two_sample_x = "combined_pvalue_two_sample_x"

        write_combined_pvalue_two_sample_y = (
            f"{combined_p_value_y:.{precision}f}"
            if combined_p_value_y is not None
            else None
        )
        write_combined_pvalue_two_sample_y_txt = f"Combined p-value from two-sample MMD {total_test_repeats} test repetitions on y data (train vs gen)- sample size = {sample_size}:"
        write_header_combined_pvalue_two_sample_y = "combined_pvalue_two_sample_y"

        write_data_mmd_x = self.make_array_stats_string(mmd_x_dists_stats, precision)
        write_data_mmd_x_txt = f"MMD distances between training and generated x data (est. with sample size={sample_size}*{len(mmd_x_dists)}):"
        write_header_mmd_x = "mmd_x_mean, std, mmd_x_median, [(mmd_x_q25, mmd_x_q75), (mmd_x_q025, mmd_x_q975)]"

        write_data_mmd_x_ref = self.make_array_stats_string(mmd_x_refs_stats, precision)
        write_data_mmd_x_ref_txt = f"Ref-MMD distances between two training x data samples (est. with sample size={sample_size}*{len(mmd_x_refs)}):"
        write_header_mmd_x_ref = f"mmd_x_ref_mean, mmd_x_ref_std, mmd_x_ref_median, [(mmd_x_ref_q25, mmd_x_ref_q75), (mmd_x_ref_q025, mmd_x_ref_q975)]"

        write_data_mmd_y = self.make_array_stats_string(mmd_y_dists_stats, precision)
        write_data_mmd_y_txt = f"MMD distances between training and generated y data (est. with sample size={sample_size}*{len(mmd_y_dists)}):"
        write_header_mmd_y = "mmd_y_mean, std, mmd_y_median, [(mmd_y_q25, mmd_y_q75), (mmd_y_q025, mmd_y_q975)]"

        write_data_mmd_y_ref = self.make_array_stats_string(mmd_y_refs_stats, precision)
        write_data_mmd_y_ref_txt = f"Ref-MMD distances between two training y data samples (est. with sample size={sample_size}*{len(mmd_y_refs)}):"
        write_header_mmd_y_ref = f"mmd_y_ref_mean, mmd_y_ref_std, mmd_y_ref_median, [(mmd_y_ref_q25, mmd_y_ref_q75), (mmd_y_ref_q025, mmd_y_ref_q975)]"

        # pickle X Y mmd restuls vecs
        import pickle

        with open(
            self.logging_dir + "/prior_vs_gen_dists_mmd_results_vecs.pkl", "wb"
        ) as f:
            pickle.dump(self.results_vecs, f)

        # store estimated gamma parameters on disk in .txt files
        with open(self.logging_dir + f"/mmd_x_kernel_param.txt", "w") as f:
            f.write(
                f"estimated MMD x kernel parameter gamma : {str(self.mmd_est_params['x'])}"
            )
        with open(self.logging_dir + f"/mmd_y_kernel_param.txt", "w") as f:
            f.write(
                f"estimated MMD y kernel parameter gamma : {str(self.mmd_est_params['y'])}"
            )

        return {
            "data": {
                "write_data_mmd_x": (write_data_mmd_x, write_data_mmd_x_txt),
                "write_data_mmd_y": (write_data_mmd_y, write_data_mmd_y_txt),
                "write_combined_pvalue_two_sample_x": (
                    write_combined_pvalue_two_sample_x,
                    write_combined_pvalue_two_sample_x_txt,
                ),
                "write_combined_pvalue_two_sample_y": (
                    write_combined_pvalue_two_sample_y,
                    write_combined_pvalue_two_sample_y_txt,
                ),
                "write_data_mmd_x_ref": (
                    write_data_mmd_x_ref,
                    write_data_mmd_x_ref_txt,
                ),
                "write_data_mmd_y_ref": (
                    write_data_mmd_y_ref,
                    write_data_mmd_y_ref_txt,
                ),
            },
            "headers": {
                "write_header_mmd_x": write_header_mmd_x,
                "write_header_mmd_y": write_header_mmd_y,
                "write_header_combined_pvalue_two_sample_x": write_header_combined_pvalue_two_sample_x,
                "write_header_combined_pvalue_two_sample_y": write_header_combined_pvalue_two_sample_y,
                "write_header_mmd_x_ref": write_header_mmd_x_ref,
                "write_header_mmd_y_ref": write_header_mmd_y_ref,
            },
        }

    def make_min_max_stats(
        self, build_y_stats=True, joint_output=True, make_plots=True
    ):
        """
        Make min and max statistics for all data.

        :param build_y_stats: if True, build min and max statistics for y data
        :param joint_output: whether the model being diagnosed has a joint output or not
        :param make_plots: if True, make plots of the statistics (not implemented yet)
        :return: dictionary of statistics formatted to be logged by write_results()
        """
        print("Making min and max statistics...")

        train_x = self.train_x
        train_y = self.train_y

        dim_x = self.experiment.dim_x
        dim_y = self.experiment.dim_y
        precision = self.round_digits

        if train_x is not None and train_y is not None:
            train_x = train_x.view(train_x.shape[0], dim_x)
            train_y = train_y.view(train_y.shape[0], dim_y)

            min_train_x_vec = torch.min(train_x, 1)[0].numpy()
            max_train_x_vec = torch.max(train_x, 1)[0].numpy()
            ranges_train_x_vec = max_train_x_vec - min_train_x_vec
            self.results_vecs["min_train_x_vec"] = min_train_x_vec
            self.results_vecs["max_train_x_vec"] = max_train_x_vec
            self.results_vecs["ranges_train_x_vec"] = ranges_train_x_vec

            min_train_x_vec_stats = self.compute_array_stats(min_train_x_vec)
            max_train_x_vec_stats = self.compute_array_stats(max_train_x_vec)
            ranges_train_x_vec_stats = self.compute_array_stats(ranges_train_x_vec)

            write_data_min_train_x = self.make_array_stats_string(
                min_train_x_vec_stats, precision
            )
            write_data_min_train_x_txt = "Training models MIN stats:"
            write_header_min_train_x = "avg_min_train_x_vec, median_min_train_x_vec, (q25_min_train_x_vec, q75_min_train_x_vec), (q025_min_train_x_vec, q975_min_train_x_vec)"

            write_data_max_train_x = self.make_array_stats_string(
                max_train_x_vec_stats, precision
            )
            write_data_max_train_x_txt = "Training models MAX stats:"
            write_header_max_train_x = "avg_max_train_x_vec, median_max_train_x_vec, (q25_max_train_x_vec, q75_max_train_x_vec), (q025_max_train_x_vec, q975_max_train_x_vec)"

            write_data_ranges_train_x = self.make_array_stats_string(
                ranges_train_x_vec_stats, precision
            )
            write_data_ranges_train_x_txt = "Training models RANGES stats:"
            write_header_ranges_train_x = "avg_ranges_train_x_vec, sd_ranges_train_x_vec, median_ranges_train_x_vec, (q25_ranges_train_x_vec, q75_ranges_train_x_vec), (q025_ranges_train_x_vec, q975_ranges_train_x_vec)"

            if build_y_stats:
                min_train_y_vec = torch.min(train_y, 1)[0].numpy()
                max_train_y_vec = torch.max(train_y, 1)[0].numpy()
                ranges_train_y_vec = max_train_y_vec - min_train_y_vec

            else:
                min_train_y_vec = max_train_y_vec = ranges_train_y_vec = None

            self.results_vecs["min_train_y_vec"] = min_train_y_vec
            self.results_vecs["max_train_y_vec"] = max_train_y_vec
            self.results_vecs["ranges_train_y_vec"] = ranges_train_y_vec

            min_train_y_vec_stats = self.compute_array_stats(min_train_y_vec)
            max_train_y_vec_stats = self.compute_array_stats(max_train_y_vec)
            ranges_train_y_vec_stats = self.compute_array_stats(ranges_train_y_vec)

            write_data_min_train_y = self.make_array_stats_string(
                min_train_y_vec_stats, precision
            )
            write_data_min_train_y_txt = "Training tt MIN stats:"
            write_header_min_train_y = "avg_min_train_y_vec, median_min_train_y_vec, (q25_min_train_y_vec, q75_min_train_y_vec), (q025_min_train_y_vec, q975_min_train_y_vec)"

            write_data_max_train_y = self.make_array_stats_string(
                max_train_y_vec_stats, precision
            )
            write_data_max_train_y_txt = "Training tt MAX stats:"
            write_header_max_train_y = "avg_max_train_y_vec, median_max_train_y_vec, (q25_max_train_y_vec, q75_max_train_y_vec), (q025_max_train_y_vec, q975_max_train_y_vec)"

            write_data_ranges_train_y = self.make_array_stats_string(
                ranges_train_y_vec_stats, precision
            )
            write_data_ranges_train_y_txt = "Training tt RANGES stats:"
            write_header_ranges_train_y = "avg_ranges_train_y_vec, sd_ranges_train_y_vec, median_ranges_train_y_vec, (q25_ranges_train_y_vec, q75_ranges_train_y_vec), (q025_ranges_train_y_vec, q975_ranges_train_y_vec)"

        else:
            raise ValueError(
                "No training data to compute min and max statistics."
                "Load data into the experiment object first."
            )

        if self.generated_x is not None and self.generated_y is not None:
            min_gen_x_vec = torch.min(self.generated_x, 1)[0].numpy()
            max_gen_x_vec = torch.max(self.generated_x, 1)[0].numpy()
            ranges_gen_x_vec = max_gen_x_vec - min_gen_x_vec

            self.results_vecs["min_gen_x_vec"] = min_gen_x_vec
            self.results_vecs["max_gen_x_vec"] = max_gen_x_vec
            self.results_vecs["ranges_gen_x_vec"] = ranges_gen_x_vec

            min_gen_x_vec_stats = self.compute_array_stats(min_gen_x_vec)
            max_gen_x_vec_stats = self.compute_array_stats(max_gen_x_vec)
            ranges_gen_x_vec_stats = self.compute_array_stats(ranges_gen_x_vec)

            write_data_min_gen_x = self.make_array_stats_string(
                min_gen_x_vec_stats, precision
            )
            write_data_min_gen_x_txt = "Generated models MIN stats:"
            write_header_min_gen_x = "avg_min_gen_x_vec, median_min_gen_x_vec, (q25_min_gen_x_vec, q75_min_gen_x_vec), (q025_min_gen_x_vec, q975_min_gen_x_vec)"

            write_data_max_gen_x = self.make_array_stats_string(
                max_gen_x_vec_stats, precision
            )
            write_data_max_gen_x_txt = "Generated models MAX stats:"
            write_header_max_gen_x = "avg_max_gen_x_vec, median_max_gen_x_vec, (q25_max_gen_x_vec, q75_max_gen_x_vec), (q025_max_gen_x_vec, q975_max_gen_x_vec)"

            write_data_ranges_gen_x = self.make_array_stats_string(
                ranges_gen_x_vec_stats, precision
            )
            write_data_ranges_gen_x_txt = "Generated models RANGES stats:"
            write_header_ranges_gen_x = "avg_ranges_gen_x_vec, sd_ranges_gen_x_vec, median_ranges_gen_x_vec, (q25_ranges_gen_x_vec, q75_ranges_gen_x_vec), (q025_ranges_gen_x_vec, q975_ranges_gen_x_vec)"

            if build_y_stats:
                min_gen_y_vec = torch.min(self.generated_y, 1)[0].numpy()
                max_gen_y_vec = torch.max(self.generated_y, 1)[0].numpy()
                ranges_gen_y_vec = max_gen_y_vec - min_gen_y_vec

            else:
                min_gen_y_vec = max_gen_y_vec = ranges_gen_y_vec = None

            self.results_vecs["min_gen_y_vec"] = min_gen_y_vec
            self.results_vecs["max_gen_y_vec"] = max_gen_y_vec
            self.results_vecs["ranges_gen_y_vec"] = ranges_gen_y_vec

            min_gen_y_vec_stats = self.compute_array_stats(min_gen_y_vec)
            max_gen_y_vec_stats = self.compute_array_stats(max_gen_y_vec)
            ranges_gen_y_vec_stats = self.compute_array_stats(ranges_gen_y_vec)

            write_data_min_gen_y = self.make_array_stats_string(
                min_gen_y_vec_stats, precision
            )
            write_data_min_gen_y_txt = "Generated tt MIN stats:"
            write_header_min_gen_y = "avg_min_gen_y_vec, median_min_gen_y_vec, (q25_min_gen_y_vec, q75_min_gen_y_vec), (q025_min_gen_y_vec, q975_min_gen_y_vec)"

            write_data_max_gen_y = self.make_array_stats_string(
                max_gen_y_vec_stats, precision
            )
            write_data_max_gen_y_txt = "Generated tt MAX stats:"
            write_header_max_gen_y = "avg_max_gen_y_vec, median_max_gen_y_vec, (q25_max_gen_y_vec, q75_max_gen_y_vec), (q025_max_gen_y_vec, q975_max_gen_y_vec)"

            write_data_ranges_gen_y = self.make_array_stats_string(
                ranges_gen_y_vec_stats, precision
            )
            write_data_ranges_gen_y_txt = "Generated tt RANGES stats:"
            write_header_ranges_gen_y = "avg_ranges_gen_y_vec, sd_ranges_gen_y_vec, median_ranges_gen_y_vec, (q25_ranges_gen_y_vec, q75_ranges_gen_y_vec), (q025_ranges_gen_y_vec, q975_ranges_gen_y_vec)"

        else:
            raise ValueError(
                "No generated data to compute min and max statistics."
                "Build diagnostics data first."
            )

        if (
            self.reconstructed_train_x is not None
            and self.reconstructed_train_y is not None
        ):
            min_recon_train_x_vec = torch.min(self.reconstructed_train_x, 1)[0].numpy()
            max_recon_train_x_vec = torch.max(self.reconstructed_train_x, 1)[0].numpy()
            ranges_recon_train_x_vec = max_recon_train_x_vec - min_recon_train_x_vec

            self.results_vecs["min_recon_train_x_vec"] = min_recon_train_x_vec
            self.results_vecs["max_recon_train_x_vec"] = max_recon_train_x_vec
            self.results_vecs["ranges_recon_train_x_vec"] = ranges_recon_train_x_vec

            min_recon_train_x_vec_stats = self.compute_array_stats(
                min_recon_train_x_vec
            )
            max_recon_train_x_vec_stats = self.compute_array_stats(
                max_recon_train_x_vec
            )
            ranges_recon_train_x_vec_stats = self.compute_array_stats(
                ranges_recon_train_x_vec
            )

            write_data_min_recon_train_x = self.make_array_stats_string(
                min_recon_train_x_vec_stats, precision
            )
            write_data_min_recon_train_x_txt = (
                "Reconstructed training models MIN stats:"
            )
            write_header_min_recon_train_x = "avg_min_recon_train_x_vec, median_min_recon_train_x_vec, (q25_min_recon_train_x_vec, q75_min_recon_train_x_vec), (q025_min_recon_train_x_vec, q975_min_recon_train_x_vec)"

            write_data_max_recon_train_x = self.make_array_stats_string(
                max_recon_train_x_vec_stats, precision
            )
            write_data_max_recon_train_x_txt = (
                "Reconstructed training models MAX stats:"
            )
            write_header_max_recon_train_x = "avg_max_recon_train_x_vec, median_max_recon_train_x_vec, (q25_max_recon_train_x_vec, q75_max_recon_train_x_vec), (q025_max_recon_train_x_vec, q975_max_recon_train_x_vec)"

            write_data_ranges_recon_train_x = self.make_array_stats_string(
                ranges_recon_train_x_vec_stats, precision
            )
            write_data_ranges_recon_train_x_txt = (
                "Reconstructed training models RANGES stats:"
            )
            write_header_ranges_recon_train_x = "avg_ranges_recon_train_x_vec, sd_ranges_recon_train_x_vec, median_ranges_recon_train_x_vec, (q25_ranges_recon_train_x_vec, q75_ranges_recon_train_x_vec), (q025_ranges_recon_train_x_vec, q975_ranges_recon_train_x_vec)"

            if build_y_stats:
                min_recon_train_y_vec = torch.min(self.reconstructed_train_y, 1)[
                    0
                ].numpy()
                max_recon_train_y_vec = torch.max(self.reconstructed_train_y, 1)[
                    0
                ].numpy()
                ranges_recon_train_y_vec = max_recon_train_y_vec - min_recon_train_y_vec

            else:
                min_recon_train_y_vec = (
                    max_recon_train_y_vec
                ) = ranges_recon_train_y_vec = None

            self.results_vecs["min_recon_train_y_vec"] = min_recon_train_y_vec
            self.results_vecs["max_recon_train_y_vec"] = max_recon_train_y_vec
            self.results_vecs["ranges_recon_train_y_vec"] = ranges_recon_train_y_vec

            min_recon_train_y_vec_stats = self.compute_array_stats(
                min_recon_train_y_vec
            )
            max_recon_train_y_vec_stats = self.compute_array_stats(
                max_recon_train_y_vec
            )
            ranges_recon_train_y_vec_stats = self.compute_array_stats(
                ranges_recon_train_y_vec
            )

            write_data_min_recon_train_y = self.make_array_stats_string(
                min_recon_train_y_vec_stats, precision
            )
            write_data_min_recon_train_y_txt = "Reconstructed training tt MIN stats:"
            write_header_min_recon_train_y = "avg_min_recon_train_y_vec, median_min_recon_train_y_vec, (q25_min_recon_train_y_vec, q75_min_recon_train_y_vec), (q025_min_recon_train_y_vec, q975_min_recon_train_y_vec)"

            write_data_max_recon_train_y = self.make_array_stats_string(
                max_recon_train_y_vec_stats, precision
            )
            write_data_max_recon_train_y_txt = "Reconstructed training tt MAX stats:"
            write_header_max_recon_train_y = "avg_max_recon_train_y_vec, median_max_recon_train_y_vec, (q25_max_recon_train_y_vec, q75_max_recon_train_y_vec), (q025_max_recon_train_y_vec, q975_max_recon_train_y_vec)"

            write_data_ranges_recon_train_y = self.make_array_stats_string(
                ranges_recon_train_y_vec_stats, precision
            )
            write_data_ranges_recon_train_y_txt = (
                "Reconstructed training tt RANGES stats:"
            )
            write_header_ranges_recon_train_y = "avg_ranges_recon_train_y_vec, sd_ranges_recon_train_y_vec, median_ranges_recon_train_y_vec, (q25_ranges_recon_train_y_vec, q75_ranges_recon_train_y_vec), (q025_ranges_recon_train_y_vec, q975_ranges_recon_train_y_vec)"

        else:
            raise ValueError(
                "No reconstructed training data to compute min and max statistics."
                "Build diagnostics data first."
            )

        if (
            self.reconstructed_val_x is not None
            and self.reconstructed_val_y is not None
        ):
            min_recon_val_x_vec = torch.min(self.reconstructed_val_x, 1)[0].numpy()
            max_recon_val_x_vec = torch.max(self.reconstructed_val_x, 1)[0].numpy()
            ranges_recon_val_x_vec = max_recon_val_x_vec - min_recon_val_x_vec

            self.results_vecs["min_recon_val_x_vec"] = min_recon_val_x_vec
            self.results_vecs["max_recon_val_x_vec"] = max_recon_val_x_vec
            self.results_vecs["ranges_recon_val_x_vec"] = ranges_recon_val_x_vec

            min_recon_val_x_vec_stats = self.compute_array_stats(min_recon_val_x_vec)
            max_recon_val_x_vec_stats = self.compute_array_stats(max_recon_val_x_vec)
            ranges_recon_val_x_vec_stats = self.compute_array_stats(
                ranges_recon_val_x_vec
            )

            write_data_min_recon_val_x = self.make_array_stats_string(
                min_recon_val_x_vec_stats, precision
            )
            write_data_min_recon_val_x_txt = (
                "Reconstructed validation models MIN stats:"
            )
            write_header_min_recon_val_x = "avg_min_recon_val_x_vec, median_min_recon_val_x_vec, (q25_min_recon_val_x_vec, q75_min_recon_val_x_vec), (q025_min_recon_val_x_vec, q975_min_recon_val_x_vec)"

            write_data_max_recon_val_x = self.make_array_stats_string(
                max_recon_val_x_vec_stats, precision
            )
            write_data_max_recon_val_x_txt = (
                "Reconstructed validation models MAX stats:"
            )
            write_header_max_recon_val_x = "avg_max_recon_val_x_vec, median_max_recon_val_x_vec, (q25_max_recon_val_x_vec, q75_max_recon_val_x_vec), (q025_max_recon_val_x_vec, q975_max_recon_val_x_vec)"

            write_data_ranges_recon_val_x = self.make_array_stats_string(
                ranges_recon_val_x_vec_stats, precision
            )
            write_data_ranges_recon_val_x_txt = (
                "Reconstructed validation models RANGES stats:"
            )
            write_header_ranges_recon_val_x = "avg_ranges_recon_val_x_vec, sd_ranges_recon_val_x_vec, median_ranges_recon_val_x_vec, (q25_ranges_recon_val_x_vec, q75_ranges_recon_val_x_vec), (q025_ranges_recon_val_x_vec, q975_ranges_recon_val_x_vec)"

            if build_y_stats:
                min_recon_val_y_vec = torch.min(self.reconstructed_val_y, 1)[0].numpy()
                max_recon_val_y_vec = torch.max(self.reconstructed_val_y, 1)[0].numpy()
                ranges_recon_val_y_vec = max_recon_val_y_vec - min_recon_val_y_vec
            else:
                min_recon_val_y_vec = (
                    max_recon_val_y_vec
                ) = ranges_recon_val_y_vec = None

            self.results_vecs["min_recon_val_y_vec"] = min_recon_val_y_vec
            self.results_vecs["max_recon_val_y_vec"] = max_recon_val_y_vec
            self.results_vecs["ranges_recon_val_y_vec"] = ranges_recon_val_y_vec

            min_recon_val_y_vec_stats = self.compute_array_stats(min_recon_val_y_vec)
            max_recon_val_y_vec_stats = self.compute_array_stats(max_recon_val_y_vec)
            ranges_recon_val_y_vec_stats = self.compute_array_stats(
                ranges_recon_val_y_vec
            )

            write_data_min_recon_val_y = self.make_array_stats_string(
                min_recon_val_y_vec_stats, precision
            )
            write_data_min_recon_val_y_txt = "Reconstructed validation tt MIN stats:"
            write_header_min_recon_val_y = "avg_min_recon_val_y_vec, median_min_recon_val_y_vec, (q25_min_recon_val_y_vec, q75_min_recon_val_y_vec), (q025_min_recon_val_y_vec, q975_min_recon_val_y_vec)"

            write_data_max_recon_val_y = self.make_array_stats_string(
                max_recon_val_y_vec_stats, precision
            )
            write_data_max_recon_val_y_txt = "Reconstructed validation tt MAX stats:"
            write_header_max_recon_val_y = "avg_max_recon_val_y_vec, median_max_recon_val_y_vec, (q25_max_recon_val_y_vec, q75_max_recon_val_y_vec), (q025_max_recon_val_y_vec, q975_max_recon_val_y_vec)"

            write_data_ranges_recon_val_y = self.make_array_stats_string(
                ranges_recon_val_y_vec_stats, precision
            )
            write_data_ranges_recon_val_y_txt = (
                "Reconstructed validation tt RANGES stats:"
            )
            write_header_ranges_recon_val_y = "avg_ranges_recon_val_y_vec, sd_ranges_recon_val_y_vec, median_ranges_recon_val_y_vec, (q25_ranges_recon_val_y_vec, q75_ranges_recon_val_y_vec), (q025_ranges_recon_val_y_vec, q975_ranges_recon_val_y_vec)"

        else:
            raise ValueError(
                "No reconstructed validation data to compute min and max statistics."
                "Build diagnostics data first."
            )

        return {
            "data": {
                "write_data_min_train_x": (
                    write_data_min_train_x,
                    write_data_min_train_x_txt,
                ),
                "write_data_max_train_x": (
                    write_data_max_train_x,
                    write_data_max_train_x_txt,
                ),
                "write_data_ranges_train_x": (
                    write_data_ranges_train_x,
                    write_data_ranges_train_x_txt,
                ),
                "write_data_min_gen_x": (
                    write_data_min_gen_x,
                    write_data_min_gen_x_txt,
                ),
                "write_data_max_gen_x": (
                    write_data_max_gen_x,
                    write_data_max_gen_x_txt,
                ),
                "write_data_ranges_gen_x": (
                    write_data_ranges_gen_x,
                    write_data_ranges_gen_x_txt,
                ),
                "write_data_min_recon_train_x": (
                    write_data_min_recon_train_x,
                    write_data_min_recon_train_x_txt,
                ),
                "write_data_max_recon_train_x": (
                    write_data_max_recon_train_x,
                    write_data_max_recon_train_x_txt,
                ),
                "write_data_ranges_recon_train_x": (
                    write_data_ranges_recon_train_x,
                    write_data_ranges_recon_train_x_txt,
                ),
                "write_data_min_recon_val_x": (
                    write_data_min_recon_val_x,
                    write_data_min_recon_val_x_txt,
                ),
                "write_data_max_recon_val_x": (
                    write_data_max_recon_val_x,
                    write_data_max_recon_val_x_txt,
                ),
                "write_data_ranges_recon_val_x": (
                    write_data_ranges_recon_val_x,
                    write_data_ranges_recon_val_x_txt,
                ),
                "write_data_min_train_y": (
                    write_data_min_train_y,
                    write_data_min_train_y_txt,
                ),
                "write_data_max_train_y": (
                    write_data_max_train_y,
                    write_data_max_train_y_txt,
                ),
                "write_data_ranges_train_y": (
                    write_data_ranges_train_y,
                    write_data_ranges_train_y_txt,
                ),
                "write_data_min_gen_y": (
                    write_data_min_gen_y,
                    write_data_min_gen_y_txt,
                ),
                "write_data_max_gen_y": (
                    write_data_max_gen_y,
                    write_data_max_gen_y_txt,
                ),
                "write_data_ranges_gen_y": (
                    write_data_ranges_gen_y,
                    write_data_ranges_gen_y_txt,
                ),
                "write_data_min_recon_train_y": (
                    write_data_min_recon_train_y,
                    write_data_min_recon_train_y_txt,
                ),
                "write_data_max_recon_train_y": (
                    write_data_max_recon_train_y,
                    write_data_max_recon_train_y_txt,
                ),
                "write_data_ranges_recon_train_y": (
                    write_data_ranges_recon_train_y,
                    write_data_ranges_recon_train_y_txt,
                ),
                "write_data_min_recon_val_y": (
                    write_data_min_recon_val_y,
                    write_data_min_recon_val_y_txt,
                ),
                "write_data_max_recon_val_y": (
                    write_data_max_recon_val_y,
                    write_data_max_recon_val_y_txt,
                ),
                "write_data_ranges_recon_val_y": (
                    write_data_ranges_recon_val_y,
                    write_data_ranges_recon_val_y_txt,
                ),
            },
            "headers": {
                "write_header_min_train_x": write_header_min_train_x,
                "write_header_max_train_x": write_header_max_train_x,
                "write_header_ranges_train_x": write_header_ranges_train_x,
                "write_header_min_gen_x": write_header_min_gen_x,
                "write_header_max_gen_x": write_header_max_gen_x,
                "write_header_ranges_gen_x": write_header_ranges_gen_x,
                "write_header_min_recon_train_x": write_header_min_recon_train_x,
                "write_header_max_recon_train_x": write_header_max_recon_train_x,
                "write_header_ranges_recon_train_x": write_header_ranges_recon_train_x,
                "write_header_min_recon_val_x": write_header_min_recon_val_x,
                "write_header_max_recon_val_x": write_header_max_recon_val_x,
                "write_header_ranges_recon_val_x": write_header_ranges_recon_val_x,
                "write_header_min_train_y": write_header_min_train_y,
                "write_header_max_train_y": write_header_max_train_y,
                "write_header_ranges_train_y": write_header_ranges_train_y,
                "write_header_min_gen_y": write_header_min_gen_y,
                "write_header_max_gen_y": write_header_max_gen_y,
                "write_header_ranges_gen_y": write_header_ranges_gen_y,
                "write_header_min_recon_train_y": write_header_min_recon_train_y,
                "write_header_max_recon_train_y": write_header_max_recon_train_y,
                "write_header_ranges_recon_train_y": write_header_ranges_recon_train_y,
                "write_header_min_recon_val_y": write_header_min_recon_val_y,
                "write_header_max_recon_val_y": write_header_max_recon_val_y,
                "write_header_ranges_recon_val_y": write_header_ranges_recon_val_y,
            },
        }

    def make_pixel_stats(
        self,
        build_y_stats=True,
        joint_output=True,
        make_plots=True,
        n_neighbors=100,
        min_dist=0.3,
        denseMAP=True,
    ):
        """
        Make mean, std and iqr statistics for all data.

        :param build_y_stats: if True, build mean, std and iqr statistics for y data
        :param joint_output: whether the model being diagnosed has a joint output or not
        :param make_plots: if True, make plots of the statistics
        :param n_neighbors: number of neighbors to consider for UMAP embedding
        :param min_dist: minimum distance between points in UMAP embedding
        :return: dictionary of statistics formatted to be logged by write_results()
        """
        print("Making pixel statistics...")

        train_x = self.train_x
        train_y = self.train_y

        val_x = self.val_x
        val_y = self.val_y

        dim_x = self.experiment.dim_x
        dim_y = self.experiment.dim_y

        precision = self.round_digits

        if (
            train_x is not None
            and train_y is not None
            and val_x is not None
            and val_y is not None
        ):
            train_x = train_x.view(train_x.shape[0], dim_x)
            train_y = train_y.view(train_y.shape[0], dim_y)

            val_x = val_x.view(val_x.shape[0], dim_x)
            val_y = val_y.view(val_y.shape[0], dim_y)

            mean_train_x_vec = torch.mean(train_x, 0).numpy()  # pixel wise
            self.results_vecs["mean_train_x_vec"] = mean_train_x_vec

            mean_train_x_vec_stats = self.compute_array_stats(mean_train_x_vec)

            write_data_pixel_mean_train_x = self.make_array_stats_string(
                mean_train_x_vec_stats, precision
            )
            write_data_pixel_mean_train_x_txt = "Ref_ Training models pixels mean: "
            write_header_pixel_mean_train_x = "mean_train_x,median_train_x,[(q25_train_x,q75_train_x),(q025_train_x,q975_train_x)]"

            sd_train_x_vec = torch.std(train_x, 0).numpy()  # pixel wise
            self.results_vecs["sd_train_x_vec"] = sd_train_x_vec

            sd_train_x_vec_stats = self.compute_array_stats(sd_train_x_vec)

            write_data_pixel_sd_train_x = self.make_array_stats_string(
                sd_train_x_vec_stats, precision
            )
            write_data_pixel_sd_train_x_txt = "Ref_ Average std on training pixels: "
            write_header_pixel_sd_train_x = "sd_train_x,median_sd_train_x,[(q25_sd_train_x,q75_sd_train_x),(q025_sd_train_x,q975_sd_train_x)]"

            covm_train_x = np.cov(train_x.numpy(), rowvar=False)
            self.results_vecs["covm_train_x"] = covm_train_x

            mean_val_x_vec = torch.mean(val_x, 0).numpy()  # pixel wise
            self.results_vecs["mean_val_x_vec"] = mean_val_x_vec

            mean_val_x_vec_stats = self.compute_array_stats(mean_val_x_vec)

            write_data_pixel_mean_val_x = self.make_array_stats_string(
                mean_val_x_vec_stats, precision
            )
            write_data_pixel_mean_val_x_txt = "Ref_ Validation models pixels mean: "
            write_header_pixel_mean_val_x = "mean_val_x,median_val_x,[(q25_val_x,q75_val_x),(q025_val_x,q975_val_x)]"

            sd_val_x_vec = torch.std(val_x, 0).numpy()  # pixel wise
            self.results_vecs["sd_val_x_vec"] = sd_val_x_vec

            sd_val_x_vec_stats = self.compute_array_stats(sd_val_x_vec)

            write_data_pixel_sd_val_x = self.make_array_stats_string(
                sd_val_x_vec_stats, precision
            )
            write_data_pixel_sd_val_x_txt = "Ref_ Average std on validation pixels: "
            write_header_pixel_sd_val_x = "sd_val_x,median_sd_val_x,[(q25_sd_val_x,q75_sd_val_x),(q025_sd_val_x,q975_sd_val_x)]"

            covm_val_x = np.cov(val_x.numpy(), rowvar=False)
            self.results_vecs["covm_val_x"] = covm_val_x

            if build_y_stats:
                mean_train_y_vec = torch.mean(train_y, 0).numpy()
                sd_train_y_vec = torch.std(train_y, 0).numpy()
                covm_train_y = np.cov(train_y.numpy(), rowvar=False)
                mean_val_y_vec = torch.mean(val_y, 0).numpy()
                sd_val_y_vec = torch.std(val_y, 0).numpy()
                covm_val_y = np.cov(val_y.numpy(), rowvar=False)
            else:
                mean_train_y_vec = (
                    sd_train_y_vec
                ) = covm_train_y = mean_val_y_vec = sd_val_y_vec = covm_val_y = None

            self.results_vecs["mean_train_y_vec"] = mean_train_y_vec
            self.results_vecs["sd_train_y_vec"] = sd_train_y_vec
            self.results_vecs["covm_train_y"] = covm_train_y
            self.results_vecs["mean_val_y_vec"] = mean_val_y_vec
            self.results_vecs["sd_val_y_vec"] = sd_val_y_vec
            self.results_vecs["covm_val_y"] = covm_val_y

            mean_train_y_vec_stats = self.compute_array_stats(mean_train_y_vec)
            sd_train_y_vec_stats = self.compute_array_stats(sd_train_y_vec)
            mean_val_y_vec_stats = self.compute_array_stats(mean_val_y_vec)
            sd_val_y_vec_stats = self.compute_array_stats(sd_val_y_vec)

            write_data_pixel_mean_train_y = self.make_array_stats_string(
                mean_train_y_vec_stats, precision
            )
            write_data_pixel_mean_train_y_txt = "Ref_ Training travel times mean: "
            write_header_pixel_mean_train_y = "mean_train_y,median_train_y,[(q25_train_y,q75_train_y),(q025_train_y,q975_train_y)]"

            write_data_pixel_sd_train_y = self.make_array_stats_string(
                sd_train_y_vec_stats, precision
            )
            write_data_pixel_sd_train_y_txt = (
                "Ref_ Average std on training travel times: "
            )
            write_header_pixel_sd_train_y = "sd_train_y,median_sd_train_y,[(q25_sd_train_y,q75_sd_train_y),(q025_sd_train_y,q975_sd_train_y)]"

            write_data_pixel_mean_val_y = self.make_array_stats_string(
                mean_val_y_vec_stats, precision
            )
            write_data_pixel_mean_val_y_txt = "Ref_ Validation travel times mean: "
            write_header_pixel_mean_val_y = "mean_val_y,median_val_y,[(q25_val_y,q75_val_y),(q025_val_y,q975_val_y)]"

            write_data_pixel_sd_val_y = self.make_array_stats_string(
                sd_val_y_vec_stats, precision
            )
            write_data_pixel_sd_val_y_txt = (
                "Ref_ Average std on validation travel times: "
            )
            write_header_pixel_sd_val_y = "sd_val_y,median_sd_val_y,[(q25_sd_val_y,q75_sd_val_y),(q025_sd_val_y,q975_sd_val_y)]"

        else:
            raise ValueError(
                "No training data to compute mean and std statistics."
                "Run build_diagnostics_data() first."
            )
        if self.generated_x is not None and self.generated_y is not None:
            mean_gen_x_vec = torch.mean(self.generated_x, 0).numpy()
            self.results_vecs["mean_gen_x_vec"] = mean_gen_x_vec

            mean_gen_x_vec_stats = self.compute_array_stats(mean_gen_x_vec)

            write_data_pixel_mean_gen_x = self.make_array_stats_string(
                mean_gen_x_vec_stats, precision
            )
            write_data_pixel_mean_gen_x_txt = "Generated models pixels mean: "
            write_header_pixel_mean_gen_x = "mean_gen_x,median_gen_x,[(q25_gen_x,q75_gen_x),(q025_gen_x,q975_gen_x)]"

            sd_gen_x_vec = torch.std(self.generated_x, 0).numpy()
            self.results_vecs["sd_gen_x_vec"] = sd_gen_x_vec

            sd_gen_x_vec_stats = self.compute_array_stats(sd_gen_x_vec)

            write_data_pixel_sd_gen_x = self.make_array_stats_string(
                sd_gen_x_vec_stats, precision
            )
            write_data_pixel_sd_gen_x_txt = "Average std on generated models pixels: "
            write_header_pixel_sd_gen_x = "sd_gen_x,median_sd_gen_x,[(q25_sd_gen_x,q75_sd_gen_x),(q025_sd_gen_x,q975_sd_gen_x)]"

            covm_gen_x = np.cov(self.generated_x.numpy(), rowvar=False)
            self.results_vecs["covm_gen_x"] = covm_gen_x

            if build_y_stats:
                mean_gen_y_vec = torch.mean(self.generated_y, 0).numpy()
                sd_gen_y_vec = torch.std(self.generated_y, 0).numpy()
                covm_gen_y = np.cov(self.generated_y.numpy(), rowvar=False)
            else:
                mean_gen_y_vec = sd_gen_y_vec = covm_gen_y = None

            self.results_vecs["mean_gen_y_vec"] = mean_gen_y_vec
            self.results_vecs["sd_gen_y_vec"] = sd_gen_y_vec
            self.results_vecs["covm_gen_y"] = covm_gen_y

            mean_gen_y_vec_stats = self.compute_array_stats(mean_gen_y_vec)
            sd_gen_y_vec_stats = self.compute_array_stats(sd_gen_y_vec)

            write_data_pixel_mean_gen_y = self.make_array_stats_string(
                mean_gen_y_vec_stats, precision
            )
            write_data_pixel_mean_gen_y_txt = "Generated travel times mean: "
            write_header_pixel_mean_gen_y = "mean_gen_y,median_gen_y,[(q25_gen_y,q75_gen_y),(q025_gen_y,q975_gen_y)]"

            write_data_pixel_sd_gen_y = self.make_array_stats_string(
                sd_gen_y_vec_stats, precision
            )
            write_data_pixel_sd_gen_y_txt = "Average std on generated travel times: "
            write_header_pixel_sd_gen_y = "sd_gen_y,median_sd_gen_y,[(q25_sd_gen_y,q75_sd_gen_y),(q025_sd_gen_y,q975_sd_gen_y)]"

        else:
            raise ValueError(
                "No generated data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        if (
            self.reconstructed_train_x is not None
            and self.reconstructed_train_y is not None
        ):
            mean_recon_train_x_vec = torch.mean(self.reconstructed_train_x, 0).numpy()
            self.results_vecs["mean_recon_train_x_vec"] = mean_recon_train_x_vec

            mean_recon_train_x_vec_stats = self.compute_array_stats(
                mean_recon_train_x_vec
            )

            write_data_recon_train_pixel_mean_x = self.make_array_stats_string(
                mean_recon_train_x_vec_stats, precision
            )
            write_data_recon_train_pixel_mean_x_txt = (
                "Reconstructed training models pixels mean: "
            )
            write_header_recon_train_pixel_mean_x = "mean_recon_train_x,median_recon_train_x,(q25_recon_train_x,q75_recon_train_x),(q025_recon_train_x,q975_recon_train_x)"

            sd_recon_train_x_vec = torch.std(self.reconstructed_train_x, 0).numpy()
            self.results_vecs["sd_recon_train_x_vec"] = sd_recon_train_x_vec

            sd_recon_train_x_vec_stats = self.compute_array_stats(sd_recon_train_x_vec)

            write_data_recon_train_pixel_sd_x = self.make_array_stats_string(
                sd_recon_train_x_vec_stats, precision
            )
            write_data_recon_train_pixel_sd_x_txt = (
                "Average std on reconstructed training models pixels: "
            )
            write_header_recon_train_pixel_sd_x = "sd_recon_train_x,median_sd_recon_train_x,[(q25_sd_recon_train_x,q75_sd_recon_train_x),(q025_sd_recon_train_x,q975_sd_recon_train_x)]"

            covm_recon_train_x = np.cov(
                self.reconstructed_train_x.numpy(), rowvar=False
            )
            self.results_vecs["covm_recon_train_x"] = covm_recon_train_x

            if build_y_stats:
                mean_recon_train_y_vec = torch.mean(
                    self.reconstructed_train_y, 0
                ).numpy()
                sd_recon_train_y_vec = torch.std(self.reconstructed_train_y, 0).numpy()
                covm_recon_train_y = np.cov(
                    self.reconstructed_train_y.numpy(), rowvar=False
                )
            else:
                mean_recon_train_y_vec = (
                    sd_recon_train_y_vec
                ) = covm_recon_train_y = None

            self.results_vecs["mean_recon_train_y_vec"] = mean_recon_train_y_vec
            self.results_vecs["sd_recon_train_y_vec"] = sd_recon_train_y_vec
            self.results_vecs["covm_recon_train_y"] = covm_recon_train_y

            mean_recon_train_y_vec_stats = self.compute_array_stats(
                mean_recon_train_y_vec
            )
            sd_recon_train_y_vec_stats = self.compute_array_stats(sd_recon_train_y_vec)

            write_data_recon_train_pixel_mean_y = self.make_array_stats_string(
                mean_recon_train_y_vec_stats, precision
            )
            write_data_recon_train_pixel_mean_y_txt = (
                "Reconstructed training travel times mean: "
            )
            write_header_recon_train_pixel_mean_y = "mean_recon_train_y,median_recon_train_y,[(q25_recon_train_y,q75_recon_train_y),(q025_recon_train_y,q975_recon_train_y)]"

            write_data_recon_train_pixel_sd_y = self.make_array_stats_string(
                sd_recon_train_y_vec_stats, precision
            )
            write_data_recon_train_pixel_sd_y_txt = (
                "Average std on reconstructed training travel times: "
            )
            write_header_recon_train_pixel_sd_y = "sd_recon_train_y,median_sd_recon_train_y,[(q25_sd_recon_train_y,q75_sd_recon_train_y),(q025_sd_recon_train_y,q975_sd_recon_train_y)]"
        else:
            raise ValueError(
                "No reconstructed training data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        if (
            self.reconstructed_val_x is not None
            and self.reconstructed_val_y is not None
        ):
            mean_recon_val_x_vec = torch.mean(self.reconstructed_val_x, 0).numpy()
            self.results_vecs["mean_recon_val_x_vec"] = mean_recon_val_x_vec

            mean_recon_val_x_vec_stats = self.compute_array_stats(mean_recon_val_x_vec)

            write_data_recon_val_pixel_mean_x = self.make_array_stats_string(
                mean_recon_val_x_vec_stats, precision
            )
            write_data_recon_val_pixel_mean_x_txt = (
                "Reconstructed validation models pixels mean: "
            )
            write_header_recon_val_pixel_mean_x = "mean_recon_val_x,median_recon_val_x,[(q25_recon_val_x,q75_recon_val_x),(q025_recon_val_x,q975_recon_val_x)]"

            sd_recon_val_x_vec = torch.std(self.reconstructed_val_x, 0).numpy()
            self.results_vecs["sd_recon_val_x_vec"] = sd_recon_val_x_vec

            sd_recon_val_x_vec_stats = self.compute_array_stats(sd_recon_val_x_vec)

            write_data_recon_val_pixel_sd_x = self.make_array_stats_string(
                sd_recon_val_x_vec_stats, precision
            )
            write_data_recon_val_pixel_sd_x_txt = (
                "Average std on reconstructed validation models pixels: "
            )
            write_header_recon_val_pixel_sd_x = "sd_recon_val_x,median_sd_recon_val_x,[(q25_sd_recon_val_x,q75_sd_recon_val_x),(q025_sd_recon_val_x,q975_sd_recon_val_x)]"

            covm_recon_val_x = np.cov(self.reconstructed_val_x.numpy(), rowvar=False)
            self.results_vecs["covm_recon_val_x"] = covm_recon_val_x

            if build_y_stats:
                mean_recon_val_y_vec = torch.mean(self.reconstructed_val_y, 0).numpy()
                sd_recon_val_y_vec = torch.std(self.reconstructed_val_y, 0).numpy()
                covm_recon_val_y = np.cov(
                    self.reconstructed_val_y.numpy(), rowvar=False
                )
            else:
                mean_recon_val_y_vec = sd_recon_val_y_vec = covm_recon_val_y = None

            self.results_vecs["mean_recon_val_y_vec"] = mean_recon_val_y_vec
            self.results_vecs["sd_recon_val_y_vec"] = sd_recon_val_y_vec
            self.results_vecs["covm_recon_val_y"] = covm_recon_val_y

            mean_recon_val_y_vec_stats = self.compute_array_stats(mean_recon_val_y_vec)
            sd_recon_train_y_vec_stats = self.compute_array_stats(sd_recon_val_y_vec)

            write_data_recon_val_pixel_mean_y = self.make_array_stats_string(
                mean_recon_val_y_vec_stats, precision
            )
            write_data_recon_val_pixel_mean_y_txt = (
                "Reconstructed validation travel times mean: "
            )
            write_header_recon_val_pixel_mean_y = "mean_recon_val_y,median_recon_val_y,[(q25_recon_val_y,q75_recon_val_y),(q025_recon_val_y,q975_recon_val_y)]"

            write_data_recon_val_pixel_sd_y = self.make_array_stats_string(
                sd_recon_train_y_vec_stats, precision
            )
            write_data_recon_val_pixel_sd_y_txt = (
                "Average std on reconstructed validation travel times: "
            )
            write_header_recon_val_pixel_sd_y = "sd_recon_val_y,median_sd_recon_val_y,[(q25_sd_recon_val_y,q75_sd_recon_val_y),(q025_sd_recon_val_y,q975_sd_recon_val_y)]"

        else:
            raise ValueError(
                "No reconstructed validation data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        if make_plots:
            self.plot_sd_boxplots(joint_output=joint_output)
            self.plot_summary_stats(joint_output=joint_output)
            self.plot_x_y_umap_tsne_scatters(
                n_neighbors=n_neighbors, min_dist=min_dist, denseMAP=denseMAP
            )

        return {
            "data": {
                "write_data_pixel_mean_train_x": (
                    write_data_pixel_mean_train_x,
                    write_data_pixel_mean_train_x_txt,
                ),
                "write_data_pixel_mean_val_x": (
                    write_data_pixel_mean_val_x,
                    write_data_pixel_mean_val_x_txt,
                ),
                "write_data_pixel_mean_gen_x": (
                    write_data_pixel_mean_gen_x,
                    write_data_pixel_mean_gen_x_txt,
                ),
                "write_data_recon_train_pixel_mean_x": (
                    write_data_recon_train_pixel_mean_x,
                    write_data_recon_train_pixel_mean_x_txt,
                ),
                "write_data_recon_val_pixel_mean_x": (
                    write_data_recon_val_pixel_mean_x,
                    write_data_recon_val_pixel_mean_x_txt,
                ),
                "write_data_pixel_mean_train_y": (
                    write_data_pixel_mean_train_y,
                    write_data_pixel_mean_train_y_txt,
                ),
                "write_data_pixel_mean_val_y": (
                    write_data_pixel_mean_val_y,
                    write_data_pixel_mean_val_y_txt,
                ),
                "write_data_pixel_mean_gen_y": (
                    write_data_pixel_mean_gen_y,
                    write_data_pixel_mean_gen_y_txt,
                ),
                "write_data_recon_train_pixel_mean_y": (
                    write_data_recon_train_pixel_mean_y,
                    write_data_recon_train_pixel_mean_y_txt,
                ),
                "write_data_recon_val_pixel_mean_y": (
                    write_data_recon_val_pixel_mean_y,
                    write_data_recon_val_pixel_mean_y_txt,
                ),
                "write_data_pixel_sd_train_x": (
                    write_data_pixel_sd_train_x,
                    write_data_pixel_sd_train_x_txt,
                ),
                "write_data_pixel_sd_val_x": (
                    write_data_pixel_sd_val_x,
                    write_data_pixel_sd_val_x_txt,
                ),
                "write_data_pixel_sd_gen_x": (
                    write_data_pixel_sd_gen_x,
                    write_data_pixel_sd_gen_x_txt,
                ),
                "write_data_recon_train_pixel_sd_x": (
                    write_data_recon_train_pixel_sd_x,
                    write_data_recon_train_pixel_sd_x_txt,
                ),
                "write_data_recon_val_pixel_sd_x": (
                    write_data_recon_val_pixel_sd_x,
                    write_data_recon_val_pixel_sd_x_txt,
                ),
                "write_data_pixel_sd_train_y": (
                    write_data_pixel_sd_train_y,
                    write_data_pixel_sd_train_y_txt,
                ),
                "write_data_pixel_sd_val_y": (
                    write_data_pixel_sd_val_y,
                    write_data_pixel_sd_val_y_txt,
                ),
                "write_data_pixel_sd_gen_y": (
                    write_data_pixel_sd_gen_y,
                    write_data_pixel_sd_gen_y_txt,
                ),
                "write_data_recon_train_pixel_sd_y": (
                    write_data_recon_train_pixel_sd_y,
                    write_data_recon_train_pixel_sd_y_txt,
                ),
                "write_data_recon_val_pixel_sd_y": (
                    write_data_recon_val_pixel_sd_y,
                    write_data_recon_val_pixel_sd_y_txt,
                ),
            },
            "headers": {
                "write_header_pixel_mean_train_x": write_header_pixel_mean_train_x,
                "write_header_pixel_mean_val_x": write_header_pixel_mean_val_x,
                "write_header_pixel_mean_gen_x": write_header_pixel_mean_gen_x,
                "write_header_recon_train_pixel_mean_x": write_header_recon_train_pixel_mean_x,
                "write_header_recon_val_pixel_mean_x": write_header_recon_val_pixel_mean_x,
                "write_header_pixel_mean_train_y": write_header_pixel_mean_train_y,
                "write_header_pixel_mean_val_y": write_header_pixel_mean_val_y,
                "write_header_pixel_mean_gen_y": write_header_pixel_mean_gen_y,
                "write_header_recon_train_pixel_mean_y": write_header_recon_train_pixel_mean_y,
                "write_header_recon_val_pixel_mean_y": write_header_recon_val_pixel_mean_y,
                "write_header_pixel_sd_train_x": write_header_pixel_sd_train_x,
                "write_header_pixel_sd_val_x": write_header_pixel_sd_val_x,
                "write_header_pixel_sd_gen_x": write_header_pixel_sd_gen_x,
                "write_header_recon_train_pixel_sd_x": write_header_recon_train_pixel_sd_x,
                "write_header_recon_val_pixel_sd_x": write_header_recon_val_pixel_sd_x,
                "write_header_pixel_sd_train_y": write_header_pixel_sd_train_y,
                "write_header_pixel_sd_val_y": write_header_pixel_sd_val_y,
                "write_header_pixel_sd_gen_y": write_header_pixel_sd_gen_y,
                "write_header_recon_train_pixel_sd_y": write_header_recon_train_pixel_sd_y,
                "write_header_recon_val_pixel_sd_y": write_header_recon_val_pixel_sd_y,
            },
        }

    def make_recons_stats(self, build_y_stats=True, joint_output=True, make_plots=True):
        """
        Make mean, std and iqr statistics for all data.

        :param build_y_stats: if True, build mean, std and iqr statistics for y data
        :param joint_output: whether the model being diagnosed has a joint output or not
        :param make_plots: if True, make plots of the statistics
        :return: dictionary of statistics formatted to be logged by write_results()
        """
        print("Making reconstruction statistics...")

        train_x = self.train_x
        train_y = self.train_y

        val_x = self.val_x
        val_y = self.val_y

        dim_x = self.experiment.dim_x
        dim_y = self.experiment.dim_y

        precision = self.round_digits

        if train_x is not None and train_y is not None:
            train_x = train_x.view(train_x.shape[0], dim_x)
            train_y = train_y.view(train_y.shape[0], dim_y)

            recon_train_x_rmse_vec = rmse_torch(
                self.reconstructed_train_x, train_x.view(-1, dim_x)
            ).numpy()
            # recon_train_x_rmse_vec = torch.sqrt(torch.mean((self.reconstructed_train_x - train_x.view(-1, dim_x)) ** 2, 1)).numpy()
            self.results_vecs["recon_train_x_rmse_vec"] = recon_train_x_rmse_vec

            recon_train_x_rmse_vec_stats = self.compute_array_stats(
                recon_train_x_rmse_vec
            )

            write_data_recon_train_x = self.make_array_stats_string(
                recon_train_x_rmse_vec_stats, precision
            )
            write_data_recon_train_x_txt = (
                "Reconstructed training models RMSE vs training models: "
            )
            write_header_recon_train_x = "mean_recon_train_x_rmse,median_recon_train_x_rmse,(q25_recon_train_x_rmse,q75_recon_train_x_rmse),(q025_recon_train_x_rmse,q975_recon_train_x_rmse)"

            if build_y_stats:
                recon_train_y_rmse_vec = rmse_torch(
                    self.reconstructed_train_y, train_y.view(-1, dim_y)
                ).numpy()
                # recon_train_y_rmse_vec = torch.sqrt(torch.mean((self.reconstructed_train_y - train_y.view(-1, dim_y)) ** 2, 1)).numpy()
            else:
                recon_train_y_rmse_vec = None

            self.results_vecs["recon_train_y_rmse_vec"] = recon_train_y_rmse_vec

            recon_train_y_rmse_vec_stats = self.compute_array_stats(
                recon_train_y_rmse_vec
            )

            write_data_recon_train_y = self.make_array_stats_string(
                recon_train_y_rmse_vec_stats, precision
            )
            write_data_recon_train_y_txt = (
                "Reconstructed training travel times RMSE vs training travel times: "
            )
            write_header_recon_train_y = "mean_recon_train_y_rmse,median_recon_train_y_rmse,(q25_recon_train_y_rmse,q75_recon_train_y_rmse),(q025_recon_train_y_rmse,q975_recon_train_y_rmse)"

        else:
            raise ValueError(
                "No training data to compute mean and std statistics."
                "Load data into the experiment object first."
            )

        if val_x is not None and val_y is not None:
            val_x = val_x.view(val_x.shape[0], dim_x)
            val_y = val_y.view(val_y.shape[0], dim_y)

            recon_val_x_rmse_vec = rmse_torch(
                self.reconstructed_val_x, val_x.view(-1, dim_x)
            ).numpy()
            # recon_val_x_rmse_vec = torch.sqrt(torch.mean((self.reconstructed_val_x - val_x.view(-1, dim_x)) ** 2, 1)).numpy()
            self.results_vecs["recon_val_x_rmse_vec"] = recon_val_x_rmse_vec

            recon_val_x_rmse_vec_stats = self.compute_array_stats(recon_val_x_rmse_vec)

            write_data_recon_val_x = self.make_array_stats_string(
                recon_val_x_rmse_vec_stats, precision
            )
            write_data_recon_val_x_txt = (
                "Reconstructed validation models RMSE vs validation models: "
            )
            write_header_recon_val_x = "mean_recon_val_x_rmse,median_recon_val_x_rmse,(q25_recon_val_x_rmse,q75_recon_val_x_rmse),(q025_recon_val_x_rmse,q975_recon_val_x_rmse)"

            if build_y_stats:
                recon_val_y_rmse_vec = rmse_torch(
                    self.reconstructed_val_y, val_y.view(-1, dim_y)
                ).numpy()
                # recon_val_y_rmse_vec = torch.sqrt(torch.mean((self.reconstructed_val_y - val_y.view(-1, dim_y)) ** 2, 1))
            else:
                recon_val_y_rmse_vec = None

            self.results_vecs["recon_val_y_rmse_vec"] = recon_val_y_rmse_vec

            recon_val_y_rmse_vec_stats = self.compute_array_stats(recon_val_y_rmse_vec)

            write_data_recon_val_y = self.make_array_stats_string(
                recon_val_y_rmse_vec_stats, precision
            )
            write_data_recon_val_y_txt = "Reconstructed validation travel times RMSE vs validation travel times: "
            write_header_recon_val_y = "mean_recon_val_y_rmse,median_recon_val_y_rmse,(q25_recon_val_y_rmse,q75_recon_val_y_rmse),(q025_recon_val_y_rmse,q975_recon_val_y_rmse)"

        else:
            raise ValueError(
                "No validation data to compute mean and std statistics."
                "Load data into the experiment object first."
            )

        if make_plots:
            self.plot_true_vs_recon(data_type="train", joint_output=joint_output)
            self.plot_true_vs_recon(data_type="val", joint_output=joint_output)
            self.plot_recon_rmse_hists(joint_output=joint_output)

        return {
            "data": {
                "write_data_recon_train_x": (
                    write_data_recon_train_x,
                    write_data_recon_train_x_txt,
                ),
                "write_data_recon_val_x": (
                    write_data_recon_val_x,
                    write_data_recon_val_x_txt,
                ),
                "write_data_recon_train_y": (
                    write_data_recon_train_y,
                    write_data_recon_train_y_txt,
                ),
                "write_data_recon_val_y": (
                    write_data_recon_val_y,
                    write_data_recon_val_y_txt,
                ),
            },
            "headers": {
                "write_header_recon_train_x": write_header_recon_train_x,
                "write_header_recon_val_x": write_header_recon_val_x,
                "write_header_recon_train_y": write_header_recon_train_y,
                "write_header_recon_val_y": write_header_recon_val_y,
            },
        }

    def make_variance_loss_stats(
        self, build_y_stats=True, joint_output=True, make_plots=True
    ):
        """
        Compute variance loss statistics (Variance Loss in Variational Autoencoders, Andrea Asperti, 2020, LOD))
        :param build_y_stats: if True, build mean, std and iqr statistics for y data
        :param joint_output: whether the model being diagnosed has a joint output or not
        :param make_plots: if True, make plots of the statistics
        :return: dictionary of statistics formatted to be logged by write_results()
        """

        precision = self.round_digits

        if len(self.results_vecs) == 0:
            print(
                "No results vectors to compute to use for variance loss statistics.\n"
                "Running make_pixel_stats() first."
            )
            self.make_pixel_stats(build_y_stats=build_y_stats)

        var_train_x_vec = self.results_vecs["sd_train_x_vec"] ** 2
        var_recon_train_x_vec = self.results_vecs["sd_recon_train_x_vec"] ** 2
        var_gen_x_vec = self.results_vecs["sd_gen_x_vec"] ** 2

        var_loss_train_x_vec = var_train_x_vec - var_recon_train_x_vec
        var_loss_train_gen_x_vec = var_train_x_vec - var_gen_x_vec
        self.results_vecs["var_loss_train_x_vec"] = var_loss_train_x_vec
        self.results_vecs["var_loss_train_gen_x_vec"] = var_loss_train_gen_x_vec

        var_loss_train_x_vec_stats = self.compute_array_stats(var_loss_train_x_vec)

        write_data_var_loss_train_x = self.make_array_stats_string(
            var_loss_train_x_vec_stats, precision
        )
        write_data_var_loss_train_x_txt = (
            "Variance loss between training models and their reconstruction: "
        )
        write_header_var_loss_train_x = "var_loss_train_x,median_var_loss_train_x,(q25_var_loss_train_x,q75_var_loss_train_x),(q025_var_loss_train_x,q975_var_loss_train_x)"

        var_loss_train_gen_x_vec_stats = self.compute_array_stats(
            var_loss_train_gen_x_vec
        )

        write_data_var_loss_train_gen_x = self.make_array_stats_string(
            var_loss_train_gen_x_vec_stats, precision
        )
        write_data_var_loss_train_gen_x_txt = (
            "Variance loss between training models and generated models: "
        )
        write_header_var_loss_train_gen_x = "var_loss_train_gen_x,median_var_loss_train_gen_x,(q25_var_loss_train_gen_x,q75_var_loss_train_gen_x),(q025_var_loss_train_gen_x,q975_var_loss_train_gen_x)"

        var_val_x_vec = self.results_vecs["sd_val_x_vec"] ** 2
        var_recon_val_x_vec = self.results_vecs["sd_recon_val_x_vec"] ** 2
        var_loss_val_x_vec = var_val_x_vec - var_recon_val_x_vec
        self.results_vecs["var_loss_val_x_vec"] = var_loss_val_x_vec

        var_val_x_vec_stats = self.compute_array_stats(var_loss_val_x_vec)

        write_data_var_loss_val_x = self.make_array_stats_string(
            var_val_x_vec_stats, precision
        )
        write_data_var_loss_val_x_txt = (
            "Variance loss between validation models and their reconstruction: "
        )
        write_header_var_loss_val_x = "var_loss_val_x,median_var_loss_val_x,(q25_var_loss_val_x,q75_var_loss_val_x),(q025_var_loss_val_x,q975_var_loss_val_x)"

        if build_y_stats:
            var_train_y_vec = self.results_vecs["sd_train_y_vec"] ** 2
            var_recon_train_y_vec = self.results_vecs["sd_recon_train_y_vec"] ** 2
            var_gen_y_vec = self.results_vecs["sd_gen_y_vec"] ** 2

            var_loss_train_y_vec = var_train_y_vec - var_recon_train_y_vec
            var_loss_train_gen_y_vec = var_train_y_vec - var_gen_y_vec

            var_val_y_vec = self.results_vecs["sd_val_y_vec"] ** 2
            var_recon_val_y_vec = self.results_vecs["sd_recon_val_y_vec"] ** 2
            var_loss_val_y_vec = var_val_y_vec - var_recon_val_y_vec

        else:
            var_loss_train_y_vec = var_loss_val_y_vec = var_loss_train_gen_y_vec = None

        self.results_vecs["var_loss_train_y_vec"] = var_loss_train_y_vec
        self.results_vecs["var_loss_train_gen_y_vec"] = var_loss_train_gen_y_vec
        self.results_vecs["var_loss_val_y_vec"] = var_loss_val_y_vec

        var_loss_train_y_vec_stats = self.compute_array_stats(var_loss_train_y_vec)
        var_loss_train_gen_y_vec_stats = self.compute_array_stats(
            var_loss_train_gen_y_vec
        )
        var_loss_val_y_vec_stats = self.compute_array_stats(var_loss_val_y_vec)

        write_data_var_loss_train_y = self.make_array_stats_string(
            var_loss_train_y_vec_stats, precision
        )
        write_data_var_loss_train_y_txt = (
            "Variance loss between training travel times and their reconstruction: "
        )
        write_header_var_loss_train_y = "var_loss_train_y,median_var_loss_train_y,(q25_var_loss_train_y,q75_var_loss_train_y),(q025_var_loss_train_y,q975_var_loss_train_y)"

        write_data_var_loss_train_gen_y = self.make_array_stats_string(
            var_loss_train_gen_y_vec_stats, precision
        )
        write_data_var_loss_train_gen_y_txt = (
            "Variance loss between training travel times and generated travel times: "
        )
        write_header_var_loss_train_gen_y = "var_loss_train_gen_y,median_var_loss_train_gen_y,(q25_var_loss_train_gen_y,q75_var_loss_train_gen_y),(q025_var_loss_train_gen_y,q975_var_loss_train_gen_y)"

        write_data_var_loss_val_y = self.make_array_stats_string(
            var_loss_val_y_vec_stats, precision
        )
        write_data_var_loss_val_y_txt = (
            "Variance loss between validation travel timesand their reconstruction: "
        )
        write_header_var_loss_val_y = "var_loss_val_y,median_var_loss_val_y,(q25_var_loss_val_y,q75_var_loss_val_y),(q025_var_loss_val_y,q975_var_loss_val_y)"

        if make_plots:
            self.plot_variance_loss_hists(joint_output=joint_output)
        return {
            "data": {
                "write_data_var_loss_train_x": (
                    write_data_var_loss_train_x,
                    write_data_var_loss_train_x_txt,
                ),
                "write_data_var_loss_train_gen_x": (
                    write_data_var_loss_train_gen_x,
                    write_data_var_loss_train_gen_x_txt,
                ),
                "write_data_var_loss_val_x": (
                    write_data_var_loss_val_x,
                    write_data_var_loss_val_x_txt,
                ),
                "write_data_var_loss_train_y": (
                    write_data_var_loss_train_y,
                    write_data_var_loss_train_y_txt,
                ),
                "write_data_var_loss_train_gen_y": (
                    write_data_var_loss_train_gen_y,
                    write_data_var_loss_train_gen_y_txt,
                ),
                "write_data_var_loss_val_y": (
                    write_data_var_loss_val_y,
                    write_data_var_loss_val_y_txt,
                ),
            },
            "headers": {
                "write_header_var_loss_train_x": write_header_var_loss_train_x,
                "write_header_var_loss_train_gen_x": write_header_var_loss_train_gen_x,
                "write_header_var_loss_val_x": write_header_var_loss_val_x,
                "write_header_var_loss_train_y": write_header_var_loss_train_y,
                "write_header_var_loss_train_gen_y": write_header_var_loss_train_gen_y,
                "write_header_var_loss_val_y": write_header_var_loss_val_y,
            },
        }

    def make_resim_stats(self, make_plots=True):
        """
        Compute resimulated travel times statistics.
        :param make_plots: if True, make plots of the statistics
        :return: dictionary of statistics formatted to be logged by write_results()
        """
        print("Making resimulation statistics...")

        ny = self.experiment.ny
        nx = self.experiment.nx
        dim_y = self.experiment.dim_y

        precision = self.round_digits

        solver_type = self.experiment.config.solver_type
        solver_args = [
            self.experiment.config.rays,
            self.experiment.config.nx,
            self.experiment.config.ny,
            self.experiment.config.spacing,
            self.experiment.config.sources_x,
        ]
        so_file = (
            self.experiment.config.so_file if solver_type == "eikonal-nl" else None
        )

        if self.train_x is None or self.val_x is None:
            raise ValueError(
                "No training or validation data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        if (
            self.reconstructed_train_x is None
            or self.reconstructed_val_x is None
            or self.generated_x is None
        ):
            raise ValueError(
                "No reconstructed training or validation data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        recon_train_sample_size = self.reconstructed_train_x.shape[0]
        recon_val_sample_size = self.reconstructed_val_x.shape[0]
        gen_sample_size = self.generated_x.shape[0]

        recon_train_x = self.reconstructed_train_x.numpy().reshape(
            (recon_train_sample_size, ny, nx), order="C"
        )
        recon_train_y = self.reconstructed_train_y.numpy().reshape(
            (recon_train_sample_size, dim_y)
        )
        recon_val_x = self.reconstructed_val_x.numpy().reshape(
            (recon_val_sample_size, ny, nx), order="C"
        )
        recon_val_y = self.reconstructed_val_y.numpy().reshape(
            (recon_val_sample_size, dim_y)
        )
        gen_x = self.generated_x.numpy().reshape((gen_sample_size, ny, nx), order="C")
        gen_y = self.generated_y.numpy()

        # make resimulations
        recon_train_resim = resimulate(
            recon_train_x, solver_type, solver_args, so_file=so_file
        )
        recon_val_resim = resimulate(
            recon_val_x, solver_type, solver_args, so_file=so_file
        )
        gen_resim = resimulate(gen_x, solver_type, solver_args, so_file=so_file)

        self.resimulated_recon_train_x = recon_train_resim
        self.resimulated_recon_val_x = recon_val_resim
        self.resimulated_gen_x = gen_resim

        # compute rmse between ground truths and resimulations

        # ground truth Y vs F(Recon(X)) - train and val
        rmse_train_y_vs_recon_x_resim_vec = compute_rmse_grdTruth(
            samples=recon_train_resim, ref_model=self.train_y.numpy()
        )
        rmse_val_y_vs_recon_x_resim_vec = compute_rmse_grdTruth(
            samples=recon_val_resim, ref_model=self.val_y.numpy()
        )

        # Gen(Y) vs F(Gen(X))
        rmse_gen_y_vs_gen_x_resim_vec = compute_rmse_grdTruth(
            samples=gen_resim, ref_model=gen_y
        )

        # Recon(Y) vs F(Recon(X))
        rmse_train_recon_y_vs_recon_x_resim_vec = compute_rmse_grdTruth(
            samples=recon_train_resim, ref_model=recon_train_y
        )
        rmse_val_recon_y_vs_recon_x_resim_vec = compute_rmse_grdTruth(
            samples=recon_val_resim, ref_model=recon_val_y
        )

        self.results_vecs[
            "rmse_train_y_vs_recon_x_resim_vec"
        ] = rmse_train_y_vs_recon_x_resim_vec
        self.results_vecs[
            "rmse_val_y_vs_recon_x_resim_vec"
        ] = rmse_val_y_vs_recon_x_resim_vec
        self.results_vecs[
            "rmse_gen_y_vs_gen_x_resim_vec"
        ] = rmse_gen_y_vs_gen_x_resim_vec
        self.results_vecs[
            "rmse_train_recon_y_vs_recon_x_resim_vec"
        ] = rmse_train_recon_y_vs_recon_x_resim_vec
        self.results_vecs[
            "rmse_val_recon_y_vs_recon_x_resim_vec"
        ] = rmse_val_recon_y_vs_recon_x_resim_vec

        rmse_train_y_vs_resim_vec_stats = self.compute_array_stats(
            rmse_train_y_vs_recon_x_resim_vec
        )

        write_data_rmse_train_y_vs_resim = self.make_array_stats_string(
            rmse_train_y_vs_resim_vec_stats, precision
        )
        write_data_rmse_train_y_vs_resim_txt = (
            "Resimulation error (RMSE) (ns) - ground truth Y vs F(Recon(X)) - Train:"
        )
        write_header_rmse_train_y_vs_resim = (
            "mean_rmse_train_y_vs_resim_recon_x,median_rmse_train_y_vs_resim_recon_x,"
            "(q25_rmse_train_y_vs_resim_recon_x,q75_rmse_train_y_vs_resim_recon_x),"
            "(q025_rmse_train_y_vs_resim_recon_x,q975_rmse_train_y_vs_resim_recon_x)"
        )

        rmse_val_y_vs_resim_vec_stats = self.compute_array_stats(
            rmse_val_y_vs_recon_x_resim_vec
        )

        write_data_rmse_val_y_vs_resim = self.make_array_stats_string(
            rmse_val_y_vs_resim_vec_stats, precision
        )
        write_data_rmse_val_y_vs_resim_txt = (
            "Resimulation error (RMSE) (ns) - ground truth Y vs F(Recon(X)) - Val:"
        )
        write_header_rmse_val_y_vs_resim = (
            "mean_rmse_val_y_vs_resim_recon_x,median_rmse_val_y_vs_resim_recon_x,"
            "(q25_rmse_val_y_vs_resim_recon_x,q75_rmse_val_y_vs_resim_recon_x),"
            "(q025_rmse_val_y_vs_resim_recon_x,q975_rmse_val_y_vs_resim_recon_x)"
        )

        rmse_gen_y_vs_resim_y_vec_stats = self.compute_array_stats(
            rmse_gen_y_vs_gen_x_resim_vec
        )

        write_data_rmse_gen_y_vs_resim_y = self.make_array_stats_string(
            rmse_gen_y_vs_resim_y_vec_stats, precision
        )
        write_data_rmse_gen_y_vs_resim_y_txt = (
            "Resimulation error (RMSE) (ns) - Gen(Y) vs F(Gen(X)):"
        )
        write_header_rmse_gen_y_vs_resim_y = (
            "mean_rmse_gen_y_vs_resim_gen_x,median_rmse_gen_y_vs_resim_gen_x,"
            "(q25_rmse_gen_y_vs_resim_gen_x,q75_rmse_gen_y_vs_resim_gen_x),"
            "(q025_rmse_gen_y_vs_resim_gen_x,q975_rmse_gen_y_vs_resim_gen_x)"
        )

        rmse_train_recon_y_vs_resim_vec_stats = self.compute_array_stats(
            rmse_train_recon_y_vs_recon_x_resim_vec
        )

        write_data_rmse_train_recon_y_vs_resim = self.make_array_stats_string(
            rmse_train_recon_y_vs_resim_vec_stats, precision
        )
        write_data_rmse_train_recon_y_vs_resim_txt = (
            "Resimulation error (RMSE) (ns) - Recon(Y) vs F(Recon(X)) - Train:"
        )
        write_header_rmse_train_recon_y_vs_resim = (
            "mean_rmse_train_recon_y_vs_resim_recon_x,median_rmse_train_recon_y_vs_resim_recon_x,"
            "(q25_rmse_train_recon_y_vs_resim_recon_x,q75_rmse_train_recon_y_vs_resim_recon_x),"
            "(q025_rmse_train_recon_y_vs_resim_recon_x,q975_rmse_train_recon_y_vs_resim_recon_x)"
        )

        rmse_val_recon_y_vs_resim_vec_stats = self.compute_array_stats(
            rmse_val_recon_y_vs_recon_x_resim_vec
        )
        write_data_rmse_val_recon_y_vs_resim = self.make_array_stats_string(
            rmse_val_recon_y_vs_resim_vec_stats, precision
        )
        write_data_rmse_val_recon_y_vs_resim_txt = (
            "Resimulation error (RMSE) (ns) - Recon(Y) vs F(Recon(X)) - Val:"
        )
        write_header_rmse_val_recon_y_vs_resim = (
            "mean_rmse_val_recon_y_vs_resim_recon_x,median_rmse_val_recon_y_vs_resim_recon_x,"
            "(q25_rmse_val_recon_y_vs_resim_recon_x,q75_rmse_val_recon_y_vs_resim_recon_x),"
            "(q025_rmse_val_recon_y_vs_resim_recon_x,q975_rmse_val_recon_y_vs_resim_recon_x)"
        )

        if make_plots:
            self.plot_resim_rmse_hists()
            self.plot_resim_rmse_boxplots()
            self.plot_resimulations()

        return {
            "data": {
                "write_data_rmse_train_y_vs_resim": (
                    write_data_rmse_train_y_vs_resim,
                    write_data_rmse_train_y_vs_resim_txt,
                ),
                "write_data_rmse_val_y_vs_resim": (
                    write_data_rmse_val_y_vs_resim,
                    write_data_rmse_val_y_vs_resim_txt,
                ),
                "write_data_rmse_gen_y_vs_resim_y": (
                    write_data_rmse_gen_y_vs_resim_y,
                    write_data_rmse_gen_y_vs_resim_y_txt,
                ),
                "write_data_rmse_train_recon_y_vs_resim": (
                    write_data_rmse_train_recon_y_vs_resim,
                    write_data_rmse_train_recon_y_vs_resim_txt,
                ),
                "write_data_rmse_val_recon_y_vs_resim": (
                    write_data_rmse_val_recon_y_vs_resim,
                    write_data_rmse_val_recon_y_vs_resim_txt,
                ),
            },
            "headers": {
                "write_header_rmse_train_y_vs_resim": write_header_rmse_train_y_vs_resim,
                "write_header_rmse_val_y_vs_resim": write_header_rmse_val_y_vs_resim,
                "write_header_rmse_gen_y_vs_resim_y": write_header_rmse_gen_y_vs_resim_y,
                "write_header_rmse_train_recon_y_vs_resim": write_header_rmse_train_recon_y_vs_resim,
                "write_header_rmse_val_recon_y_vs_resim": write_header_rmse_val_recon_y_vs_resim,
            },
        }

    def make_ssim_stats(self, ssim_kwargs):
        """
        Compute SSIM statistics.
        """
        print("Making SSIM statistics...")

        if self.train_x is None or self.val_x is None:
            raise ValueError(
                "No training or validation data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        if self.reconstructed_train_x is None or self.reconstructed_val_x is None:
            raise ValueError(
                "No reconstructed training or validation data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        dims = [self.experiment.ny, self.experiment.nx]
        train_sample_size = self.train_x.shape[0]
        val_sample_size = self.val_x.shape[0]
        dim_x = self.experiment.dim_x

        train_x = self.train_x.view(train_sample_size, dim_x).numpy()
        val_x = self.val_x.view(val_sample_size, dim_x).numpy()

        recon_train_x = self.reconstructed_train_x.view(
            train_sample_size, dim_x
        ).numpy()
        recon_val_x = self.reconstructed_val_x.view(val_sample_size, dim_x).numpy()

        # make ssim range based on training data
        vmax = max(
            np.max(train_x), np.max(recon_train_x), np.max(val_x), np.max(recon_val_x)
        )
        vmin = min(
            np.min(train_x), np.min(recon_train_x), np.min(val_x), np.min(recon_val_x)
        )
        ssim_kwargs["data_range"] = vmax - vmin

        ssim_train_vs_recon_x_vec = self.compute_ssim(
            train_x, recon_train_x, dims, ssim_kwargs
        )
        ssim_val_vs_recon_x_vec = self.compute_ssim(
            val_x, recon_val_x, dims, ssim_kwargs
        )

        self.results_vecs["ssim_train_vs_recon_x"] = ssim_train_vs_recon_x_vec
        self.results_vecs["ssim_val_vs_recon_x"] = ssim_val_vs_recon_x_vec

        ssim_train_vs_recon_x_vec_stats = self.compute_array_stats(
            ssim_train_vs_recon_x_vec
        )

        write_data_ssim_train_vs_recon_x = self.make_array_stats_string(
            ssim_train_vs_recon_x_vec_stats, self.round_digits
        )
        write_data_ssim_train_vs_recon_x_txt = (
            "SSIM - training models vs reconstructed training models: "
        )
        write_header_ssim_train_vs_recon_x = (
            "mean_ssim_train_vs_recon_x,median_ssim_train_vs_recon_x,"
            "(q25_ssim_train_vs_recon_x,q75_ssim_train_vs_recon_x),"
            "(q025_ssim_train_vs_recon_x,q975_ssim_train_vs_recon_x)"
        )

        ssim_val_vs_recon_x_vec_stats = self.compute_array_stats(
            ssim_val_vs_recon_x_vec
        )

        write_data_ssim_val_vs_recon_x = self.make_array_stats_string(
            ssim_val_vs_recon_x_vec_stats, self.round_digits
        )
        write_data_ssim_val_vs_recon_x_txt = (
            "SSIM - validation models vs reconstructed validation models: "
        )
        write_header_ssim_val_vs_recon_x = (
            "mean_ssim_val_vs_recon_x,median_ssim_val_vs_recon_x,"
            "(q25_ssim_val_vs_recon_x,q75_ssim_val_vs_recon_x),"
            "(q025_ssim_val_vs_recon_x,q975_ssim_val_vs_recon_x)"
        )

        return {
            "data": {
                "write_data_ssim_train_vs_recon_x": (
                    write_data_ssim_train_vs_recon_x,
                    write_data_ssim_train_vs_recon_x_txt,
                ),
                "write_data_ssim_val_vs_recon_x": (
                    write_data_ssim_val_vs_recon_x,
                    write_data_ssim_val_vs_recon_x_txt,
                ),
            },
            "headers": {
                "write_header_ssim_train_vs_recon_x": write_header_ssim_train_vs_recon_x,
                "write_header_ssim_val_vs_recon_x": write_header_ssim_val_vs_recon_x,
            },
        }

    def make_cosine_sim_stats(self, build_y_stats=True):
        """
        Compute cosine similarity statistics.
        """
        print("Making cosine similarity statistics...")

        if self.train_x is None or self.val_x is None:
            raise ValueError(
                "No training or validation data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        if self.reconstructed_train_x is None or self.reconstructed_val_x is None:
            raise ValueError(
                "No reconstructed training or validation data to compute mean and std statistics."
                "Build diagnostics data first."
            )

        train_sample_size = self.train_x.shape[0]
        val_sample_size = self.val_x.shape[0]
        dim_x = self.experiment.dim_x
        dim_y = self.experiment.dim_y

        train_x = self.train_x.view(train_sample_size, dim_x).numpy()
        val_x = self.val_x.view(val_sample_size, dim_x).numpy()
        train_y = self.train_y.view(train_sample_size, dim_y).numpy()
        val_y = self.val_y.view(val_sample_size, dim_y).numpy()

        recon_train_x = self.reconstructed_train_x.view(
            train_sample_size, dim_x
        ).numpy()
        recon_val_x = self.reconstructed_val_x.view(val_sample_size, dim_x).numpy()
        gen_x = self.generated_x.view(val_sample_size, dim_x).numpy()
        recon_train_y = self.reconstructed_train_y.view(
            train_sample_size, dim_y
        ).numpy()
        recon_val_y = self.reconstructed_val_y.view(val_sample_size, dim_y).numpy()
        gen_y = self.generated_y.view(val_sample_size, dim_y).numpy()

        cossim_train_vs_recon_x_vec = self.compute_cosine_sim(train_x, recon_train_x)
        cossim_val_vs_recon_x_vec = self.compute_cosine_sim(val_x, recon_val_x)
        cossim_train_vs_gen_x_vec = self.compute_cosine_sim(train_x, gen_x)

        self.results_vecs["cossim_train_vs_recon_x"] = cossim_train_vs_recon_x_vec
        self.results_vecs["cossim_val_vs_recon_x"] = cossim_val_vs_recon_x_vec
        self.results_vecs["cossim_train_vs_gen_x"] = cossim_train_vs_gen_x_vec

        cossim_train_vs_recon_x_vec_stats = self.compute_array_stats(
            cossim_train_vs_recon_x_vec
        )

        write_data_cossim_train_vs_recon_x = self.make_array_stats_string(
            cossim_train_vs_recon_x_vec_stats, self.round_digits
        )
        write_data_cossim_train_vs_recon_x_txt = (
            "Cosine similarity - training models vs reconstructed training models: "
        )
        write_header_cossim_train_vs_recon_x = (
            "mean_cossim_train_vs_recon_x,median_cossim_train_vs_recon_x,"
            "(q25_cossim_train_vs_recon_x,q75_cossim_train_vs_recon_x),"
            "(q025_cossim_train_vs_recon_x,q975_cossim_train_vs_recon_x)"
        )

        cossim_val_vs_recon_x_vec_stats = self.compute_array_stats(
            cossim_val_vs_recon_x_vec
        )

        write_data_cossim_val_vs_recon_x = self.make_array_stats_string(
            cossim_val_vs_recon_x_vec_stats, self.round_digits
        )
        write_data_cossim_val_vs_recon_x_txt = (
            "Cosine similarity - validation models vs reconstructed validation models: "
        )
        write_header_cossim_val_vs_recon_x = (
            "mean_cossim_val_vs_recon_x,median_cossim_val_vs_recon_x,"
            "(q25_cossim_val_vs_recon_x,q75_cossim_val_vs_recon_x),"
            "(q025_cossim_val_vs_recon_x,q975_cossim_val_vs_recon_x)"
        )

        cossim_train_vs_gen_x_vec_stats = self.compute_array_stats(
            cossim_train_vs_gen_x_vec
        )

        write_data_cossim_train_vs_gen_x = self.make_array_stats_string(
            cossim_train_vs_gen_x_vec_stats, self.round_digits
        )
        write_data_cossim_train_vs_gen_x_txt = (
            "Cosine similarity - training models vs generated models: "
        )
        write_header_cossim_train_vs_gen_x = (
            "mean_cossim_train_vs_gen_x,median_cossim_train_vs_gen_x,"
            "(q25_cossim_train_vs_gen_x,q75_cossim_train_vs_gen_x),"
            "(q025_cossim_train_vs_gen_x,q975_cossim_train_vs_gen_x)"
        )

        if build_y_stats:
            cossim_train_vs_recon_y_vec = self.compute_cosine_sim(
                train_y, recon_train_y
            )
            cossim_val_vs_recon_y_vec = self.compute_cosine_sim(val_y, recon_val_y)
            cossim_train_vs_gen_y_vec = self.compute_cosine_sim(train_y, gen_y)
        else:
            cossim_train_vs_recon_y_vec = (
                cossim_val_vs_recon_y_vec
            ) = cossim_train_vs_gen_y_vec = None

        self.results_vecs["cossim_train_vs_recon_y"] = cossim_train_vs_recon_y_vec
        self.results_vecs["cossim_val_vs_recon_y"] = cossim_val_vs_recon_y_vec
        self.results_vecs["cossim_train_vs_gen_y"] = cossim_train_vs_gen_y_vec

        cossim_train_vs_recon_y_vec_stats = self.compute_array_stats(
            cossim_train_vs_recon_y_vec
        )

        write_data_cossim_train_vs_recon_y = self.make_array_stats_string(
            cossim_train_vs_recon_y_vec_stats, self.round_digits
        )
        write_data_cossim_train_vs_recon_y_txt = "Cosine similarity - training travel times vs reconstructed training travel times: "
        write_header_cossim_train_vs_recon_y = (
            "mean_cossim_train_vs_recon_y,median_cossim_train_vs_recon_y,"
            "(q25_cossim_train_vs_recon_y,q75_cossim_train_vs_recon_y),"
            "(q025_cossim_train_vs_recon_y,q975_cossim_train_vs_recon_y)"
        )

        cossim_val_vs_recon_y_vec_stats = self.compute_array_stats(
            cossim_val_vs_recon_y_vec
        )

        write_data_cossim_val_vs_recon_y = self.make_array_stats_string(
            cossim_val_vs_recon_y_vec_stats, self.round_digits
        )
        write_data_cossim_val_vs_recon_y_txt = "Cosine similarity - validation travel times vs reconstructed validation travel times: "
        write_header_cossim_val_vs_recon_y = (
            "mean_cossim_val_vs_recon_y,median_cossim_val_vs_recon_y,"
            "(q25_cossim_val_vs_recon_y,q75_cossim_val_vs_recon_y),"
            "(q025_cossim_val_vs_recon_y,q975_cossim_val_vs_recon_y)"
        )

        cossim_train_vs_gen_y_vec_stats = self.compute_array_stats(
            cossim_train_vs_gen_y_vec
        )

        write_data_cossim_train_vs_gen_y = self.make_array_stats_string(
            cossim_train_vs_gen_y_vec_stats, self.round_digits
        )
        write_data_cossim_train_vs_gen_y_txt = (
            "Cosine similarity - training travel times vs generated travel times: "
        )
        write_header_cossim_train_vs_gen_y = (
            "mean_cossim_train_vs_gen_y,median_cossim_train_vs_gen_y,"
            "(q25_cossim_train_vs_gen_y,q75_cossim_train_vs_gen_y),"
            "(q025_cossim_train_vs_gen_y,q975_cossim_train_vs_gen_y)"
        )

        return {
            "data": {
                "write_data_cossim_train_vs_recon_x": (
                    write_data_cossim_train_vs_recon_x,
                    write_data_cossim_train_vs_recon_x_txt,
                ),
                "write_data_cossim_val_vs_recon_x": (
                    write_data_cossim_val_vs_recon_x,
                    write_data_cossim_val_vs_recon_x_txt,
                ),
                "write_data_cossim_train_vs_gen_x": (
                    write_data_cossim_train_vs_gen_x,
                    write_data_cossim_train_vs_gen_x_txt,
                ),
                "write_data_cossim_train_vs_recon_y": (
                    write_data_cossim_train_vs_recon_y,
                    write_data_cossim_train_vs_recon_y_txt,
                ),
                "write_data_cossim_val_vs_recon_y": (
                    write_data_cossim_val_vs_recon_y,
                    write_data_cossim_val_vs_recon_y_txt,
                ),
                "write_data_cossim_train_vs_gen_y": (
                    write_data_cossim_train_vs_gen_y,
                    write_data_cossim_train_vs_gen_y_txt,
                ),
            },
            "headers": {
                "write_header_cossim_train_vs_recon_x": write_header_cossim_train_vs_recon_x,
                "write_header_cossim_val_vs_recon_x": write_header_cossim_val_vs_recon_x,
                "write_header_cossim_train_vs_gen_x": write_header_cossim_train_vs_gen_x,
                "write_header_cossim_train_vs_recon_y": write_header_cossim_train_vs_recon_y,
                "write_header_cossim_val_vs_recon_y": write_header_cossim_val_vs_recon_y,
                "write_header_cossim_train_vs_gen_y": write_header_cossim_train_vs_gen_y,
            },
        }

    def inspect_latent_distribution(
        self,
        make_plots=True,
        n_neighbors=100,
        min_dist=0.3,
        mmd_params=None,
        two_sample_test_params=None,
        test_sample_size=300,
        denseMAP=True,
    ):
        """
        Inspect latent space distribution.
        :param make_plots: if True, make plots of the latent distribution inspections
        :param n_neighbors: number of neighbors to consider for UMAP embedding
        :param min_dist: minimum distance between points in UMAP embedding
        :param two_sample_test_params: parameters for the two sample MMD test
        :param test_sample_size: sample size for the two sample MMD test
        :return: dictionary of statistics formatted to be logged by write_results()
        """
        import copy

        print("Inspecting latent space distribution...")

        precision = self.round_digits

        mmd_params_z = copy.deepcopy(mmd_params)
        two_sample_test_params_z = copy.deepcopy(two_sample_test_params)

        z_dist = self.experiment.latent_dist
        z_dist_params = self.experiment.latent_dist_params_list
        latent_dim = self.experiment.latent_dim

        sample_size = test_sample_size

        self.latent_prior_mean = np.mean(self.latent_vector.numpy(), axis=0)
        self.latent_train_codes_mean = np.mean(self.latent_train_codes.numpy(), axis=0)
        self.latent_val_codes_mean = np.mean(self.latent_val_codes.numpy(), axis=0)

        mean_diff_prior_train_norm = np.linalg.norm(
            self.latent_prior_mean - self.latent_train_codes_mean
        ) / (
            latent_dim**0.5
        )  # norm of difference
        mean_diff_prior_val_norm = np.linalg.norm(
            self.latent_prior_mean - self.latent_val_codes_mean
        ) / (
            latent_dim**0.5
        )  # norm of difference

        self.latent_prior_covm = np.cov(self.latent_vector.numpy(), rowvar=False)
        self.latent_train_codes_covm = np.cov(
            self.latent_train_codes.numpy(), rowvar=False
        )
        self.latent_val_codes_covm = np.cov(self.latent_val_codes.numpy(), rowvar=False)

        self.covm_diff_prior_train = np.abs(
            self.latent_prior_covm - self.latent_train_codes_covm
        )
        self.covm_diff_prior_val = np.abs(
            self.latent_prior_covm - self.latent_val_codes_covm
        )

        covm_diff_prior_train_norm = (
            np.linalg.norm(self.covm_diff_prior_train, ord="fro") / latent_dim
        )  # frobenius norm of difference matrix
        covm_diff_prior_val_norm = (
            np.linalg.norm(self.covm_diff_prior_val, ord="fro") / latent_dim
        )  # frobenius norm of difference matrix

        # multivariate normality test
        if (
            self.experiment.latent_dist_name == "standardnormal"
            or self.experiment.latent_dist_name == "normal"
        ):
            # test multivariate normality (Henze-Zirkler test)
            latent_train_mvn_test = pg.multivariate_normality(
                self.latent_train_codes, alpha=0.05
            )
            latent_val_mvn_test = pg.multivariate_normality(
                self.latent_val_codes, alpha=0.05
            )
        else:
            latent_train_mvn_test = latent_val_mvn_test = None

        # two sample MMD test
        if two_sample_test_params is not None:
            latent_train_mmd_two_sample_tests_vec = []
            latent_val_mmd_two_sample_tests_vec = []
            latent_train_mmd_two_sample_tests_pvalues = []
            latent_val_mmd_two_sample_tests_pvalues = []

            latent_train_mmd_values_vec = []
            latent_val_mmd_values_vec = []

            latent_mmd_ref = []

            all_data = np.vstack(
                (self.latent_train_codes, self.latent_val_codes, self.latent_vector)
            )
            gamma_z = self.estimate_mmd_kernel_gamma(all_data)
            mmd_params_z["kernel_params"]["gamma"] = gamma_z
            two_sample_test_params_z["kernel_params"]["gamma"] = gamma_z
            self.mmd_est_params["z"] = gamma_z
            print(f"Estimated MMD kernel gamma for latent space: {gamma_z}")
            del all_data

            num_repeats = self.latent_val_codes.shape[0] // sample_size
            for i in range(num_repeats):
                idx_start_i = i * sample_size
                idx_end_i = (i + 1) * sample_size

                latent_vector = self.latent_vector[idx_start_i:idx_end_i, :]

                print(f"Repeat {i + 1}/{num_repeats}...")
                for j in range(num_repeats):
                    print(f"  Sub-repeat {j + 1}/{num_repeats}...")

                    idx_start_j = j * sample_size
                    idx_end_j = (j + 1) * sample_size

                    latent_train_codes = self.latent_train_codes[
                        idx_start_j:idx_end_j, :
                    ]
                    latent_val_codes = self.latent_val_codes[idx_start_j:idx_end_j, :]

                    train_test = two_sample_mmd_test(
                        latent_vector, latent_train_codes, **two_sample_test_params_z
                    )
                    val_test = two_sample_mmd_test(
                        latent_vector, latent_val_codes, **two_sample_test_params_z
                    )

                    latent_train_mmd_two_sample_tests_vec.append(train_test[2])
                    latent_train_mmd_two_sample_tests_pvalues.append(train_test[1])
                    latent_val_mmd_two_sample_tests_vec.append(val_test[2])
                    latent_val_mmd_two_sample_tests_pvalues.append(val_test[1])

                    latent_train_mmd_values_vec.append(
                        mmd(latent_vector, latent_train_codes, **mmd_params_z)[0]
                    )
                    latent_val_mmd_values_vec.append(
                        mmd(latent_vector, latent_val_codes, **mmd_params_z)[0]
                    )

                    if j != i:
                        latent_vector_2 = self.latent_vector[idx_start_j:idx_end_j, :]

                        latent_mmd_ref.append(
                            mmd(latent_vector, latent_vector_2, **mmd_params_z)[0]
                        )

            latent_train_mmd_two_sample_tests_vec = np.array(
                latent_train_mmd_two_sample_tests_vec
            )
            latent_val_mmd_two_sample_tests_vec = np.array(
                latent_val_mmd_two_sample_tests_vec
            )
            latent_train_mmd_two_sample_tests_pvalues = np.array(
                latent_train_mmd_two_sample_tests_pvalues
            )
            latent_val_mmd_two_sample_tests_pvalues = np.array(
                latent_val_mmd_two_sample_tests_pvalues
            )
            latent_train_mmd_values_vec = np.array(latent_train_mmd_values_vec)
            latent_val_mmd_values_vec = np.array(latent_val_mmd_values_vec)
            latent_mmd_ref = np.array(latent_mmd_ref)
        else:
            latent_train_mmd_two_sample_tests_vec = (
                latent_val_mmd_two_sample_tests_vec
            ) = (
                latent_train_mmd_values_vec
            ) = (
                latent_val_mmd_values_vec
            ) = (
                latent_mmd_ref
            ) = (
                latent_train_mmd_two_sample_tests_pvalues
            ) = latent_val_mmd_two_sample_tests_pvalues = None

        self.results_vecs[
            "latent_train_mmd_two_sample_tests"
        ] = latent_train_mmd_two_sample_tests_vec
        self.results_vecs[
            "latent_train_mmd_two_sample_tests_pvalues"
        ] = latent_train_mmd_two_sample_tests_pvalues
        self.results_vecs[
            "latent_val_mmd_two_sample_tests"
        ] = latent_val_mmd_two_sample_tests_vec
        self.results_vecs[
            "latent_val_mmd_two_sample_tests_pvalues"
        ] = latent_val_mmd_two_sample_tests_pvalues
        self.results_vecs["latent_train_mmd_values"] = latent_train_mmd_values_vec
        self.results_vecs["latent_val_mmd_values"] = latent_val_mmd_values_vec

        write_data_train_mvn_test = (
            str(latent_train_mvn_test) if latent_train_mvn_test is not None else "N/A"
        )  # ['hz', 'pval', 'normal'?]
        write_data_train_mvn_test_txt = (
            "'Multivariate normal test - training latent codes : "
        )
        write_header_train_mvn_test = "latent_train_mvn_test"

        write_data_val_mvn_test = (
            str(latent_val_mvn_test) if latent_val_mvn_test is not None else "N/A"
        )  # ['hz', 'pval', 'normal'?]
        write_data_val_mvn_test_txt = (
            "'Multivariate normal test - validation latent codes : "
        )
        write_header_val_mvn_test = "latent_val_mvn_test"

        if two_sample_test_params is not None:  # if testing was done
            from scipy.stats import chi2

            total_test_repeats = len(latent_val_mmd_two_sample_tests_pvalues)
            # compute proportion of H0 rejections
            prop_latent_train_mmd_two_sample_test = np.mean(
                latent_train_mmd_two_sample_tests_vec
            )
            prop_latent_val_mmd_two_sample_test = np.mean(
                latent_val_mmd_two_sample_tests_vec
            )

            # combine p_values
            combined_stat_train = -2 * np.sum(
                np.log(latent_train_mmd_two_sample_tests_pvalues)
            )
            combined_p_value_train = 1 - chi2.cdf(
                combined_stat_train, 2 * total_test_repeats
            )

            combined_stat_val = -2 * np.sum(
                np.log(latent_val_mmd_two_sample_tests_pvalues)
            )
            combined_p_value_val = 1 - chi2.cdf(
                combined_stat_val, 2 * total_test_repeats
            )
        else:
            prop_latent_train_mmd_two_sample_test = (
                prop_latent_val_mmd_two_sample_test
            ) = combined_p_value_train = combined_p_value_val = None

        write_prop_train_mmd_two_sample_test = (
            f"{prop_latent_train_mmd_two_sample_test:.{precision}f}"
            if prop_latent_train_mmd_two_sample_test is not None
            else None
        )
        write_prop_train_mmd_two_sample_test_txt = f"Proportion of H0 rejections (samples from same dist.) in MMD {total_test_repeats} test repetitions  - training latent codes: "
        write_header_prop_train_mmd_two_sample_test = (
            "prop_reject_latent_train_two_sample_test"
        )

        write_prop_val_mmd_two_sample_test = (
            f"{prop_latent_val_mmd_two_sample_test:.{precision}f}"
            if prop_latent_val_mmd_two_sample_test is not None
            else None
        )
        write_prop_val_mmd_two_sample_test_txt = f"Proportion of H0 rejections (samples from same dist.) MMD {total_test_repeats} test repetitions  - validation latent codes: "
        write_header_prop_val_mmd_two_sample_test = (
            "prop_reject_latent_val_two_sample_test"
        )

        write_combined_pvalue_train_mmd_two_sample_test = (
            f"{combined_p_value_train:.{precision}f}"
            if combined_p_value_train is not None
            else None
        )
        write_combined_pvalue_train_mmd_two_sample_test_txt = f"Combined p-value from two sample MMD {total_test_repeats} test repetitions - training latent codes: "
        write_header_combined_pvalue_train_mmd_two_sample_test = (
            "combined_pvalue_latent_train_two_sample_test"
        )

        write_combined_pvalue_val_mmd_two_sample_test = (
            f"{combined_p_value_val:.{precision}f}"
            if combined_p_value_val is not None
            else None
        )
        write_combined_pvalue_val_mmd_two_sample_test_txt = f"Combined p-value from two sample MMD {total_test_repeats} test repetitions - validation latent codes: "
        write_header_combined_pvalue_val_mmd_two_sample_test = (
            "combined_pvalue_latent_val_two_sample_test"
        )

        latent_train_mmd_values_stats = self.compute_array_stats(
            latent_train_mmd_values_vec
        )
        write_data_train_mmd_values = self.make_array_stats_string(
            latent_train_mmd_values_stats, precision
        )
        write_data_train_mmd_values_txt = f"MMD values - training latent codes from {total_test_repeats}*{sample_size}: "
        write_header_train_mmd_values = (
            "mean_latent_train_mmd_values, std_latent_train_mmd_values, median_latent_train_mmd_values,"
            "[(q25_latent_train_mmd_values,q75_latent_train_mmd_values),"
            "(q025_latent_train_mmd_values,q975_latent_train_mmd_values)]"
        )

        latent_val_mmd_values_stats = self.compute_array_stats(
            latent_val_mmd_values_vec
        )
        write_data_val_mmd_values = self.make_array_stats_string(
            latent_val_mmd_values_stats, precision
        )
        write_data_val_mmd_values_txt = f"MMD values - validation latent codes from {total_test_repeats}*{sample_size}:: "
        write_header_val_mmd_values = (
            "mean_latent_val_mmd_values, std_latent_val_mmd_values, median_latent_val_mmd_values,"
            "[(q25_latent_val_mmd_values,q75_latent_val_mmd_values),"
            "(q025_latent_val_mmd_values,q975_latent_val_mmd_values)]"
        )

        latent_mmd_ref_values_stats = self.compute_array_stats(latent_mmd_ref)
        write_data_mmd_ref_values = self.make_array_stats_string(
            latent_mmd_ref_values_stats, precision
        )
        write_data_mmd_ref_values_txt = (
            f"Ref - MMD values latent space from {total_test_repeats}*{sample_size}: "
        )
        write_header_mmd_ref_values = (
            "mean_mmd_ref_values, std_mmd_ref_values, ,median_mmd_ref_values,"
            "[(q25_mmd_ref_values,q75_mmd_ref_values),"
            "(q025_mmd_ref_values,q975_mmd_ref_values)]"
        )

        write_mean_diff_prior_train_norm = f"{mean_diff_prior_train_norm:.{precision}f}"
        write_mean_diff_prior_train_norm_txt = (
            "Norm of difference between prior and training latent codes means vectors:"
        )
        write_header_mean_diff_prior_train_norm = "mean_diff_latent_prior_train_norm"

        write_mean_diff_prior_val_norm = f"{mean_diff_prior_val_norm:.{precision}f}"
        write_mean_diff_prior_val_norm_txt = "Norm of difference between prior and validation latent codes means vectors:"
        write_header_mean_diff_prior_val_norm = "mean_diff_latent_prior_val_norm"

        write_covm_diff_prior_train_norm = f"{covm_diff_prior_train_norm:.{precision}f}"
        write_covm_diff_prior_train_norm_txt = "Frobinus norm of difference between prior and training latent codes covariance matrices:"
        write_header_covm_diff_prior_train_norm = "covm_diff_latent_prior_train_norm"

        write_covm_diff_prior_val_norm = f"{covm_diff_prior_val_norm:.{precision}f}"
        write_covm_diff_prior_val_norm_txt = "Frobinus norm of difference between prior and validation latent codes covariance matrices:"
        write_header_covm_diff_prior_val_norm = "covm_diff_latent_prior_val_norm"

        if make_plots:
            self.plot_latent_cov_matrices()
            self.plot_latent_histograms()
            self.plot_latent_umap_tsne_scatters(
                n_neighbors=n_neighbors, min_dist=min_dist, denseMAP=denseMAP
            )

        return {
            "data": {
                "write_data_train_mvn_test": (
                    write_data_train_mvn_test,
                    write_data_train_mvn_test_txt,
                ),
                "write_data_val_mvn_test": (
                    write_data_val_mvn_test,
                    write_data_val_mvn_test_txt,
                ),
                "write_prop_train_mmd_two_sample_test": (
                    write_prop_train_mmd_two_sample_test,
                    write_prop_train_mmd_two_sample_test_txt,
                ),
                "write_combined_pvalue_train_mmd_two_sample_test": (
                    write_combined_pvalue_train_mmd_two_sample_test,
                    write_combined_pvalue_train_mmd_two_sample_test_txt,
                ),
                "write_data_train_mmd_values": (
                    write_data_train_mmd_values,
                    write_data_train_mmd_values_txt,
                ),
                "write_prop_val_mmd_two_sample_test": (
                    write_prop_val_mmd_two_sample_test,
                    write_prop_val_mmd_two_sample_test_txt,
                ),
                "write_combined_pvalue_val_mmd_two_sample_test": (
                    write_combined_pvalue_val_mmd_two_sample_test,
                    write_combined_pvalue_val_mmd_two_sample_test_txt,
                ),
                "write_data_val_mmd_values": (
                    write_data_val_mmd_values,
                    write_data_val_mmd_values_txt,
                ),
                "write_data_latent_mmd_ref_values": (
                    write_data_mmd_ref_values,
                    write_data_mmd_ref_values_txt,
                ),
                "write_mean_diff_prior_train_norm": (
                    write_mean_diff_prior_train_norm,
                    write_mean_diff_prior_train_norm_txt,
                ),
                "write_mean_diff_prior_val_norm": (
                    write_mean_diff_prior_val_norm,
                    write_mean_diff_prior_val_norm_txt,
                ),
                "write_covm_diff_prior_train_norm": (
                    write_covm_diff_prior_train_norm,
                    write_covm_diff_prior_train_norm_txt,
                ),
                "write_covm_diff_prior_val_norm": (
                    write_covm_diff_prior_val_norm,
                    write_covm_diff_prior_val_norm_txt,
                ),
            },
            "headers": {
                "write_header_train_mvn_test": write_header_train_mvn_test,
                "write_header_val_mvn_test": write_header_val_mvn_test,
                "write_header_prop_train_mmd_two_sample_test": write_header_prop_train_mmd_two_sample_test,
                "write_header_combined_pvalue_train_mmd_two_sample_test": write_header_combined_pvalue_train_mmd_two_sample_test,
                "write_header_train_mmd_values": write_header_train_mmd_values,
                "write_header_prop_val_mmd_two_sample_test": write_header_prop_val_mmd_two_sample_test,
                "write_header_combined_pvalue_val_mmd_two_sample_test": write_header_combined_pvalue_val_mmd_two_sample_test,
                "write_header_val_mmd_values": write_header_val_mmd_values,
                "write_header_mmd_ref_values": write_header_mmd_ref_values,
                "write_header_mean_diff_prior_train_norm": write_header_mean_diff_prior_train_norm,
                "write_header_mean_diff_prior_val_norm": write_header_mean_diff_prior_val_norm,
                "write_header_covm_diff_prior_train_norm": write_header_covm_diff_prior_train_norm,
                "write_header_covm_diff_prior_val_norm": write_header_covm_diff_prior_val_norm,
            },
        }

    # plotting functions
    def plot_true_vs_recon(
        self, data_type, joint_output=True, n=5, dpi=600, show=False
    ):
        """
        Plot true vs reconstructed models.
        :param data_type: whether to plot training or validation data reconstructions. Can be either "train" or "val".
        :param joint_output: whether the model being diagnosed is a joint distribution model. If true, both x and y data will be plotted.
        :param n: number of data points to plot
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        """
        print("Plotting true vs reconstructions...")

        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config(mpl)

        if (
            data_type.lower() in ["train", "training"]
            and self.experiment.training_x is None
        ):
            raise ValueError(
                "No training data to plot. Load training data into the experiment object first."
            )

        if (
            data_type.lower() in ["val", "validation"]
            and self.experiment.validation_x is None
        ):
            raise ValueError(
                "No validation data to plot. Load validation data into the experiment object first."
            )

        if joint_output:
            ncol = 3
        else:
            ncol = 2

        nx = self.experiment.nx
        ny = self.experiment.ny

        if data_type.lower() in ["train", "training"]:
            data_x = self.train_x
            data_y = self.train_y
            recon_x = self.reconstructed_train_x
            recon_y = self.reconstructed_train_y
            title = "Training"
        elif data_type.lower() in ["val", "validation"]:
            data_x = self.val_x
            data_y = self.val_y
            recon_x = self.reconstructed_val_x
            recon_y = self.reconstructed_val_y
            title = "Validation"

        rand_idx = np.random.randint(0, data_x.shape[0], size=n)

        for id in rand_idx:
            id_x = (
                data_x[id, :, :, :].numpy().reshape(-1)
                if len(data_x.shape) > 2
                else data_x[id, :].numpy().reshape(-1)
            )
            recon_id_x = recon_x[id, :].numpy().reshape(-1)
            rmse_x = np.sqrt(np.mean((id_x - recon_id_x) ** 2))

            vmin = min(np.min(id_x), np.min(recon_id_x))
            vmax = max(np.max(id_x), np.max(recon_id_x))

            ssim_x = ssim(
                id_x.reshape((ny, nx), order="C"),
                recon_id_x.reshape((ny, nx), order="C"),
                data_range=vmax - vmin,
                gaussian_weights=True,
            )

            if joint_output:
                fig, (ax1, ax2, ax3) = plt.subplots(1, ncol, figsize=(30, 15))

                id_y = data_y[id, :].numpy().reshape(-1)
                recon_id_y = recon_y[id, :].numpy().reshape(-1)
                rmse_y = np.sqrt(np.mean((id_y - recon_id_y) ** 2))

                im1 = ax1.imshow(id_x.reshape(ny, nx, order="C"))
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.title.set_text("True")
                im1.set_clim(vmin, vmax)
                ax1_divider = make_axes_locatable(ax1)
                cax = ax1_divider.append_axes("right", size="5%", pad="2%")
                fig.colorbar(im1, cax=cax)

                im2 = ax2.imshow(recon_id_x.reshape(ny, nx, order="C"))
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.title.set_text(
                    f"Reconstructed, rmse = {rmse_x:.3f}, ssim = {ssim_x:.3f}"
                )
                im2.set_clim(vmin, vmax)
                ax2_divider = make_axes_locatable(ax2)
                cax = ax2_divider.append_axes("right", size="5%", pad="2%")
                fig.colorbar(im2, cax=cax)

                ax3.plot(id_y, label="True")
                ax3.plot(recon_id_y, label="Reconstructed")
                ax3.title.set_text(f"Reconstruction rmse = {rmse_y:.3f}")
                ax3.legend(fontsize="small")
                ax3.spines["top"].set_visible(False)
                ax3.spines["right"].set_visible(False)
                plt.tight_layout()
                save_location = (
                    self.reconstructions_diag_dir
                    + f"/True_vs_recon_{title}_joint_{id}.pdf"
                )

            else:
                fig, (ax1, ax2) = plt.subplots(1, ncol, figsize=(30, 15))

                im1 = ax1.imshow(id_x.reshape(ny, nx, order="C"))
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.title.set_text("True")
                im1.set_clim(vmin, vmax)
                ax1_divider = make_axes_locatable(ax1)
                cax = ax1_divider.append_axes("right", size="5%", pad="2%")
                fig.colorbar(im1, cax=cax)

                im2 = ax2.imshow(recon_id_x.reshape(ny, nx, order="C"))
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.title.set_text("Reconstructed")
                im2.set_clim(vmin, vmax)
                ax2_divider = make_axes_locatable(ax2)
                cax = ax2_divider.append_axes("right", size="5%", pad="2%")
                fig.colorbar(im2, cax=cax)
                save_location = (
                    self.reconstructions_diag_dir + f"/True_vs_recon_{title}_{id}.pdf"
                )

            plt.tight_layout()
            plt.savefig(save_location, dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()

    def plot_sd_boxplots(
        self,
        joint_output=True,
        whis_low=0,
        whis_high=100,
        dpi=600,
        show=False,
        **kwargs,
    ):
        """
        Plot sns boxplots of pixel wise standard deviations comparing training, training reconstruction, validation reconstruction and generated.
        :param joint_output: whether the model being diagnosed is a joint distribution model. If true, both x and y data will be plotted.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to plot_boxplots()
        """
        print("Plotting standard deviation boxplots...")

        x_sd_array = np.array(
            [
                self.results_vecs["sd_train_x_vec"],
                self.results_vecs["sd_recon_train_x_vec"],
                self.results_vecs["sd_recon_val_x_vec"],
                self.results_vecs["sd_gen_x_vec"],
            ]
        ).reshape(1, 4, -1)
        x_sd_labels = ["Train", "Recons.-Train", "Recons.-Val", "Generated"]
        plot.plot_boxplots(
            values_all=x_sd_array,
            labels=x_sd_labels,
            axes_plot_titles=[
                [""],
                "",
                "Models pixel wise standard deviation values distribution",
            ],
            whis_low=whis_low,
            whis_high=whis_high,
            save_location=f"{self.logging_dir}/models_pixels_std_boxplots.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        if joint_output:
            y_sd_array = np.array(
                [
                    self.results_vecs["sd_train_y_vec"],
                    self.results_vecs["sd_recon_train_y_vec"],
                    self.results_vecs["sd_recon_val_y_vec"],
                    self.results_vecs["sd_gen_y_vec"],
                ]
            ).reshape(1, 4, -1)
            y_sd_labels = ["Train", "Recons.-Train", "Recons.-Val", "Generated"]

            plot.plot_boxplots(
                values_all=y_sd_array,
                labels=y_sd_labels,
                axes_plot_titles=[
                    [""],
                    "",
                    "Travel times components wise standard deviation values distribution",
                ],
                whis_low=whis_low,
                whis_high=whis_high,
                save_location=f"{self.logging_dir}/TT_comps_std_boxplots.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

    def plot_summary_stats(self, joint_output=True, dpi=600, show=False, **kwargs):
        """
        Plot mean, variance and std.
        :param joint_output: whether the model being diagnosed is a joint distribution model. If true, both x and y data will be plotted.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to plot_boxplots()
        """
        print("Plotting summary statistics...")

        nx = self.experiment.nx
        ny = self.experiment.ny

        titles = ["Train", "Recons.-Train", "Recons.-Test", "Generated"]

        x_mean_array = np.concatenate(
            [
                self.results_vecs["mean_train_x_vec"].reshape((1, ny, nx), order="C"),
                self.results_vecs["mean_recon_train_x_vec"].reshape(
                    (1, ny, nx), order="C"
                ),
                self.results_vecs["mean_recon_val_x_vec"].reshape(
                    (1, ny, nx), order="C"
                ),
                self.results_vecs["mean_gen_x_vec"].reshape((1, ny, nx), order="C"),
            ],
            axis=0,
        )

        x_sd_array = np.concatenate(
            [
                self.results_vecs["sd_train_x_vec"].reshape((1, ny, nx), order="C"),
                self.results_vecs["sd_recon_train_x_vec"].reshape(
                    (1, ny, nx), order="C"
                ),
                self.results_vecs["sd_recon_val_x_vec"].reshape((1, ny, nx), order="C"),
                self.results_vecs["sd_gen_x_vec"].reshape((1, ny, nx), order="C"),
            ],
            axis=0,
        )

        x_var_array = x_sd_array**2

        plot.plot_matrices(
            x_mean_array,
            titles,
            plot_title="Models pixels means",
            save_location=f"{self.logging_dir}/models_pixels_mean.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        plot.plot_matrices(
            x_sd_array,
            titles,
            plot_title="Models pixels standard deviations",
            save_location=f"{self.logging_dir}/models_pixels_sd.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        plot.plot_matrices(
            x_var_array,
            titles,
            plot_title="Models pixels variances",
            save_location=f"{self.logging_dir}/models_pixels_var.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        vmin_x = min(
            np.min(self.results_vecs["covm_train_x"]),
            np.min(self.results_vecs["covm_recon_train_x"]),
            np.min(self.results_vecs["covm_val_x"]),
            np.min(self.results_vecs["covm_recon_val_x"]),
            np.min(self.results_vecs["covm_gen_x"]),
        )
        vmax_x = max(
            np.max(self.results_vecs["covm_train_x"]),
            np.max(self.results_vecs["covm_recon_train_x"]),
            np.max(self.results_vecs["covm_val_x"]),
            np.max(self.results_vecs["covm_recon_val_x"]),
            np.max(self.results_vecs["covm_gen_x"]),
        )

        plot.plot_cov(
            self.results_vecs["covm_train_x"],
            "Covariance matrix of trainining X",
            vmin_vmax=(vmin_x, vmax_x),
            save_location=f"{self.logging_dir}/covm_train_x.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )
        plot.plot_cov(
            self.results_vecs["covm_recon_train_x"],
            "Covariance matrix of reconstructed training X",
            vmin_vmax=(vmin_x, vmax_x),
            save_location=f"{self.logging_dir}/covm_recon_train_x.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )
        plot.plot_cov(
            self.results_vecs["covm_val_x"],
            "Covariance matrix of validation X",
            vmin_vmax=(vmin_x, vmax_x),
            save_location=f"{self.logging_dir}/covm_val_x.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )
        plot.plot_cov(
            self.results_vecs["covm_recon_val_x"],
            "Covariance matrix of reconstructed validation X",
            vmin_vmax=(vmin_x, vmax_x),
            save_location=f"{self.logging_dir}/covm_recon_val_x.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )
        plot.plot_cov(
            self.results_vecs["covm_gen_x"],
            "Covariance matrix of generated X",
            vmin_vmax=(vmin_x, vmax_x),
            save_location=f"{self.logging_dir}/covm_gen_x.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        if joint_output:
            e_r = int(np.sqrt(self.experiment.dim_y))

            y_mean_array = np.concatenate(
                [
                    self.results_vecs["mean_train_y_vec"].reshape(
                        (1, e_r, e_r), order="C"
                    ),
                    self.results_vecs["mean_recon_train_y_vec"].reshape(
                        (1, e_r, e_r), order="C"
                    ),
                    self.results_vecs["mean_recon_val_y_vec"].reshape(
                        (1, e_r, e_r), order="C"
                    ),
                    self.results_vecs["mean_gen_y_vec"].reshape(
                        (1, e_r, e_r), order="C"
                    ),
                ],
                axis=0,
            )

            y_sd_array = np.concatenate(
                [
                    self.results_vecs["sd_train_y_vec"].reshape(
                        (1, e_r, e_r), order="C"
                    ),
                    self.results_vecs["sd_recon_train_y_vec"].reshape(
                        (1, e_r, e_r), order="C"
                    ),
                    self.results_vecs["sd_recon_val_y_vec"].reshape(
                        (1, e_r, e_r), order="C"
                    ),
                    self.results_vecs["sd_gen_y_vec"].reshape((1, e_r, e_r), order="C"),
                ],
                axis=0,
            )

            y_var_array = y_sd_array**2

            plot.plot_matrices(
                y_mean_array,
                titles,
                plot_title="Travel times components means",
                save_location=f"{self.logging_dir}/TT_comps_mean.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

            plot.plot_matrices(
                y_sd_array,
                titles,
                plot_title="Travel times components standard deviations",
                save_location=f"{self.logging_dir}/TT_comps_sd.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

            plot.plot_matrices(
                y_var_array,
                titles,
                plot_title="Travel times components variances",
                save_location=f"{self.logging_dir}/TT_comps_var.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

            vmin_y = min(
                np.min(self.results_vecs["covm_train_y"]),
                np.min(self.results_vecs["covm_recon_train_y"]),
                np.min(self.results_vecs["covm_val_y"]),
                np.min(self.results_vecs["covm_recon_val_y"]),
                np.min(self.results_vecs["covm_gen_y"]),
            )
            vmax_y = max(
                np.max(self.results_vecs["covm_train_y"]),
                np.max(self.results_vecs["covm_recon_train_y"]),
                np.max(self.results_vecs["covm_val_y"]),
                np.max(self.results_vecs["covm_recon_val_y"]),
                np.max(self.results_vecs["covm_gen_y"]),
            )

            plot.plot_cov(
                self.results_vecs["covm_train_y"],
                "Covariance matrix of trainining Y",
                vmin_vmax=(vmin_y, vmax_y),
                save_location=f"{self.logging_dir}/covm_train_y.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )
            plot.plot_cov(
                self.results_vecs["covm_recon_train_y"],
                "Covariance matrix of reconstructed training Y",
                vmin_vmax=(vmin_y, vmax_y),
                save_location=f"{self.logging_dir}/covm_recon_train_y.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )
            plot.plot_cov(
                self.results_vecs["covm_val_y"],
                "Covariance matrix of validation Y",
                vmin_vmax=(vmin_y, vmax_y),
                save_location=f"{self.logging_dir}/covm_val_y.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )
            plot.plot_cov(
                self.results_vecs["covm_recon_val_y"],
                "Covariance matrix of reconstructed validation Y",
                vmin_vmax=(vmin_y, vmax_y),
                save_location=f"{self.logging_dir}/covm_recon_val_y.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )
            plot.plot_cov(
                self.results_vecs["covm_gen_y"],
                "Covariance matrix of generated Y",
                vmin_vmax=(vmin_y, vmax_y),
                save_location=f"{self.logging_dir}/covm_gen_y.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

    def plot_variance_loss_hists(
        self, joint_output=True, dpi=600, show=False, **kwargs
    ):
        """
        Plot histograms of variance loss.
        :param joint_output: whether the model being diagnosed is a joint distribution model. If true, both x and y data will be plotted.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to plot_histograms()
        """
        print("Plotting variance loss histograms...")

        colors = ["lightseagreen", "mediumseagreen", "slateblue"]

        x_var_loss = np.concatenate(
            [
                self.results_vecs["var_loss_train_x_vec"].reshape(1, -1),
                self.results_vecs["var_loss_val_x_vec"].reshape(1, -1),
                self.results_vecs["var_loss_train_gen_x_vec"].reshape(1, -1),
            ],
            axis=0,
        )

        x_var_loss_labels = ["Train Recon.", "Val. Recon.", "Gen. vs Train"]

        plot.plot_histograms(
            x_var_loss,
            x_var_loss_labels,
            colors=colors,
            plot_title="Models pixels variance loss",
            save_location=f"{self.logging_dir}/models_variance_loss.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        if joint_output:
            y_var_loss = np.concatenate(
                [
                    self.results_vecs["var_loss_train_y_vec"].reshape(1, -1),
                    self.results_vecs["var_loss_val_y_vec"].reshape(1, -1),
                    self.results_vecs["var_loss_train_gen_y_vec"].reshape(1, -1),
                ],
                axis=0,
            )

            y_var_loss_labels = ["Train Recon.", "Val. Recon.", "Gen. vs Train"]

            plot.plot_histograms(
                y_var_loss,
                y_var_loss_labels,
                colors=colors,
                plot_title="Travel times components variance loss",
                save_location=f"{self.logging_dir}/TT_comps_variance_loss.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

    def plot_recon_rmse_hists(self, joint_output=True, dpi=600, show=False, **kwargs):
        """
        Plot histograms of reconstruction rmse.
        :param joint_output: whether the model being diagnosed is a joint distribution model. If true, both x and y data will be plotted.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to plot_histograms()
        """
        print("Plotting reconstruction rmse histograms...")

        colors = ["lightseagreen", "darkgreen"]
        labels = ["Train", "Validation"]

        x_recon_rmse = np.concatenate(
            [
                self.results_vecs["recon_train_x_rmse_vec"].reshape(1, -1),
                self.results_vecs["recon_val_x_rmse_vec"].reshape(1, -1),
            ],
            axis=0,
        )

        plot.plot_histograms(
            x_recon_rmse,
            labels,
            colors=colors,
            plot_title="Reconstruction RMSE on models (ns/m)",
            save_location=f"{self.reconstructions_diag_dir}/models_recon_rmse.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        if joint_output:
            y_recon_rmse = np.concatenate(
                [
                    self.results_vecs["recon_train_y_rmse_vec"].reshape(1, -1),
                    self.results_vecs["recon_val_y_rmse_vec"].reshape(1, -1),
                ],
                axis=0,
            )

            plot.plot_histograms(
                y_recon_rmse,
                labels,
                colors=colors,
                plot_title="Reconstruction RMSE on travel times (ns)",
                save_location=f"{self.reconstructions_diag_dir}/TT_recon_rmse.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

    def plot_x_y_umap_tsne_scatters(
        self,
        n_neighbors=100,
        min_dist=0.3,
        dpi=600,
        show=False,
        fit_prior_only=False,
        denseMAP=True,
        **kwargs,
    ):
        """
        Plot UMAP & TSNE scatters of training, validation and generated data
        :param n_neighbors: number of neighbors to use to compute the UMAP embedding
        :param min_dist: minimum distance between points in the UMAP embedding
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param fit_prior_only: whether to fit UMAP based only on samples from the prior
        :param kwargs: additional arguments to be passed to plot_umap_scatter()
        """
        print(
            "Plotting UMAP & TSNE scatters of training, validation and generated X & Y..."
        )

        def make_dim_redux(data, title_string, type):
            """
            data is a list of tensors [train, recon_train, recon_val, generated]
            """
            colors = ["grey", "cornflowerblue", "lightseagreen", "lightcoral"]
            labels = ["Train", "Recon.-Train", "Recon.-Val", "Generated"]
            markers = ["o", ".", "x", "*"]

            if fit_prior_only:
                # only train
                all_data = data[0].numpy()
            else:
                # concatenate train, reconstructions and generated
                all_data = np.vstack(
                    [data[0].numpy(), data[1].numpy(), data[2].numpy(), data[3].numpy()]
                )

            if type.lower() == "umap":
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=self.experiment.seed,
                    n_components=2,
                    densmap=denseMAP,
                )

            elif type.lower() == "tsne":
                if not fit_prior_only:
                    reducer = TSNE(
                        perplexity=n_neighbors,
                        n_components=2,
                        random_state=self.experiment.seed,
                    )
                else:
                    raise ValueError(
                        "TSNE cannot be fitted on prior samples only. Use UMAP instead."
                    )
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

                embeddings_train = np.expand_dims(
                    np.hstack(
                        (
                            np.expand_dims(
                                embeddings[train_id_start:train_id_end, 0], axis=1
                            ),
                            np.expand_dims(
                                embeddings[train_id_start:train_id_end, 1], axis=1
                            ),
                        )
                    ),
                    axis=0,
                )
                embeddings_recon_train = np.expand_dims(
                    np.hstack(
                        (
                            np.expand_dims(
                                embeddings[recon_train_id_start:recon_train_id_end, 0],
                                axis=1,
                            ),
                            np.expand_dims(
                                embeddings[recon_train_id_start:recon_train_id_end, 1],
                                axis=1,
                            ),
                        )
                    ),
                    axis=0,
                )
                embeddings_recon_val = np.expand_dims(
                    np.hstack(
                        (
                            np.expand_dims(
                                embeddings[recon_val_id_start:recon_val_id_end, 0],
                                axis=1,
                            ),
                            np.expand_dims(
                                embeddings[recon_val_id_start:recon_val_id_end, 1],
                                axis=1,
                            ),
                        )
                    ),
                    axis=0,
                )
                embeddings_gen = np.expand_dims(
                    np.hstack(
                        (
                            np.expand_dims(
                                embeddings[gen_id_start:gen_id_end, 0], axis=1
                            ),
                            np.expand_dims(
                                embeddings[gen_id_start:gen_id_end, 1], axis=1
                            ),
                        )
                    ),
                    axis=0,
                )
            else:
                embeddings_train = np.expand_dims(embeddings, axis=0)
                embeddings_recon_train = np.expand_dims(
                    reducer.transform(data[1].numpy()), axis=0
                )
                embeddings_recon_val = np.expand_dims(
                    reducer.transform(data[2].numpy()), axis=0
                )
                embeddings_gen = np.expand_dims(
                    reducer.transform(data[3].numpy()), axis=0
                )

            embeddings_data = np.concatenate(
                [
                    embeddings_train,
                    embeddings_recon_train,
                    embeddings_recon_val,
                    embeddings_gen,
                ],
                axis=0,
            )

            optional_string = ""
            if type.lower() == "tsne":
                optional_string = f"_KL = {reducer.kl_divergence_:.3f}"

            plot.plot_scatters(
                data=embeddings_data,
                labels=labels,
                plot_title=f"{type} embedding scatters of training, validation and generated {title_string}{optional_string}",
                colors=colors,
                markers=markers,
                save_location=f"{self.logging_dir}/{type}_scatter_{title_string}.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

        if (
            self.train_x is None
            or self.val_x is None
            or self.train_y is None
            or self.val_y is None
        ):
            raise ValueError(
                "No training or validation data to plot. Build diagnostics data first."
            )

        if (
            self.generated_x is None
            or self.reconstructed_train_x is None
            or self.reconstructed_val_x is None
            or self.reconstructed_train_y is None
            or self.reconstructed_val_y is None
        ):
            raise ValueError(
                "No generated or reconstructed data to plot. "
                "Build diagnostics data first."
            )

        dim_x = self.experiment.dim_x
        train_x = self.train_x.view(-1, dim_x)
        generated_x = self.generated_x
        recon_train_x = self.reconstructed_train_x
        recon_val_x = self.reconstructed_val_x

        make_dim_redux(
            [train_x, recon_train_x, recon_val_x, generated_x], "X", type="umap"
        )
        make_dim_redux(
            [train_x, recon_train_x, recon_val_x, generated_x], "X", type="tsne"
        )

        dim_y = self.experiment.dim_y
        train_y = self.train_y.view(-1, dim_y)
        generated_y = self.generated_y
        recon_train_y = self.reconstructed_train_y
        recon_val_y = self.reconstructed_val_y

        make_dim_redux(
            [train_y, recon_train_y, recon_val_y, generated_y], "Y", type="umap"
        )
        make_dim_redux(
            [train_y, recon_train_y, recon_val_y, generated_y], "Y", type="tsne"
        )

    def plot_resimulations(
        self, include_recon_y=True, n=5, dpi=600, show=False, **kwargs
    ):
        """
        Plot true vs reconstructed models and true vs resimulated travel times for both training and validation examples.
        For generated models, plot generated travel times vs resimulated travel times.
        :param n: number of plots to make for each case
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to base_config()
        """
        # TODO : refactor to isolate and generalize plotting part and reuse it

        print("Plotting resimulations...")

        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config(mpl, **kwargs)

        nx = self.experiment.nx
        ny = self.experiment.ny

        rand_idx = np.random.randint(0, self.train_x.shape[0], size=n)

        for i in range(n):
            id = rand_idx[i]

            id_recon_train_x = self.reconstructed_train_x[id, :].numpy().reshape(-1)
            id_train_x = (
                self.train_x[id, :, :, :].numpy().reshape(-1)
                if len(self.train_x.shape) > 2
                else self.train_x[id, :].numpy().reshape(-1)
            )
            id_resim_recon_x = self.resimulated_recon_train_x[id, :].reshape(-1)
            id_train_y = self.train_y[id, :].numpy().reshape(-1)
            id_recon_train_y = self.reconstructed_train_y[id, :].numpy().reshape(-1)

            rmse_resim_vs_train_grd_y = np.sqrt(
                np.mean((id_resim_recon_x - id_train_y) ** 2)
            )
            rmse_resim_vs_train_recon_y = np.sqrt(
                np.mean((id_resim_recon_x - id_recon_train_y) ** 2)
            )

            vmin = min(np.min(id_train_x), np.min(id_recon_train_x))
            vmax = max(np.max(id_train_x), np.max(id_recon_train_x))

            rmse_x = np.sqrt(np.mean((id_train_x - id_recon_train_x) ** 2))
            ssim_x = ssim(
                id_train_x.reshape((ny, nx), order="C"),
                id_recon_train_x.reshape((ny, nx), order="C"),
                data_range=vmax - vmin,
                gaussian_weights=True,
            )

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 15))
            im1 = ax1.imshow(id_train_x.reshape(ny, nx, order="C"))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.title.set_text("True")
            im1.set_clim(vmin, vmax)
            ax1_divider = make_axes_locatable(ax1)
            cax = ax1_divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im1, cax=cax)

            im2 = ax2.imshow(id_recon_train_x.reshape(ny, nx, order="C"))
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.title.set_text(
                f"Reconstructed: rmse = {rmse_x:.3f}, ssim = {ssim_x:.3f}"
            )
            im2.set_clim(vmin, vmax)
            ax2_divider = make_axes_locatable(ax2)
            cax = ax2_divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im2, cax=cax)

            ax3.plot(id_train_y, label="True")
            ax3.plot(id_resim_recon_x, label="Resimulated")
            title_txt = f"RMSE(Y, F(Recon(X))) (ns) = {rmse_resim_vs_train_grd_y:.3f}"
            if include_recon_y:
                ax3.plot(id_recon_train_y, label="Reconstructed")
                title_txt = f"{title_txt}; RMSE(Recon(Y), F(Recon(X))) (ns) = {rmse_resim_vs_train_recon_y:.3f}"
            ax3.legend(fontsize="small")
            ax3.title.set_text(title_txt)
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            plt.tight_layout()
            if show:
                plt.show()
            else:
                plt.savefig(
                    f"{self.resimulations_diag_dir}/train_recon_resim_{id}.pdf",
                    dpi=dpi,
                    bbox_inches="tight",
                )
            plt.close()

            id_recon_val_x = self.reconstructed_val_x[id, :].numpy().reshape(-1)
            id_val_x = (
                self.val_x[id, :, :, :].numpy().reshape(-1)
                if len(self.val_x.shape) > 2
                else self.val_x[id, :].numpy().reshape(-1)
            )
            id_resim_recon_x = self.resimulated_recon_val_x[id, :].reshape(-1)
            id_val_y = self.val_y[id, :].numpy().reshape(-1)
            id_recon_val_y = self.reconstructed_val_y[id, :].numpy().reshape(-1)

            rmse_resim_vs_val = np.sqrt(np.mean((id_resim_recon_x - id_val_y) ** 2))
            rmse_resim_vs_val_recon_y = np.sqrt(
                np.mean((id_resim_recon_x - id_recon_val_y) ** 2)
            )

            rmse_val_x = np.sqrt(np.mean((id_val_x - id_recon_val_x) ** 2))
            ssim_val_x = ssim(
                id_val_x.reshape((ny, nx), order="C"),
                id_recon_val_x.reshape((ny, nx), order="C"),
                data_range=vmax - vmin,
                gaussian_weights=True,
            )

            vmin = min(np.min(id_val_x), np.min(id_recon_val_x))
            vmax = max(np.max(id_val_x), np.max(id_recon_val_x))

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 15))
            im1 = ax1.imshow(id_val_x.reshape(ny, nx, order="C"))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.title.set_text("True")
            im1.set_clim(vmin, vmax)
            ax1_divider = make_axes_locatable(ax1)
            cax = ax1_divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im1, cax=cax)

            im2 = ax2.imshow(id_recon_val_x.reshape(ny, nx, order="C"))
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.title.set_text(
                f"Reconstructed: rmse = {rmse_val_x:.3f}, ssim = {ssim_val_x:.3f}"
            )
            im2.set_clim(vmin, vmax)
            ax2_divider = make_axes_locatable(ax2)
            cax = ax2_divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im2, cax=cax)

            ax3.plot(id_val_y, label="True")
            ax3.plot(id_resim_recon_x, label="Resimulated")
            title_txt = f"RMSE(Y, F(Recon(X))) (ns) = {rmse_resim_vs_val:.3f}"
            if include_recon_y:
                ax3.plot(id_recon_val_y, label="Reconstructed")
                title_txt = f"{title_txt}; RMSE(Recon(Y), F(Recon(X))) (ns) = {rmse_resim_vs_val_recon_y:.3f}"

            ax3.legend(fontsize="small")
            ax3.title.set_text(title_txt)
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            plt.tight_layout()
            if show:
                plt.show()
            else:
                plt.savefig(
                    f"{self.resimulations_diag_dir}/val_recon_resim_{id}.pdf",
                    dpi=dpi,
                    bbox_inches="tight",
                )
            plt.close()

            id_gen_x = self.generated_x[id, :].numpy().reshape(-1)
            id_resim_gen_x = self.resimulated_gen_x[id, :].reshape(-1)
            id_gen_y = self.generated_y[id, :].numpy().reshape(-1)
            rmse_resim_vs_gen = np.sqrt(np.mean((id_resim_gen_x - id_gen_y) ** 2))
            vmin = min(np.min(id_gen_x), np.min(id_gen_x))
            vmax = max(np.max(id_gen_x), np.max(id_gen_x))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
            im1 = ax1.imshow(id_gen_x.reshape(ny, nx, order="C"))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.title.set_text("Generated")
            im1.set_clim(vmin, vmax)
            ax1_divider = make_axes_locatable(ax1)
            cax = ax1_divider.append_axes("right", size="5%", pad="2%")
            fig.colorbar(im1, cax=cax)

            ax2.plot(id_gen_y, label="Generated")
            ax2.plot(id_resim_gen_x, label="Resimulated")
            ax2.legend(fontsize="small")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.title.set_text(f"Resimulation rmse (ns) = {rmse_resim_vs_gen:.3f}")
            plt.tight_layout()
            if show:
                plt.show()
            else:
                plt.savefig(
                    f"{self.resimulations_diag_dir}/gen_resim_{id}.pdf",
                    dpi=dpi,
                    bbox_inches="tight",
                )
            plt.close()

    def plot_resim_rmse_hists(
        self, include_recon_y=True, dpi=600, show=False, **kwargs
    ):
        """
        Plot histograms of resimulation rmse.
        :param include_recon_y: whether to include the resimulation rmse with respect to the reconstructed y
                                as a reference, instead of ground truth data
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to plot_histograms()
        """
        print("Plotting resimulation rmse histograms...")

        colors = ["lightseagreen", "mediumseagreen", "slateblue"]
        labels = ["Training", "Validation", "Generated"]

        resim_rmse_vecs = np.concatenate(
            [
                self.results_vecs["rmse_train_y_vs_recon_x_resim_vec"].reshape(1, -1),
                self.results_vecs["rmse_val_y_vs_recon_x_resim_vec"].reshape(1, -1),
                self.results_vecs["rmse_gen_y_vs_gen_x_resim_vec"].reshape(1, -1),
            ],
            axis=0,
        )
        plot.plot_histograms(
            resim_rmse_vecs,
            labels,
            colors=colors,
            plot_title="Resimulation RMSE on travel times (ns) - Ground truth Y (train & val)",
            save_location=f"{self.resimulations_diag_dir}/resim_rmse_hist_grdt_y.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        if include_recon_y:
            resim_rmse_vecs = np.concatenate(
                [
                    self.results_vecs[
                        "rmse_train_recon_y_vs_recon_x_resim_vec"
                    ].reshape(1, -1),
                    self.results_vecs["rmse_val_recon_y_vs_recon_x_resim_vec"].reshape(
                        1, -1
                    ),
                    self.results_vecs["rmse_gen_y_vs_gen_x_resim_vec"].reshape(1, -1),
                ],
                axis=0,
            )
            plot.plot_histograms(
                resim_rmse_vecs,
                labels,
                colors=colors,
                plot_title="Resimulation RMSE on travel times (ns) - Reconstructed Y (train & val)",
                save_location=f"{self.resimulations_diag_dir}/resim_rmse_hist_recon_y.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

    def plot_resim_rmse_boxplots(
        self, include_recon_y=True, dpi=600, show=False, **kwargs
    ):
        """
        Plot boxplots of resimulation rmse.
        :param include_recon_y: whether to include the resimulation rmse with respect to the reconstructed y
                                as a reference, instead of ground truth data
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to plot_boxplots()
        """
        print("Plotting resimulation rmse boxplots...")

        resim_rmse_labels = ["Training", "Validation", "Generated"]

        resim_rmse_vecs = np.array(
            [
                self.results_vecs["rmse_train_y_vs_recon_x_resim_vec"],
                self.results_vecs["rmse_val_y_vs_recon_x_resim_vec"],
                self.results_vecs["rmse_gen_y_vs_gen_x_resim_vec"],
            ]
        ).reshape(1, 3, -1)

        plot.plot_boxplots(
            values_all=resim_rmse_vecs,
            labels=resim_rmse_labels,
            axes_plot_titles=[
                [""],
                "",
                "Resimulation RMSE on travel times (ns) - Ground truth Y (train & val)",
            ],
            save_location=f"{self.resimulations_diag_dir}/resim_rmse_boxplots_grdt_y.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        if include_recon_y:
            resim_rmse_vecs = np.array(
                [
                    self.results_vecs["rmse_train_recon_y_vs_recon_x_resim_vec"],
                    self.results_vecs["rmse_val_recon_y_vs_recon_x_resim_vec"],
                    self.results_vecs["rmse_gen_y_vs_gen_x_resim_vec"],
                ]
            ).reshape(1, 3, -1)

            plot.plot_boxplots(
                values_all=resim_rmse_vecs,
                labels=resim_rmse_labels,
                axes_plot_titles=[
                    [""],
                    "",
                    "Resimulation RMSE on travel times (ns) - Reconstructed Y (train & val)",
                ],
                save_location=f"{self.resimulations_diag_dir}/resim_rmse_boxplots_recon_y.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

    def plot_variograms(
        self,
        detailed_varios,
        datasets_selection=[],
        add_legend=False,
        dpi=600,
        show=False,
        **kwargs,
    ):
        """
        Plot variograms of training, validation and generated data.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to plot_hv_variograms()
        """
        print("Plotting variograms...")

        _datasets_map = {
            "train": self.train_x,
            "val": self.val_x,
            "recon_train": self.reconstructed_train_x,
            "recon_val": self.reconstructed_val_x,
            "gen": self.generated_x,
        }

        sample_size = self.train_x.shape[0]
        nx = self.experiment.nx
        ny = self.experiment.ny

        if len(datasets_selection) == 0:
            datasets = [
                v.numpy().reshape(1, sample_size, -1) for k, v in _datasets_map.items()
            ]
            labels = [k for k in _datasets_map.keys()]
        else:
            datasets = [
                _datasets_map[k].numpy().reshape(1, sample_size, -1)
                for k in datasets_selection
            ]
            labels = datasets_selection

        data_x = np.concatenate(datasets, axis=0)

        # data_x = np.concatenate([self.train_x.numpy().reshape(1, sample_size, -1),
        #                         self.val_x.numpy().reshape(1, sample_size, -1),
        #                         self.reconstructed_train_x.numpy().reshape(1, sample_size, -1),
        #                         self.reconstructed_val_x.numpy().reshape(1, sample_size, -1),
        #                         self.generated_x.numpy().reshape(1, sample_size, -1)], axis = 0)

        plot.plot_hv_variograms(
            data_x,
            [nx, ny],
            detailed_varios,
            labels=labels,  # ['Train', 'Validation', 'Recon.-Train', 'Recon.-Val', 'Generated'],
            seperate_vario=False,
            add_legend=add_legend,
            save_location=[
                f"{self.logging_dir}/variograms_h.pdf",
                f"{self.logging_dir}/variograms_v.pdf",
            ],
            dpi=dpi,
            show=show,
            **kwargs,
        )

    def plot_latent_histograms(self, nl=6, nco=5, dpi=600, show=False, **kwargs):
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
                    ref = i - id * 30
                    axes[ref // ncol, ref - (ref // ncol * ncol)].hist(
                        codes[:, i].reshape(-1).numpy(),
                        bins=30,
                        rwidth=0.8,
                        color="grey",
                        density=True,
                    )
                    axes[ref // ncol, ref - (ref // ncol * ncol)].spines[
                        "top"
                    ].set_visible(False)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].spines[
                        "right"
                    ].set_visible(False)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].spines[
                        "bottom"
                    ].set_visible(False)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].spines[
                        "left"
                    ].set_visible(False)
                    axes[ref // ncol, ref - (ref // ncol * ncol)].set_xlabel(f"z_{i}")
                plt.tight_layout()
                if show:
                    plt.show()
                else:
                    plt.savefig(
                        f"{save_location_dir}/{file_name_key}_histograms_{id}.pdf",
                        dpi=dpi,
                        bbox_inches="tight",
                    )
                plt.close()

        # plot latent_train_codes histograms
        plot_hist(
            self.latent_train_codes, "latent_train_codes", self.latent_space_diag_dir
        )

        # plot latent_val_codes histograms
        plot_hist(self.latent_val_codes, "latent_val_codes", self.latent_space_diag_dir)

        # plot latent prior samples histograms
        plot_hist(
            self.latent_vector, "latent_prior_samples", self.latent_space_diag_dir
        )

    def plot_latent_cov_matrices(self, dpi=600, show=False, **kwargs):
        """
        Plot covariance matrices of latent codes.
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param kwargs: additional arguments to be passed to base_config()
        """
        print("Plotting latent covariance matrices...")

        vmin = min(
            np.min(self.latent_prior_covm),
            np.min(self.latent_train_codes_covm),
            np.min(self.latent_val_codes_covm),
        )
        vmax = max(
            np.max(self.latent_prior_covm),
            np.max(self.latent_train_codes_covm),
            np.max(self.latent_val_codes_covm),
        )

        plot.plot_cov(
            self.latent_prior_covm,
            plot_title="Latent prior covariance matrix",
            vmin_vmax=(vmin, vmax),
            save_location=f"{self.latent_space_diag_dir}/covm_latent_prior.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        plot.plot_cov(
            self.latent_train_codes_covm,
            plot_title="Training latent codes covariance matrix",
            vmin_vmax=(vmin, vmax),
            save_location=f"{self.latent_space_diag_dir}/covm_latent_train_codes.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        plot.plot_cov(
            self.latent_val_codes_covm,
            plot_title="Validation latent codes covariance matrix",
            vmin_vmax=(vmin, vmax),
            save_location=f"{self.latent_space_diag_dir}/covm_latent_val_codes.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        plot.plot_cov(
            self.covm_diff_prior_train,
            plot_title="Prior covm - Train latent codes covm",
            save_location=f"{self.latent_space_diag_dir}/diff_priorcovm_train_codes.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

        plot.plot_cov(
            self.covm_diff_prior_val,
            plot_title="Prior covm - Val latent codes covm",
            save_location=f"{self.latent_space_diag_dir}/diff_priorcovm_val_codes.pdf",
            dpi=dpi,
            show=show,
            **kwargs,
        )

    def plot_latent_umap_tsne_scatters(
        self,
        n_neighbors=100,
        min_dist=0.3,
        dpi=600,
        show=False,
        fit_prior_only=False,
        denseMAP=True,
        **kwargs,
    ):
        """
        Plot UMAP scatters of training and validation latent codes.
        :param n_neighbors: number of neighbors to use to compute the UMAP embedding
        :param min_dist: minimum distance between points in the UMAP embedding
        :param dpi: resolution of the image in dots per inch (dpi)
        :param show: whether to show the plot
        :param fit_prior_only: whether to fit UMAP based only on samples from the prior
        :param kwargs: additional arguments to be passed to plot_scatters()
        """
        print(
            "Plotting UMAP & TSNE scatters of training and validation latent codes..."
        )

        def make_dim_redux(type):
            colors = ["grey", "cornflowerblue", "mediumvioletred"]
            labels = ["Prior", "Training", "Validation"]

            if fit_prior_only:
                all_data = self.latent_vector.numpy()
            else:
                all_data = np.vstack(
                    [
                        self.latent_vector.numpy(),
                        self.latent_train_codes.numpy(),
                        self.latent_val_codes.numpy(),
                    ]
                )

            if type.lower() == "umap":
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=self.experiment.seed,
                    n_components=2,
                    densmap=denseMAP,
                )
            elif type.lower() == "tsne":
                if not fit_prior_only:
                    reducer = TSNE(
                        perplexity=n_neighbors,
                        n_components=2,
                        random_state=self.experiment.seed,
                    )
                else:
                    raise ValueError(
                        "TSNE cannot be fitted on prior samples only. Use UMAP instead."
                    )
            else:
                raise ValueError("Unknown dimensionality reduction method.")

            embeddings = reducer.fit_transform(all_data)

            # read embedding for each data type separately. For each, concatenate the 1st and 2nd components of the embedding
            if not fit_prior_only:
                latent_id_start = 0
                latent_id_end = self.latent_vector.shape[0]
                latent_train_id_start = latent_id_end
                latent_train_id_end = (
                    latent_train_id_start + self.latent_train_codes.shape[0]
                )
                latent_val_id_start = latent_train_id_end
                latent_val_id_end = latent_val_id_start + self.latent_val_codes.shape[0]

                embeddings_latent = np.expand_dims(
                    np.hstack(
                        (
                            np.expand_dims(
                                embeddings[latent_id_start:latent_id_end, 0], axis=1
                            ),
                            np.expand_dims(
                                embeddings[latent_id_start:latent_id_end, 1], axis=1
                            ),
                        )
                    ),
                    axis=0,
                )
                embeddings_latent_train = np.expand_dims(
                    np.hstack(
                        (
                            np.expand_dims(
                                embeddings[
                                    latent_train_id_start:latent_train_id_end, 0
                                ],
                                axis=1,
                            ),
                            np.expand_dims(
                                embeddings[
                                    latent_train_id_start:latent_train_id_end, 1
                                ],
                                axis=1,
                            ),
                        )
                    ),
                    axis=0,
                )
                embeddings_latent_val = np.expand_dims(
                    np.hstack(
                        (
                            np.expand_dims(
                                embeddings[latent_val_id_start:latent_val_id_end, 0],
                                axis=1,
                            ),
                            np.expand_dims(
                                embeddings[latent_val_id_start:latent_val_id_end, 1],
                                axis=1,
                            ),
                        )
                    ),
                    axis=0,
                )
            else:
                embeddings_latent = np.expand_dims(embeddings, axis=0)
                embeddings_latent_train = np.expand_dims(
                    reducer.transform(self.latent_train_codes.numpy()), axis=0
                )
                embeddings_latent_val = np.expand_dims(
                    reducer.transform(self.latent_val_codes.numpy()), axis=0
                )

            embedding_data = np.concatenate(
                [embeddings_latent, embeddings_latent_train, embeddings_latent_val],
                axis=0,
            )

            optional_string = ""
            if type.lower() == "tsne":
                optional_string = f"_KL = {reducer.kl_divergence_:.3f}"

            plot.plot_scatters(
                data=embedding_data,
                labels=labels,
                plot_title=f"{type} embedding scatters of latent prior samples, training and validation latent codes{optional_string}",
                colors=colors,
                save_location=f"{self.latent_space_diag_dir}/latent_{type}_scatter.pdf",
                dpi=dpi,
                show=show,
                **kwargs,
            )

        make_dim_redux(type="umap")
        make_dim_redux(type="tsne")
