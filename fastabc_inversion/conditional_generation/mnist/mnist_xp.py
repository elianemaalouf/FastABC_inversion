"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Script to run conditional generation experiments on the MNIST dataset.
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from torch import nn
import numpy as np
import pandas as pd
from torchvision import transforms
from fastabc_inversion.conditional_generation.experiment import Experiment, inspect_data
from fastabc_inversion.utils.utilities import toggle_spectral_norm, print_spectral_norm_status
from fastabc_inversion.conditional_generation.utils.label_transform import LabelTransform
from fastabc_inversion.utils.utilities import load_experiment_from_file

class MNIST_XP(Experiment):
    def __init__(self, base_config):
        super().__init__(**base_config)

    def load_data(self, type = ['train', 'test'], batch_size = 128, run_data_inspection = False, **kwargs):
        """
        Load the MNIST data
        type: 'train' or 'test'
        image_size: size of image to resize to
        batch_size: batch size for data loader
        run_data_inspection: if True, display information about the data and show some samples
        kwargs: additional arguments for data inspection
        Returns:
            data_loader: DataLoader object for the specified type
        """

        import torchvision
        from torch.utils.data import DataLoader, Subset
        from sklearn.model_selection import train_test_split

        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)) # makes pixel values in [-1, 1], if not, values are in [0, 1]
        ])

        self.label_transform = LabelTransform(num_classes=self.dim_y, delta=None)

        if 'train' in type:
            mnist_train = torchvision.datasets.MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=self.image_transform,
                target_transform= self.label_transform # transform labels to one-hot encoded vectors
            )

            # generate indices: instead of the actual data we pass in integers instead
            train_indices, val_indices, _, _ = train_test_split(
                range(len(mnist_train)),
                mnist_train.targets,
                stratify=mnist_train.targets,
                test_size=0.1,
            )

            # generate a subset based on indices
            train_split = Subset(mnist_train, train_indices) # size = 54000
            val_split = Subset(mnist_train, val_indices) # size = 6000

            # create batches
            self.train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=True)

            self.train_size = len(self.train_loader) * batch_size
            self.val_size = len(self.val_loader) * batch_size

        if 'test' in type:
            mnist_test = torchvision.datasets.MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=self.image_transform,
                target_transform=self.label_transform # transform labels to one-hot encoded vectors
            )
            self.test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
            self.test_size = len(self.test_loader) * batch_size

        if run_data_inspection:
            if 'train' in type:
                print("Inspecting training data:")
                inspect_data(self.train_loader, save_fig_path =f"{self.experiment_dir}/train_loader_Inspectexamples.pdf", **kwargs)
                print("Inspecting validation data:")
                inspect_data(self.val_loader, save_fig_path =f"{self.experiment_dir}/val_loader_Inspectexamples.pdf", **kwargs)
            if 'test' in type:
                print("Inspecting test data:")
                inspect_data(self.test_loader, save_fig_path =f"{self.experiment_dir}/test_loader_Inspectexamples.pdf", **kwargs)

    def construct_model_architecture(self, model_selector, nn_params):
        if model_selector == 'old_architecture':
            self.construct_model_architecture_1(model_selector, nn_params)
        else:
            self.construct_model_architecture_2(model_selector, nn_params)

    def construct_model_architecture_1(self, model_selector, nn_params):
        print(f"Constructing jGNN model architecture, based on old model...")
        from fastabc_inversion.utils.utilities import params_init

        import fastabc_inversion.conditional_generation.nn.nnmodels_WAE as nnm

        self.nn_params = nn_params
        self.model_selector = model_selector

        self.latent_dim = nn_params.get("latent_dim", None)
        ngf = 64
        ndf = 64  # number of out channels (in conv layers)
        dfs = 5  # kernel size (in conv layers)

        self.netG = nnm.netG(nc=1, ngf=ngf, ngpu=1, ndata=self.dim_y, dz=self.latent_dim, npx=self.image_size, npy=self.image_size)

        self.netD = nnm.netD(nc=1, ndf=ndf, dfs=dfs, ngpu=1, ndata=self.dim_y, npx=self.image_size, npy=self.image_size, dz=self.latent_dim)

        # build full model

        self.model = nnm.netWae(
            encoder=self.netD, decoder=self.netG, ngpu=self.ngpu
        )  # encoder - decoder

        self.save_model_summary(
            input_dim=[(64, 1, 32, 32), (64, 10)]
        )

        verbose = nn_params.get("verbose", False)
        _map_module_init_param = {
            "netD": ("leaky_relu", None, verbose),
            "netG": ("relu", None, verbose),
        }
        self.model.weight_init(
            params_init, _map_module_init_param["netD"], _map_module_init_param["netG"]
        )

    def construct_model_architecture_2(self, model_selector, nn_params):
        """
        """
        print(f"Constructing jGNN model architecture...")

        from fastabc_inversion.utils.utilities import params_init

        import fastabc_inversion.conditional_generation.nn.jgnn as nnm

        self.latent_dim = nn_params.get("latent_dim", None)
        if self.latent_dim is None:
            raise ValueError("latent_dim must be specified in nn_params")

        netG_constrained_prelu = nn_params.get("netG_constrained_prelu", False)
        netG_apply_spectral_norm = nn_params.get("netG_apply_spectral_norm", True)
        netD_apply_spectral_norm = nn_params.get("netD_apply_spectral_norm", False)

        netG_activation = 'constrained_prelu' if netG_constrained_prelu else 'prelu'

        verbose = nn_params.get("verbose", False)

        self.nn_params = nn_params
        self.model_selector = model_selector

        # build neural networks and initialize weights
        self.netD = nnm.netD(input_image_channels=1, encoder_channels=128, ngpu=self.ngpu, input_label_size=10,
                             input_image_height=32, input_image_width=32, latent_dim=self.latent_dim)  # encoder
        netD_layer_types = (nn.Conv2d, nn.Linear)
        toggle_spectral_norm(self.netD, netD_apply_spectral_norm, layer_types=netD_layer_types,
                             verbose = verbose)
        print(f"Spectral norm on encoder set to : {netD_apply_spectral_norm}; verifying encoder layers:")
        print_spectral_norm_status(self.netD, layer_types=netD_layer_types)

        self.netG = nnm.netG(output_image_channels=1, decoder_channels=128, ngpu=self.ngpu, output_label_size=10,
                             output_image_height=32, output_image_width=32, latent_dim=self.latent_dim,
                             contrained_prelu = netG_constrained_prelu, image_process_type = model_selector)  # decoder
        netG_layer_types = (nn.ConvTranspose2d, nn.Linear, nn.Conv2d) # Conv2d included for upsample_conv architecture
        toggle_spectral_norm(self.netG, netG_apply_spectral_norm, layer_types=netG_layer_types,
                             verbose = verbose)
        print(f"Spectral norm on decoder set to : {netG_apply_spectral_norm}; verifying decoder layers:")
        print_spectral_norm_status(self.netG, netG_layer_types)

         # build full model

        self.model = nnm.jGNN(
            encoder=self.netD, decoder=self.netG, ngpu=self.ngpu
        )  # encoder - decoder

        self.save_model_summary(
            input_dim=[(64, 1, 32, 32),
                       (64, 10)]
        )

        _map_module_init_param = {
            "netD": ("prelu", None, verbose),
            "netG": (netG_activation, None, verbose),
        }
        self.model.weight_init(
            params_init, _map_module_init_param["netD"], _map_module_init_param["netG"]
        )

# make main
if __name__ == "__main__":
    run_training = False  # Do training once model is created or loaded from checkpoint
    run_jsae_diags = False # Whether to run diagnostics on a pre-trained model

    run_inference = False
    run_infernce_diags = True

    if run_jsae_diags or run_inference or run_infernce_diags:
        # specify epoch and experiment file location to load pre-trained model
        epoch = 91
        exp_name = "MNIST_XP_id_29_ImgDecoder_transposed_conv_XAJI0"
        root_dir = "/media/dl-rookie/Data/Final_thesis_results/Data/Conditional_Generation/MNIST"
        experiment_file_location = f"{root_dir}/{exp_name}/model_training_data/models/experiment_epoch_{epoch}.pkl"

        if run_jsae_diags: # configure diagnostics to run
            diagnostics_to_run = []  # if empty: runs all
            diags_params = {}  # if empty: takes defaults

        if run_inference or run_infernce_diags: # configure inference parameters
            inference_params = {
                "N": 500,
                "p0": 0.1,
                "epsilon_vec": [0.0001],# [0.01],#
                "norm_fct": 'kl_divergence',  # 'l2' (SSE) or 'l1' (SAE) or 'cross_entropy' or 'kl_divergence', 'slant', 'thickness', 'length'
                "max_it": 10, #20, #10,  # 30, # 50,
                "return_full_results":True,
                "sus_runs":1,
            }

            observation_vec = [0,1,2,3,4,5,6,7,8,9]  # example observation: one of each digit
            #observation_vec = [
    #[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55],  # slight perturbation to onehot vector '9'
    #[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8],            # confused with 0 and 8
    #[0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3],            # highly uncertain
    #[0.15, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.7],          # confused with 0 and 4
    #[0.05, 0.05, 0.0, 0.05, 0.1, 0.0, 0.0, 0.0, 0.05, 0.7]         # moderate uncertainty
#]  # perturbed onehot vectors '9'
            #observation_vec = [-40, -30, -20, -10, 0, 10, 20, 30, 40]  # slant observations in degrees
            #observation_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9] # thickness observations
            #observation_vec = [24, 34, 44, 54, 64, 74, 84, 94] # length observations
            num_samples_to_plot = 100 # choose a square number for better visualization in grid
            morphological_fn = ['slant', 'thickness', 'length']

        # create dummy experiment
        dummy_exp = {'seed': None, 'data_dir': None, 'experiment_rootdir': None,
                     'name': None, 'image_size': 32, 'dim_x': None, 'dim_y': 10, 'latent_dist_name': 'normal',
                     'latent_dist_params_list': [0, 1]}
        mnist_exp = MNIST_XP(dummy_exp)
        del dummy_exp

    if run_training:
        ######## EXPERIMENT PARAMETERS CONFIGURATION ########
        exp_id = 30
        gpu_id = 0

        ## Model architecture parameters group
        model_selector =  'transposed_conv' # Select the type of processing for image in decoder,
                                            #'old_architecture' # to use old architecture
                                            # or one of ['transposed_conv', 'linear', 'upsample_conv'] for new architecture
        latent_space_dimension = 3
        spectral_norm_decoder = True

        ## Loss parameters group
        p = 2
        batch_size = 200
        epochs =  100

        latent_dist_name = 'uniform' # 'normal' #
        latent_dist_params_list = [-3.0, 3.0] # [0.0, 1.0] #

        sinkhorn_epsilon = 100
        sinkhorn_regularization_starting_value = 150 # 500 #300
        sinkhorn_regularization_scaling_factor = 1  # 1 means no scaling. 2 (means divide by 2).
                                                    # -1 means equilibrate with the reconstruction
                                                    # loss dynamically (recon_loss // sink_loss)
        inflate_recon_y = False  # whether to inflate the recon_y loss by recon_x // recon_y
        inflation_param_start = 2.0 # starting weight for inflation of recon_y loss; only used if inflate_recon_y is True
        ## Training parameters group
        lr_scheduler = None #'on_plateau' # 'one_cycle'#    # learning rate scheduling
        lr_scheduler_patience = 5 #30, 10, 0 # patience for on_plateau scheduler
        lr_scheduler_factor = 0.5 #0.9 #0.1 # factor for on_plateau scheduler

        ## provide an epoch number to import checkpoint from; if set to None, no checkpoint is imported and a new model is trained
        checkpoint_epoch = None
        ##########################################################

        nn_params = {
            "latent_dim": latent_space_dimension,
            "netG_apply_spectral_norm": spectral_norm_decoder,
            "netG_constrained_prelu": True if spectral_norm_decoder else False,
            "verbose": False, # to print model architecture and initialization details
        }

        sinkhorn_params = {"p": p, "epsilon": sinkhorn_epsilon, "niter": 100}
        sinkhorn_lambda_scheduling_params = {
            "sink_lambda": sinkhorn_regularization_starting_value,
            # initial value of lambda, multiplying the sinkhorn part of the loss
            "sink_lambda_scheduler_factor": sinkhorn_regularization_scaling_factor,
            "sink_lambda_scheduler_epoch": 100,  # number of epochs after which to update lambda
        }

        model_training_params = {
            "batch_size": batch_size,
            "nb_epochs": epochs,
            "training_stop_metric_threshold": 1e-3,
            "inflate_recon_y": inflate_recon_y,
            "inflation_param_start":inflation_param_start
        }

        norms_params = {
            "l_norm_p_x": p,
            "l_norm_p_y": p,
            "norm_fct_type_x": "lpp",
            "norm_fct_type_y": "lpp"
        }

        ######## SETUP EXPERIMENT ########
        from pathlib import Path
        root_directory = Path(__file__).parent.parent.parent.parent

        seed = 42
        image_size = 32  # images will be resized to image_size x image_size
        number_of_classes = 10  # number of classes in the dataset (MNIST has 10 classes: digits 0-9) => length of one-hot encoded label vector

        xp_config = {
            'data_dir': f'{root_directory}/data',
            'experiment_rootdir': '/media/dl-rookie/Data/Final_thesis_results/Data/Conditional_Generation/MNIST',
            'name': f'MNIST_XP_id_{exp_id}_ImgDecoder_{model_selector}',
            'seed': seed,
            'gpu_id': gpu_id,
            'image_size': image_size,
            'dim_x': image_size * image_size,
            'dim_y': number_of_classes,
            'latent_dist_name': latent_dist_name,
            'latent_dist_params_list': latent_dist_params_list,
        }

        mnist_exp = MNIST_XP(xp_config)

        mnist_exp.load_data(type=['train', 'test'], batch_size=model_training_params['batch_size'],
                            run_data_inspection=False, num_batch_to_inspect=5, grid_size=5, random_samples=False)

        lr_scheduler_verbosity = True
        lr = 0.001
        betas = (0.9, 0.999)

        if lr_scheduler is not None:
            if lr_scheduler == "on_plateau":
                optim_params = {
                    "lr": lr,
                    "betas": betas,
                    "lr_scheduler": lr_scheduler,
                    "lr_factor": lr_scheduler_factor,
                    "lr_patience": lr_scheduler_patience,
                    "lr_threshold": 1e-4,
                    "lr_threshold_mode": "abs",
                    "lr_eps": 1e-10,
                    "verbose": lr_scheduler_verbosity,
                }
            else:
                optim_params = { #onecycle scheduler
                    "lr": lr,
                    "betas": betas,
                    "lr_scheduler": lr_scheduler,
                    "max_lr": lr,
                    "epochs": epochs,
                    "steps_per_epoch": mnist_exp.train_size // batch_size,
                    "verbose": False,
                }
        else:
            optim_params = {
                "lr": lr,
                "betas": betas,
            }

        lr_scheduling = True if lr_scheduler is not None else False

        from_checkpoint = f"{mnist_exp.model_training_dir}/models/checkpoint_epoch_{checkpoint_epoch}.pth" if checkpoint_epoch is not None else None

        model_log_ref = f"MNIST_{model_selector}_{mnist_exp.run_id}"


        print("Training model ...")
        mnist_exp.config_tensorboard_logging()
        mnist_exp.train_model(model_selector=model_selector,
                                 nn_params=nn_params, lr_scheduling=lr_scheduling,
                                 optim_params=optim_params,
                                 sinkhorn_params=sinkhorn_params,
                                 sinkhorn_lambda_scheduling_params=sinkhorn_lambda_scheduling_params,
                                 model_training_params=model_training_params,
                                 norms_params=norms_params,
                                 model_log_ref=model_log_ref,
                                 continue_training_from_checkpoint=from_checkpoint)

        mnist_exp.save_experiment()


    #####################Diagnostics######################
    if run_jsae_diags:
        # if not training, run diagnostics on a pre-trained model
        import fastabc_inversion.conditional_generation.diagnostics as diag
        from fastabc_inversion.conditional_generation.mnist.classifier.model import mnist

        print("Running diagnostics on pre-trained model...")

        print("Loading experiment from file...")

        # load pre-trained model from checkpoint
        mnist_exp = load_experiment_from_file(experiment_file_location)

        # create diagnostics object
        diags_mnist_exp = diag.Diagnostics(mnist_exp, epoch=epoch)

        # load pretrained classifier (example: MNIST classifier)
        classifier = mnist(pretrained="./classifier/PretrainedClassifier.pth").to(mnist_exp.device)
        diags_mnist_exp.load_classifier(classifier, append_softmax=True)

        """
        ## Evaluate classifier accuracy on test set.
        diags_mnist_exp.classifier.eval()
        accuracy = 0
        for batch_idx, (data, target) in enumerate(mnist_exp.test_loader):
            data, target = data.to(mnist_exp.device), target.to(mnist_exp.device)
            #target_trans_test = mnist_exp.label_transform.simplex_vec_to_label(target)
            # revert target to label from one-hot encoding
            target = target.argmax(dim=1)
            output = diags_mnist_exp.classifier(data)
            pred = output.argmax(dim=1, keepdim=True)
             # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy += correct
            print(f"Batch {batch_idx}: Correct predictions: {correct} out of {data.size(0)}")
        total_samples = len(mnist_exp.test_loader.dataset)
        print(f"Overall accuracy on test set: {accuracy} out of {total_samples} = {accuracy / total_samples:.4f}")
        # Overall accuracy on test set: 9770 out of 10000 = 0.977
        # Careful not to assess accuracy on val_loader, it is a subset of the classifiers training data!
        """

        diag_start_time = time.time()
        diags_mnist_exp.run_diagnostics(sample_size = diags_mnist_exp.experiment.val_size, diagnostics_to_run=diagnostics_to_run, diags_params=diags_params,
                                          logging_comment=f"{diags_mnist_exp.experiment.name}_epoch{epoch}")
        diag_end_time = time.time()
        print(f"Time to run diagnostics: {diag_end_time - diag_start_time:.3f} s")
    #######################################################

    if run_inference or run_infernce_diags:
        import fastabc_inversion.conditional_generation.sus_inference as inference
        import fastabc_inversion.conditional_generation.utils.plotting as plot

        print("Loading experiment from file...")

        # load pre-trained model from checkpoint
        mnist_exp = load_experiment_from_file(experiment_file_location)

        # align torch_clr_transform and label_transform delta
        mnist_exp.label_transform.delta = mnist_exp.torch_clr_transformer.epsilon

        if run_inference:
            print("Running inference ...")
            inference.run_sus_inference_all_observations(mnist_exp, observation_vec, inference_params)

        if run_infernce_diags:

            print("Running inference diagnostics ...")

            # classify samples and plot distributions of predicted labels at each intermediate threshold
            # load pretrained classifier (example: MNIST classifier)
            import fastabc_inversion.conditional_generation.diagnostics as diag
            from fastabc_inversion.conditional_generation.mnist.classifier.model import mnist
            from fastabc_inversion.conditional_generation.mnist.morphology.measure import distribution_measure

            classifier = mnist(pretrained="./classifier/PretrainedClassifier.pth").to(mnist_exp.device)
            diags_mnist_exp = diag.Diagnostics(mnist_exp, epoch=epoch)
            diags_mnist_exp.load_classifier(classifier, append_softmax=True)
            classifier = diags_mnist_exp.classifier

            del diags_mnist_exp

            # check if attribute all_obs_inference_results exists in mnist_exp, if not create it and
            # read inference results from files
            if not hasattr(mnist_exp, 'all_obs_inference_results'):
                mnist_exp.all_obs_inference_results = None
                inference.read_sus_inference_results_from_files(mnist_exp, observation_vec)

            for obs in observation_vec:
                dict_key = tuple(obs) if isinstance(obs, list) else obs
                eps = inference_params['epsilon_vec'][0] # assuming one epsilon level TODO: generalize
                obs_all_latent_samples = mnist_exp.all_obs_inference_results[dict_key][eps]['samples_per_thresh'][0] # assuming sus_runs=1 TODO: generalize

                # Get intermediate thresholds from inference results
                intermediate_thresholds = mnist_exp.all_obs_inference_results[dict_key][eps]['all_thresholds'][0]  # assuming sus_runs=1 TODO: generalize

                if inference_params['return_full_results']:
                    # add a large threshold value at index 0
                    intermediate_thresholds = np.insert(intermediate_thresholds, 0, np.inf)

                # Round thresholds to 3 decimal places for labels
                threshold_labels = [f"{thresh:.3f}" for thresh in intermediate_thresholds]

                all_labels = []
                all_values = []
                all_classes = []

                concordance_values = {}

                for thresh_idx, (threshold, latent_samples) in enumerate(
                        zip(intermediate_thresholds, obs_all_latent_samples)):
                    inverted_latent = torch.tensor(latent_samples, dtype=torch.float32)
                    inverted_x, inverted_y_label = inference.get_inverted_x_y(mnist_exp, inverted_latent,
                                                                              return_labels=True)

                    classified_inverted_x = classifier(inverted_x.to(mnist_exp.device))
                    classified_inverted_x_label = mnist_exp.label_transform.simplex_vec_to_label(
                        classified_inverted_x.cpu())

                    # Compute concordance (proportion of matching labels)
                    concordance = (inverted_y_label == classified_inverted_x_label).float().mean().item()
                    concordance_values[threshold] = concordance

                    indices = torch.randperm(inverted_x.size(0))[:num_samples_to_plot]

                    # at each intermediate level
                    """
                    # 1.randomly select num_samples_to_plot to plot from each observation set

                    images_to_plot = inverted_x[indices, :, :, :].squeeze().numpy()
                    labels_to_plot = inverted_y_label[indices].numpy()

                    save_fig_path = f"{mnist_exp.experiment_dir}/inference_results/label_{obs}/inverted_samples_observation_epsilon_level{thresh_idx}.pdf"

                    plot.plot_samples_grid(images=images_to_plot, labels=labels_to_plot,
                                           num_samples=num_samples_to_plot, save_location=save_fig_path)

                    """
                    # 2.
                    if inference_params['norm_fct'] in morphological_fn:
                        # measure morphological transformation of generated samples

                        inverted_x = inverted_x.squeeze()  # shape (N, H, W)
                        inverted_x = inverted_x * 0.5 + 0.5  # rescale to [0, 1] from [-1, 1]

                        key = inference_params['norm_fct']
                        measurements_df = distribution_measure(inverted_x, inverted_y_label)

                        measurements = measurements_df[key]
                        labels_from_measure = measurements_df['label']  # Get the label column

                        threshold_label = thresh_idx

                        all_labels.extend([threshold_label] * len(measurements))
                        all_values.extend(measurements)
                        all_classes.extend(labels_from_measure.tolist())

                    else:

                        threshold_label = thresh_idx

                        all_labels.extend([threshold_label] * len(inverted_y_label))
                        all_values.extend(inverted_y_label.tolist())

                # Save concordance results to text file
                concordance_save_path = f"{mnist_exp.inference_dir}/label_{obs}/concordance_per_threshold.txt"
                with open(concordance_save_path, 'w') as f:
                    f.write("threshold\tconcordance\n")
                    for threshold, concordance in concordance_values.items():
                        f.write(f"{threshold}\t{concordance}\n")
                print(f"Concordance data saved to {concordance_save_path}")

                df_dict = {
                    'labels': all_labels,
                    'values': all_values
                }

                # Add class column if morphological function was used
                if inference_params['norm_fct'] in morphological_fn:
                    df_dict['class'] = all_classes

                df = pd.DataFrame(df_dict)

                legend = None if inference_params['norm_fct'] not in morphological_fn else ['Measured']
                morpho_ticks = {'slant': [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
                                'thickness': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                'length': [14, 24, 34, 44, 54, 64, 74, 84, 94, 100],}
                yticks = range(10) if inference_params['norm_fct'] not in morphological_fn else morpho_ticks.get(inference_params['norm_fct'], None)

                plot_save_path = f"{mnist_exp.inference_dir}/label_{obs}/all_thresholds_summaries_boxplots.pdf"
                if inference_params['norm_fct'] in morphological_fn:
                    plot.plot_boxplot_with_stripplot(df, value_labels= legend, yticks=yticks, reverse_labels=True,
                                        title=f"Target = {obs} - Plotted {num_samples_to_plot} from {inference_params['N']}",
                                        save_location=plot_save_path)
                else:
                    plot.plot_class_proportions_stacked(
                        df,
                        threshold_limit=4,
                        title=f"Target = {obs}",
                        save_location=plot_save_path
                    )























