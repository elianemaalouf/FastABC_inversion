"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
This file allows to configure and run the joint Generative Neural Network with ABC by SubSet Simulation
experiment for the Geophysics problems.

Models training :
ressources : http://karpathy.github.io/2019/04/25/recipe/ ;
               https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/

Logs to tensorboard. Start logger from virtual environment terminal and specify the logging directory:
(https://pytorch.org/docs/stable/tensorboard.html)
$ tensorboard --logdir /home/.../.../.../runs
"""
import inspect
import os
import pickle
import time

import fastabc_inversion.geo_problems.utils.sinkhorn.sinkhorn_pointcloud as spc
import fastabc_inversion.geo_problems.utils.torch_data_prep as tdp
import fastabc_inversion.geo_problems.utils.torch_distances as td
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastabc_inversion.geo_problems.baseExperiment import BaseExperiment
from fastabc_inversion.geo_problems.utils.config import Config
from fastabc_inversion.geo_problems.utils.visualization import \
    plotting_tools as plot
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


def load_experiment_from_file(experiment_checkpoint_file):
    """
    Load the experiment from file.
    :param experiment_checkpoint_file: path to the experiment checkpoint file
    :return: the experiment object
    """
    with open(experiment_checkpoint_file, "rb") as f:
        experiment = pickle.load(f)
    return experiment


def norm_fn_selector(type="lpp"):
    """
    Select the function based on the type
    :param type: string indicating which function to use. Options are 'mse', 'rmse', 'lpp' (=lp^p), 'lp'.
    Default is 'lpp'.
    :return: the function to use
    """
    _map = {
        "mse": td.mse_torch,
        "rmse": td.rmse_torch,
        "lpp": td.lpp_torch,  # lp^p
        "lp": td.lp_torch,
    }

    return _map[type]


def params_init(m, args=("relu", None), verbose=False):
    """
    Function to randomly initialize the weights in the NN based on Xavier initialization.
    BatchNorm layers are initialized with default values via the reset_parameters() function.
    :param m: the layer for which to initialize the weights
    :param args: a tuple of 3 elements to configure the initialization function.
    The tuple contains:
        - first element: type of gain to be used : 'ReLU' 'Leaky Relu' etc.
        If not None, this is the first parameter of the second parameter of torch.nn.init.calculate_gain(nonlinearity, param=None)
        if None, then the default is maintained.
        - second element: the value to use for the bias. Leave it to None if the value should be 0 by default.

    Check for type of gain : https://pytorch.org/docs/2.1/nn.init.html#torch-nn-init
    :param verbose: whether to print to console the values of the weights, before and after initialization (for debugging)
    :return:
    """
    fn_params = args
    _name_map = {
        "relu": "relu",
        "leaky_relu": "leaky_relu",
        "prelu": "leaky_relu",
        "constrained_prelu": "leaky_relu",
        "swish": "swish",
        "silu": "swish",
        "beta_swish": "swish",
        "constrained_beta_swish": "swish",
    }
    name_map = _name_map[fn_params[0]]
    if name_map in ["relu", "leaky_relu"]:
        param = 0.25 if _name_map[fn_params[0]] == "leaky_relu" else None
        gain = torch.nn.init.calculate_gain(name_map, param=param)
    elif name_map in ["swish"]:
        gain = 1.75
    else:
        raise NotImplementedError("Unknown activation function {}".format(fn_params[0]))

    if isinstance(
        m,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Linear,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        if verbose:  # before initialization
            print("Before init")
            print("m.weight:", m.weight)
            print("m.bias:", m.bias)

        if fn_params[0] is None:
            torch.nn.init.xavier_uniform_(m.weight)
        else:
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            if fn_params[1] is not None:
                torch.nn.init.constant_(m.bias, fn_params[1])
            else:
                torch.nn.init.zeros_(m.bias)

        if verbose:  # after initialization
            print("After init")
            print("m.weight:", m.weight)
            print("m.bias:", m.bias)

    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.reset_parameters()


class jGNN_SuS_exp(BaseExperiment):
    _map_noise_to_dict_idx = {
        "small_gauss": 0,
        "large_gauss": 1,
        "gumbel": 2,
    }

    def __init__(
        self,
        parameters_file,
        name,
        seed=None,
        run_id=None,
        use_cuda=True,
        gpu_id=0,
        latent_dist_name="standardnormal",
        latent_dist_params_list=[0, 1],
        sinkhorn_params=None,
        model_training_params=None,
        inference_params=None,
        name_suffix="",
        noise=None,
    ):
        """
        Constructor for the Geo jGNN experiment.
        :param parameters_file: mandatory parameter to specify the path to the configuration file.
        :param name: name of the experiment
        :param use_cuda: whether to use gpu or cpu
        :param gpu_id: id of the gpu to use if use_cuda is True
        :param sinkhorn_params: dictionary of sinkhorn parameters.
        The dictionary contains:
            - epsilon: epsilon value for the sinkhorn algorithm
            - niter: number of iterations for the sinkhorn algorithm
            - p: p value for the sinkhorn algorithm
        :param model_training_params: dictionary of model training parameters.
        The dictionary contains:
            - batch_size: batch size for training
            - nb_epochs: number of epochs for training
            - best_model_metric: metric to use to select the best model, either 'loss' (whole loss on validation)
            or 'recon' (only reconstruction loss on validation)
            - training_stop_metric_threshold: threshold for the training stopping metric. If the metric is below this
        threshold, training stops.
        :param inference_params: parameters for the inference
        :param name_suffix: suffix to add to the experiment name. Start with '_'.
        :param noise : deprecated parameter, use noise_dicts instead.
        """

        self.config = Config(parameters_file)
        data_rootdir = self.config.datadir + "/Geo_jGNN_SuS_experiments"
        os.makedirs(data_rootdir, exist_ok=True)

        super().__init__(
            name=f"{name}{name_suffix}",
            run_id=run_id,
            data_rootdir=data_rootdir,
            seed=self.config.parameters["FixedSeed"] if seed is None else seed,
            latent_dist_name="standardnormal"
            if latent_dist_name is None
            else latent_dist_name,
            latent_dist_params_list=[0, 1]
            if latent_dist_params_list is None
            else latent_dist_params_list,
            sinkhorn_params={"p": 2, "epsilon": 100, "niter": 40}
            if sinkhorn_params is None
            else sinkhorn_params,
            model_training_params={
                "batch_size": 128,
                "nb_epochs": 100,
                "best_model_metric": "recon",
                "training_stop_metric_threshold": 1e-3,
            }
            if model_training_params is None
            else model_training_params,
            inference_params=inference_params,
            noise_dicts={},
        )
        # self.noise_label = noise # deprecated

        if self.device is None:
            self.device = torch.device(
                f"cuda:{gpu_id}" if (torch.cuda.is_available() and use_cuda) else "cpu"
            )
            self.ngpu = 1 if torch.cuda.is_available() and use_cuda else 0

        self.tensorboard_logging_dir = None
        self.tsb_logger = None

        self.nx = self.config.nx
        self.ny = self.config.ny
        self.dim_x = self.nx * self.ny
        self.dim_y = self.config.ndata

        self.training_x = None
        self.training_y = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.x_lower = None
        self.x_upper = None
        self.y_lower = None
        self.y_upper = None
        self.data_prep = None
        self.normalize = None

        self.validation_x = None
        self.validation_y = None

        self.test_x = None
        self.test_y = None

        self.optimizer = None

        self.train_history = None
        self.total_training_epochs = 0

        # add two extra directories
        os.makedirs(f"{self.model_training_dir}/reconstruction_val", exist_ok=True)
        os.makedirs(f"{self.model_training_dir}/reconstruction_train", exist_ok=True)

    def config_tensorboard_logging(self, tensorboard_logging_dir_root):
        """
        Configure tensorboard logging.
        :param tensorboard_logging_dir_root: root directory for tensorboard logging
        """
        self.tensorboard_logging_dir = f"{tensorboard_logging_dir_root}/{self.name}"
        os.makedirs(self.tensorboard_logging_dir, exist_ok=True)
        self.tsb_logger = SummaryWriter(log_dir=self.tensorboard_logging_dir)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if "tsb_logger" in state:
            del state["tsb_logger"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., `self.__dict__`).
        self.__dict__.update(state)
        # Recreate the unpicklable entries.
        self.tsb_logger = None  # or reinitialize it if needed

    def load_data(
        self, datasets_to_load=["train", "validation"], data_prep=None, **kwargs
    ):
        """
        Load data from files.
        :param datasets_to_load: a list containing the dataset to load.
        The list can contain one or more of 'train', 'validation', 'test'.
        :param data_prep: a dictionary containing the data preparation parameters.
        :param kwargs: additional arguments such as subset_train
        """
        for selector in datasets_to_load:
            if selector == "train":
                train_models_file = h5py.File(
                    self.config.data_folder_location + "/train_models.h5"
                )
                self.training_x = torch.tensor(
                    train_models_file.get("train_models"), dtype=torch.float32
                )
                train_truett_file = h5py.File(
                    self.config.data_folder_location + "/train_truett.h5"
                )
                self.training_y = torch.tensor(
                    train_truett_file.get("train_truett"), dtype=torch.float32
                )
                train_truett_file.close()

                self.train_size = self.training_x.shape[0]

                subset_train = kwargs.get("subset_train")
                subset_train = self.train_size if subset_train is None else subset_train

                if subset_train != self.train_size:
                    # subset training set - randomly
                    train_subset_idx = np.random.randint(
                        self.train_size, size=subset_train
                    )
                    with open(
                        self.model_training_dir + "/training_subset_idx", "wb"
                    ) as f:
                        pickle.dump(
                            train_subset_idx, f
                        )  # save the ids of the examples that were used in training
                    self.training_x = self.training_x[train_subset_idx, :, :, :]
                    self.training_y = self.training_y[train_subset_idx, :]

                    print("Subset training data loaded")
                    print("Training models shape:", self.training_x.shape)
                    print("Training travel times shape:", self.training_y.shape)
                else:
                    print("Training data loaded")
                    print("Training models shape:", self.training_x.shape)
                    print("Training travel times shape:", self.training_y.shape)

                self.train_size = subset_train

                self.x_mean = self.training_x.view(self.train_size, self.dim_x).mean(
                    axis=0
                )
                self.x_std = self.training_x.view(self.train_size, self.dim_x).std(
                    axis=0
                )
                self.y_mean = self.training_y.view(self.train_size, self.dim_y).mean(
                    axis=0
                )
                self.y_std = self.training_y.view(self.train_size, self.dim_y).std(
                    axis=0
                )

                if data_prep is not None:
                    self.normalize = True
                    self.data_prep = data_prep  # store original configuration

                    if data_prep["function"] == "standardize":
                        print("Standardizing inputs")
                        # Standardize inputs - substract mean and divide by standard deviation (component wise)
                        # self.training_x = (self.training_x.view(self.train_size,self.dim_x) - self.x_mean) / self.x_std
                        # self.training_y = (self.training_y.view(self.train_size,self.dim_y) - self.y_mean) / self.y_std
                        self.training_x = tdp.standardize(
                            self.training_x.view(self.train_size, self.dim_x),
                            self.x_mean,
                            self.x_std,
                        )
                        self.training_y = tdp.standardize(
                            self.training_y.view(self.train_size, self.dim_y),
                            self.y_mean,
                            self.y_std,
                        )
                        self.normalization_dict_x = {
                            "function": data_prep["function"],
                            "kwargs": {"mean": self.x_mean, "std": self.x_std},
                        }
                        self.normalization_dict_y = {
                            "function": data_prep["function"],
                            "kwargs": {"mean": self.y_mean, "std": self.y_std},
                        }

                    if data_prep["function"] == "min_max":
                        print("Min-max normalizing inputs")
                        (
                            self.training_x,
                            self.x_lower,
                            self.x_upper,
                        ) = tdp.min_max_normalize(
                            self.training_x.view(self.train_size, self.dim_x),
                            **data_prep["kwargs"],
                        )
                        (
                            self.training_y,
                            self.y_lower,
                            self.y_upper,
                        ) = tdp.min_max_normalize(
                            self.training_y.view(self.train_size, self.dim_y),
                            **data_prep["kwargs"],
                        )
                        self.normalization_dict_x = {
                            "function": data_prep["function"],
                            "kwargs": {
                                "lower_value": self.x_lower,
                                "upper_value": self.x_upper,
                            },
                        }
                        self.normalization_dict_y = {
                            "function": data_prep["function"],
                            "kwargs": {
                                "lower_value": self.y_lower,
                                "upper_value": self.y_upper,
                            },
                        }

            elif selector == "validation":
                val_models_file = h5py.File(
                    self.config.data_folder_location + "/val_models.h5"
                )
                self.validation_x = torch.tensor(
                    val_models_file.get("val_models"), dtype=torch.float32
                )
                val_models_file.close()
                val_truett_file = h5py.File(
                    self.config.data_folder_location + "/val_truett.h5"
                )
                self.validation_y = torch.tensor(
                    val_truett_file.get("val_truett"), dtype=torch.float32
                )
                val_truett_file.close()

                self.val_size = self.validation_x.shape[0]

                print("Validation data loaded")
                print("Validation models shape:", self.validation_x.shape)
                print("Validation travel times shape:", self.validation_y.shape)

                if data_prep is not None:
                    if data_prep["function"] == "standardize":
                        if (
                            self.x_mean is None
                            or self.x_std is None
                            or self.y_mean is None
                            or self.y_std is None
                        ):
                            raise ValueError(
                                "Training data must be loaded before validation data"
                            )
                        # self.validation_x = (self.validation_x.view(self.val_size,self.dim_x) - self.x_mean) / self.x_std
                        # self.validation_y = (self.validation_y.view(self.val_size,self.dim_y) - self.y_mean) / self.y_std
                        self.validation_x = tdp.standardize(
                            self.validation_x.view(self.val_size, self.dim_x),
                            self.x_mean,
                            self.x_std,
                        )
                        self.validation_y = tdp.standardize(
                            self.validation_y.view(self.val_size, self.dim_y),
                            self.y_mean,
                            self.y_std,
                        )
                    if data_prep["function"] == "min_max":
                        if (
                            self.x_lower is None
                            or self.x_upper is None
                            or self.y_lower is None
                            or self.y_upper is None
                        ):
                            raise ValueError(
                                "Training data must be loaded before validation data"
                            )
                        self.validation_x, _, _ = tdp.min_max_normalize(
                            self.validation_x.view(self.val_size, self.dim_x),
                            lower_value=self.x_lower,
                            upper_value=self.x_upper,
                        )
                        self.validation_y, _, _ = tdp.min_max_normalize(
                            self.validation_y.view(self.val_size, self.dim_y),
                            lower_value=self.y_lower,
                            upper_value=self.y_upper,
                        )

            elif selector == "test":
                test_models_file = h5py.File(
                    self.config.data_folder_location + "/test_models.h5"
                )
                self.test_x = torch.tensor(
                    test_models_file.get("test_models"), dtype=torch.float32
                )
                test_models_file.close()
                test_truett_file = h5py.File(
                    self.config.data_folder_location + "/test_truett_noNoise.h5"
                )
                self.test_y = torch.tensor(
                    test_truett_file.get("test_truett_noNoise"), dtype=torch.float32
                )
                test_truett_file.close()

                self.test_size = self.test_x.shape[0]

                print("Reference test data loaded")
                print("Test models shape:", self.test_x.shape)
                print("Test travel times shape:", self.test_y.shape)

                if data_prep is not None:
                    print("Not normalizing test data")
            else:
                raise ValueError(
                    "selector must be one of the following: train, validation, test"
                )

    def get_observation(self, observation_idx, noise_label):
        """
        Load observation from file.
        :param observation_path: path to the observation file
        """
        print(f"Loading noisy observation {observation_idx} from file...")
        noise_distribution = self.noise_dicts[noise_label]["distribution"]
        noise_loc = self.noise_dicts[noise_label]["location"]
        noise_scale = self.noise_dicts[noise_label]["scale"]

        observation_path = f"{self.config.data_folder_location}/noisy_ttvec_{noise_distribution}_loc{noise_loc}_scale{str(noise_scale).replace('.', 'p')}/noisy_tt_vec{observation_idx}"

        with open(observation_path, "rb") as f:
            y_obs = pickle.load(f)
        return y_obs

    def update_noise_dicts(self, noise_list):
        """
        Update the noise dictionary.
        :param noise_list: a list of noise labels to update the noise dictionary with.
        """
        for noise_label in noise_list:
            self.noise_dicts[noise_label] = self.config.noises_list[
                self._map_noise_to_dict_idx[noise_label]
            ]

    def construct_model_architecture(self, model_selector, nn_params):
        """
        Construct the model architecture.
        :param model_selector: name of the model to use
        :param nn_params: dictionary of extra neural network parameters
        """
        print(f"Constructing {model_selector} model architecture ...")

        def toggle_spectral_norm(model, use_spectral_norm):
            for module in model.modules():
                # Check if the module is a layer where spectral norm can be applied
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    if use_spectral_norm:
                        # Apply spectral normalization if not already applied
                        if not hasattr(module, "weight_orig"):
                            spectral_norm(module)
                    else:
                        pass
                        # Remove spectral normalization if it was applied
                        # if hasattr(module, 'weight_orig'):
                        #    remove_spectral_norm(module)

        def print_spectral_norm_status(model):
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    status = (
                        "Spectral Norm applied"
                        if hasattr(module, "weight_orig")
                        else "No Spectral Norm"
                    )
                    print(f"{name}: {status}")

        _map_module_init_param = {
            "netD": (nn_params["activation_dict_encoder"]["name"], None, True),
            "netG": (nn_params["activation_dict_decoder"]["name"], None, True),
            # "netD": ("relu", None, True), #
            # "netG": ("leaky_relu", None, True),
        }

        if model_selector == "jUnet":
            import fastabc_inversion.geo_problems.nn.jUnet as nnm
        else:
            raise ValueError("please select one of the models in the nn folder")

        self.input_dim = self.dim_x + self.dim_y
        self.output_dim = self.input_dim
        self.latent_dim = nn_params["latent_dim"]
        self.nn_params = nn_params
        spectral_norm_encoder = nn_params["spectral_norm_encoder"]
        spectral_norm_decoder = nn_params["spectral_norm_decoder"]

        netD_params = {
            key: value
            for key, value in nn_params.items()
            if key in inspect.signature(nnm.netD.__init__).parameters
        }
        netG_params = {
            key: value
            for key, value in nn_params.items()
            if key in inspect.signature(nnm.netG.__init__).parameters
        }

        # build neural networks and initialize weights
        self.netD = nnm.netD(
            ngpu=self.ngpu,
            ndata=self.dim_y,
            height=self.ny,
            width=self.nx,
            **netD_params,
        )  # encoder
        toggle_spectral_norm(self.netD, spectral_norm_encoder)
        print(
            f"Spectral norm on encoder set to : {spectral_norm_encoder}; verifying encoder layers:"
        )
        print_spectral_norm_status(self.netD)

        self.netG = nnm.netG(
            ngpu=self.ngpu,
            ndata=self.dim_y,
            height=self.ny,
            width=self.nx,
            **netG_params,
        )  # decoder
        toggle_spectral_norm(self.netG, spectral_norm_decoder)
        print(
            f"Spectral norm on decoder set to : {spectral_norm_decoder}; verifying decoder layers:"
        )
        print_spectral_norm_status(self.netG)

        self.model = nnm.netWae(
            encoder=self.netD, decoder=self.netG, ngpu=self.ngpu
        )  # encoder - decoder

        self.save_model_summary(
            input_dim=(self.model_training_params["batch_size"], self.input_dim)
        )
        self.model.weight_init(
            params_init, _map_module_init_param["netD"], _map_module_init_param["netG"]
        )

    def load_model_weights(self, model_weights_path):
        """
        Load the model weights from file.
        :param model_weights_path: dictionary with two keys:
            - 'netD': netD state dict
            - 'netG': netG state dict
        """
        print("Loading pretrained model ...")
        self.pretrained_model_netD_state_dict = model_weights_path["netD"]
        self.pretrained_model_netG_state_dict = model_weights_path["netG"]

        self.model.netD.load_state_dict(self.pretrained_model_netD_state_dict)
        self.model.netG.load_state_dict(self.pretrained_model_netG_state_dict)

    def set_sinkhorn_params(
        self, sinkhorn_params, sinkhorn_lambda_scheduling_params, override=False
    ):
        """
        Set the sinkhorn parameters.
        :param sinkhorn_params: dictionary of sinkhorn parameters.
        :param sinkhorn_lambda_scheduling_params: dictionary of lambda scheduling parameters for the sinkhorn part of the loss.
        :param override: whether to override the existing parameters
        """
        if sinkhorn_params is not None:
            if self.sinkhorn_params is None or override:
                self.sinkhorn_params = sinkhorn_params

        if sinkhorn_lambda_scheduling_params is not None:
            if self.sinkhorn_lambda_scheduling_params is None or override:
                self.sinkhorn_lambda_scheduling_params = (
                    sinkhorn_lambda_scheduling_params
                )

    def prep_model(
        self,
        model_selector,
        nn_params,
        load_pretrained_model,
        transfer_netG_state_dict,
        transfer_netD_state_dict,
        model_log_ref,
    ):
        """
        Set the neural network and its training configurations.
        :param model_selector: name of the model to use
        :param nn_params: dictionary of extra neural network parameters
        :param load_pretrained_model: whether to load a pretrained model
        :param transfer_netG_state_dict: the netG state dict of the pretrained model
        :param transfer_netD_state_dict: the netD state dict of the pretrained model
        :param model_log_ref: short reference to be used for logging model performance in log file
        """
        self.model_log_ref = model_log_ref
        self.model_selector = model_selector

        self.construct_model_architecture(model_selector, nn_params)

        self.load_pretrained_model = load_pretrained_model

        if self.load_pretrained_model:
            self.load_model_weights(
                {"netD": transfer_netD_state_dict, "netG": transfer_netG_state_dict}
            )

        self.model.to(self.device)
        print(f"Model was created and sent to device {self.device}...")

    def prep_model_optimizer(self, lr_scheduling, optim_params, override=False):
        """
        Configure the model optimizer. Uses Adam optimizer.
        :param lr_scheduling: whether to use learning rate scheduling
        :param optim_params: dictionary of optimizer parameters
        :param override: whether to override the existing optimizer if it is already set
        :return:
        """
        _AVAILABLE_LR_SCHEDULERS = ["on_plateau", "one_cycle"]
        print("Preparing model Adam optimizer ...")
        self.lr_scheduling = lr_scheduling

        if optim_params is not None:
            if self.optimizer is None or override:
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=optim_params["lr"],
                    betas=optim_params["betas"],
                )
                self.optim_params = optim_params
            if lr_scheduling:
                if optim_params["lr_scheduler"] == "on_plateau":
                    self.optim_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
                        optimizer=self.optimizer,
                        mode="min",
                        factor=optim_params["lr_factor"],
                        threshold_mode=optim_params["lr_threshold_mode"],
                        eps=optim_params["lr_eps"],
                        patience=optim_params["lr_patience"],
                        verbose=optim_params["verbose"],
                        threshold=optim_params["lr_threshold"],
                    )
                elif optim_params["lr_scheduler"] == "one_cycle":
                    self.optim_lr_scheduler = lr_scheduler.OneCycleLR(
                        optimizer=self.optimizer,
                        max_lr=optim_params["lr"],
                        epochs=optim_params["epochs"],
                        steps_per_epoch=optim_params["steps_per_epoch"],
                        verbose=optim_params["verbose"],
                    )
                else:
                    raise ValueError(
                        f"Invalid learning rate scheduler: {optim_params['lr_scheduler']}. Choose"
                        f"one of {_AVAILABLE_LR_SCHEDULERS}."
                    )
        else:
            raise ValueError(
                "Optimizer parameters are not provided. "
                "Provide at least optim_params['lr'] parameter."
            )

    def save_checkpoint(self, epoch, save_full_experiment=False):
        """
        Save a checkpoint of the model & training information.
        """
        checkpoint = {
            "epoch": epoch,
            "model_selector": self.model_selector,
            "model_log_ref": self.model_log_ref,
            "nn_params": self.nn_params,
            "netD_state_dict": self.model.netD.state_dict(),
            "netG_state_dict": self.model.netG.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optim_params": self.optim_params,
            "lr_scheduling": self.lr_scheduling,
            "total_training_epochs": self.total_training_epochs,
            "train_history": self.train_history,
            "sinkhorn_params": self.sinkhorn_params,
            "sinkhorn_lambda_scheduling_params": self.sinkhorn_lambda_scheduling_params,
            "model_training_params": self.model_training_params,
            "norms_params": self.norms_params,
        }
        torch.save(
            checkpoint, f"{self.model_training_dir}/models/checkpoint_epoch_{epoch}.pth"
        )

        if save_full_experiment:
            with open(
                f"{self.model_training_dir}/models/experiment_epoch_{epoch}.pkl", "wb"
            ) as f:
                pickle.dump(self, f)

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint of the model & training information.
        """
        checkpoint = torch.load(checkpoint_path)
        return checkpoint

    def load_experiment(self, checkpoint):
        """
        Load the experiment from a checkpoint.
        """
        # TODO: implement this function

        self.prep_model(
            model_selector=checkpoint["model_selector"],
            nn_params=checkpoint["nn_params"],
            load_pretrained_model=True,
            transfer_netD_state_dict=checkpoint["netD_state_dict"],
            transfer_netG_state_dict=checkpoint["netG_state_dict"],
            model_log_ref=checkpoint["model_log_ref"],
        )

        self.prep_model_optimizer(
            checkpoint["lr_scheduling"], checkpoint["optim_params"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.set_sinkhorn_params(
            checkpoint["sinkhorn_params"],
            checkpoint["sinkhorn_lambda_scheduling_params"],
            override=True,
        )

        self.model_training_params = checkpoint["model_training_params"]

        self.norms_params = checkpoint["norms_params"]

        self.train_history = checkpoint["train_history"]

    def train_model(
        self,
        model_selector,
        nn_params,
        lr_scheduling,
        optim_params,
        sinkhorn_params,
        sinkhorn_lambda_scheduling_params,
        model_training_params,
        norms_params,
        model_log_ref,
        continue_training_from_checkpoint,
    ):
        """
        Train the model.
        Parameters to pass to prep_model():
        :param model_selector: name of the model to use from the nn folder
        :param nn_params: dictionary of extra neural network parameters
        :param lr_scheduling: whether to use learning rate scheduling
        :param optim_params: dictionary of optimizer parameters. Optimizer is Adam.
        The dictionary contains:
            - lr: learning rate
            - betas: Adam betas
            - the rest of entries are configurations for the LR scheduler, if used.
            Two LR schedulers are used: ReduceLROnPlateau or OneCycleLR.
            Parameters for ReduceLROnPlateau:
            https://pytorch.org/docs/2.0/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
                - lr_scheduler: set to "on_plateau" to select ReduceLROnPlateau
                - lr_factor: factor by which to reduce the learning rate
                - lr_patience: number of epochs with no improvement after which learning rate will be reduced
                - lr_threshold: threshold for measuring the new optimum, to only focus on significant changes
                - lr_threshold_mode: one of 'rel', 'abs'.
                - lr_eps: minimal decay applied to lr.
            Parameters for OneCycleLR:
            Reference: https://pytorch.org/docs/2.0/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
                - lr_scheduler: set to "one_cycle" to select OneCycleLR
                - max_lr : Upper learning rate boundaries in the cycle for each parameter group.
                - epochs : The number of epochs to train for.
                - steps_per_epoch : The number of steps per epoch to train for (i.e, training set size // batch size)

        Parameters to set up training loop:
        :param sinkhorn_params: dictionary of sinkhorn parameters.
        The dictionary contains:
            - epsilon: epsilon value for the sinkhorn algorithm
            - niter: number of iterations for the sinkhorn algorithm
            - p: p value for the sinkhorn algorithm
        :param sinkhorn_lambda_scheduling_params: dictionary of lambda scheduling parameters for the sinkhorn part of the loss.
        The dictionary contains :
            - sink_lambda: initial value of lambda
            - sink_lambda_sched_factor: factor by which to divide lambda. If 1, lambda is constant.
            - sink_lambda_scheduler_epoch: number of epochs after which to divide lambda by sink_lambda_sched_factor
        :param model_training_params: dictionary of model training parameters.
        The dictionary contains:
            - batch_size: batch size for training
            - nb_epochs: number of epochs for training
            - best_model_metric: metric to use to select the best model, either 'loss' (whole loss on validation)
            or 'recon' (only reconstruction loss on validation)
            - training_stop_metric_threshold: threshold for the training stopping metric. If the metric is below this
        threshold, training stops.
        :param norms_params: dictionary of parameters for the norm functions.
        The dictionary contains:
            - l_norm_p_x: p value for the norm function on the x part of the data
            - l_norm_p_y: p value for the norm function on the y part of the data
            - norm_fct_type_x: type of norm function to use on the x part of the data e.g. 'mse', 'rmse', 'lp', 'ss'
            - norm_fct_type_y: type of norm function to use on the y part of the data e.g. 'mse', 'rmse', 'lp', 'ss'
        :param model_log_ref: short reference to be used for logging model performance in log file
        :param continue_training_from_checkpoint: whether to continue training from a checkpoint, contains the path to the checkpoint.
        """

        # setups
        if self.model is None:
            self.prep_model(
                model_selector=model_selector,
                nn_params=nn_params,
                load_pretrained_model=False,
                transfer_netG_state_dict=None,
                transfer_netD_state_dict=None,
                model_log_ref=model_log_ref,
            )

        if self.optimizer is None:
            self.prep_model_optimizer(lr_scheduling, optim_params)

        self.set_sinkhorn_params(
            sinkhorn_params, sinkhorn_lambda_scheduling_params, override=True
        )

        if model_training_params is not None:
            self.model_training_params = model_training_params

        if norms_params is not None:
            self.norms_params = norms_params

        epoch_offset = 0

        if continue_training_from_checkpoint is not None:
            # load checkpoint
            checkpoint = self.load_checkpoint(continue_training_from_checkpoint)

            self.load_model_weights(
                {
                    "netD": checkpoint["netD_state_dict"],
                    "netG": checkpoint["netG_state_dict"],
                }
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            epoch_offset = checkpoint["total_training_epochs"]
            self.train_history = checkpoint["train_history"]

            if sinkhorn_params is None:
                self.sinkhorn_params = checkpoint["sinkhorn_params"]

            if sinkhorn_lambda_scheduling_params is None:
                self.sinkhorn_lambda_scheduling_params = checkpoint[
                    "sinkhorn_lambda_scheduling_params"
                ]

            if model_training_params is None:
                self.model_training_params = checkpoint["model_training_params"]

            if norms_params is None:
                self.norms_params = checkpoint["norms_params"]

            print(f"Continuing training from checkpoint at epoch {epoch_offset}")

        train_with_recon_only = model_training_params["train_with_recon_only"]
        power_p = model_training_params["power_p"]
        model = self.model
        train_size = self.training_x.shape[0]
        val_size = self.validation_x.shape[0] if self.validation_x is not None else 0
        nb_epochs = self.model_training_params["nb_epochs"]
        batch_size = self.model_training_params["batch_size"]
        device = self.device
        optimizer = self.optimizer
        z_dist = self.latent_dist
        z_dist_params = self.latent_dist_params_list
        latent_dim = self.latent_dim
        input_dim = self.input_dim
        dim_x = self.dim_x
        dim_y = self.dim_y
        nx = self.nx
        ny = self.ny
        model_training_dir = self.model_training_dir
        exp_logger = self.tsb_logger
        epsilon_sk = self.sinkhorn_params["epsilon"]
        niter_sk = self.sinkhorn_params["niter"]
        p_sk = self.sinkhorn_params["p"]
        sink_scale_by_dim = self.sinkhorn_params["scale_by_dim"]
        sink_lambda = self.sinkhorn_lambda_scheduling_params["sink_lambda"]
        sink_lambda_sched_factor = self.sinkhorn_lambda_scheduling_params[
            "sink_lambda_scheduler_factor"
        ]
        sink_lambda_sched_epoch = self.sinkhorn_lambda_scheduling_params[
            "sink_lambda_scheduler_epoch"
        ]
        norm_fn_x = norm_fn_selector(self.norms_params["norm_fct_type_x"])
        norm_fn_y = norm_fn_selector(self.norms_params["norm_fct_type_y"])
        best_model_metric = self.model_training_params["best_model_metric"]
        training_stop_metric_threshold = self.model_training_params[
            "training_stop_metric_threshold"
        ]
        optim_lr_scheduler = self.optim_lr_scheduler
        inflate_recon_y = self.model_training_params["inflate_recon_y"]
        recon_y_alpha = 1

        ## objects to store training history
        if continue_training_from_checkpoint is None:
            train_hist = {}
            train_hist[
                "train_reconst"
            ] = []  # reconstruction part of the loss on training data
            train_hist["train_loss"] = []  # total loss on training data
            train_hist[
                "train_latent_dist"
            ] = []  # distribution distance in latent space - on training data
            train_hist["validation_loss"] = []  # total loss on validation data
            train_hist[
                "validation_reconst"
            ] = []  # reconstruction part of the loss on validation data
            train_hist[
                "validation_latent_dist"
            ] = []  # distribution distance in latent space - on validation data
            train_hist["avg_val_recons_rmse_x"] = []  #
            train_hist["avg_val_recons_rmse_y"] = []
            train_hist["lower_val_recons_rmse_x"] = []
            train_hist["lower_val_recons_rmse_y"] = []
            train_hist["upper_val_recons_rmse_x"] = []
            train_hist["upper_val_recons_rmse_y"] = []
            train_hist["per_epoch_ptimes"] = []  # per epoch training time duration
            train_hist["total_ptime"] = None  # total epochs training time
            train_hist["best_val_metric"] = None  # final best validation metric reached
            best_val_metric = float("inf")
        else:
            train_hist = self.train_history
            best_val_metric = train_hist["best_val_metric"]

        # setup plotting tools
        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config(mpl)

        # train NN
        ## generate a fixed vector to plot visual evolution during training
        fixed_z = (
            z_dist(
                torch.tensor(z_dist_params[0], dtype=torch.float32),
                torch.tensor(z_dist_params[1], dtype=torch.float32),
            )
            .sample((1, latent_dim))
            .to(device)
        )
        print(f"Fixed latent tensor shape: {fixed_z.shape}")
        print(
            f"Latent distribution: {z_dist.__name__}, "
            f"param 1: {z_dist_params[0]}, param 2: {z_dist_params[1]}"
        )

        ## add network graph to tenserboard
        with torch.no_grad():
            model.eval()
            dummy_input = torch.rand(10, input_dim).to(device)
            # inter_output, _, _ = model.netD(dummy_input)
            # exp_logger.add_graph(model.netD, dummy_input)
            # exp_logger.add_graph(model.netG, inter_output)
            exp_logger.add_graph(model, dummy_input)

        print("Starting training loop ...")

        prec_nugget = 1e-10
        start_time = time.time()  # start time of the whole training

        for epoch in range(nb_epochs):
            model.train()  # set networks in training mode
            # model.netG.train()
            # model.netD.train()

            recons_losses = []
            losses = []
            latent_dist = []

            cycle_steps = train_size // batch_size

            # shuffle training set
            idx = torch.randperm(train_size)
            x = (
                self.training_x[idx, :, :, :]
                if len(self.training_x.shape) > 2
                else self.training_x[idx, :].view(train_size, dim_x)
            )
            y = self.training_y[idx, :]

            if sink_lambda ** (1 / power_p) > 1:
                old_sink_lambda = sink_lambda

                if sink_lambda_sched_factor == -1 and epoch > 0:
                    sink_lambda = (running_reconstruction_loss / cycle_steps) // (
                        (running_latent_distance_loss / cycle_steps) + prec_nugget
                    )  # don't use validation data here (risk of contaminating the training with the validation set)
                    if sink_lambda == 0:
                        sink_lambda = 1

                if sink_lambda_sched_factor > 0:
                    if epoch % sink_lambda_sched_epoch == 0 and epoch != 0:
                        sink_lambda = max(sink_lambda // sink_lambda_sched_factor, 1)

                print(
                    f"Sinkhorn lambda changed from {old_sink_lambda} to {sink_lambda}"
                )

            else:
                # stop reducing sink_lambda below 1
                sink_lambda = 1
                sink_lambda_sched_factor = 1

            if inflate_recon_y:
                if (
                    epoch > 0
                ):  # don't use validation data here (risk of contaminating the training with the validation set)
                    recon_y_alpha = (running_recon_x / cycle_steps) // (
                        running_recon_y / cycle_steps
                    )
                    if recon_y_alpha == 0:
                        recon_y_alpha = 1

            running_reconstruction_loss = 0.0
            running_latent_distance_loss = 0.0

            running_recon_x = 0.0
            running_recon_y = 0.0

            epoch_start_time = time.time()  # start time of the current epoch

            for i in range(cycle_steps):
                # might miss some examples if batch size is not a divider of training size,
                # should not be very problematic since we shuffle data at each epoch

                # read databatch
                x_batch = x[i * batch_size : (i + 1) * batch_size, :]
                y_batch = y[i * batch_size : (i + 1) * batch_size, :]

                b_size = x_batch.shape[0]

                x_batch = x_batch.view(-1, dim_x).to(device)
                y_batch = y_batch.view(-1, dim_y).to(device)

                data_batch = torch.cat((x_batch, y_batch), 1)
                # print('data_batch:', data_batch.shape)

                model.zero_grad()
                model.netG.zero_grad()
                model.netD.zero_grad()

                real_codes, real_x_codes, real_y_codes = model.netD(data_batch)

                latent_vector = (
                    z_dist(
                        torch.tensor(z_dist_params[0], dtype=torch.float32),
                        torch.tensor(z_dist_params[1], dtype=torch.float32),
                    )
                    .sample((b_size, latent_dim))
                    .to(device)
                )
                # print('latent_vector:',latent_vector.shape)

                reconstructed_data = model.netG(real_codes)
                reconstrcuted_x = reconstructed_data[:, 0:dim_x]
                reconstrcuted_y = reconstructed_data[:, dim_x:]

                ## Sinkhorn Auto-Encoder loss:

                ## part 1 : reconstruction
                cx_batch = torch.mean(
                    norm_fn_x(
                        x_batch,
                        reconstrcuted_x,
                        **{"p": self.norms_params["l_norm_p_x"]},
                    )
                )
                cy_batch = torch.mean(
                    norm_fn_y(
                        y_batch,
                        reconstrcuted_y,
                        **{"p": self.norms_params["l_norm_p_y"]},
                    )
                )

                reconst_cost_batch = cx_batch + recon_y_alpha * cy_batch

                ## part 2 : Sinkhorn regularization
                if not train_with_recon_only:
                    latent_dist_batch = spc.sinkhorn_normalized(
                        latent_vector,
                        real_codes,
                        epsilon_sk,
                        b_size,
                        niter_sk,
                        p_sk,
                        device=device,
                    )
                    if sink_scale_by_dim:
                        latent_dist_batch = latent_dist_batch / latent_dim
                else:
                    latent_dist_batch = torch.tensor(0.0, dtype=torch.float32)

                ## total loss
                model_loss_batch = (
                    reconst_cost_batch + sink_lambda * latent_dist_batch
                ) * 2 ** (power_p - 1)

                model_loss_batch.backward()
                optimizer.step()

                if lr_scheduling:
                    if (
                        optim_params["lr_scheduler"] == "one_cycle"
                    ):  # after batch training
                        optim_lr_scheduler.step()

                losses.append(model_loss_batch.item())
                recons_losses.append(reconst_cost_batch.item())
                latent_dist.append(latent_dist_batch.item())

                # Update running losses
                running_reconstruction_loss += reconst_cost_batch.item()
                running_latent_distance_loss += latent_dist_batch.item()

                running_recon_x += cx_batch.item()
                running_recon_y += cy_batch.item()

            # Run evaluation on validation set at end of every epoch
            model.eval()

            val_loss = 0
            val_reconst = 0
            val_latent_dist = 0

            val_recons_rmse_x = []
            val_recons_rmse_y = []

            idx = torch.randperm(val_size)
            if val_size > 0:
                val_x = (
                    self.validation_x[idx, :, :, :]
                    if len(self.validation_x.shape) > 2
                    else self.validation_x[idx, :].view(val_size, dim_x)
                )
                val_y = self.validation_y[idx, :]
            else:
                raise ValueError("Validation set is empty")

            for i in range(val_size // batch_size):
                val_x_batch = val_x[i * batch_size : (i + 1) * batch_size, :]
                val_y_batch = val_y[i * batch_size : (i + 1) * batch_size, :]

                val_b_size = val_x_batch.shape[0]

                val_x_batch = val_x_batch.view(-1, dim_x).to(device)
                val_y_batch = val_y_batch.view(-1, dim_y).to(device)

                val_data_batch = torch.cat((val_x_batch, val_y_batch), 1)

                val_real_codes, val_real_x_codes, val_real_y_codes = model.netD(
                    val_data_batch
                )

                val_latent_vector = (
                    z_dist(
                        torch.tensor(z_dist_params[0], dtype=torch.float32),
                        torch.tensor(z_dist_params[1], dtype=torch.float32),
                    )
                    .sample((val_b_size, latent_dim))
                    .to(device)
                )

                val_reconstructed_data = model.netG(val_real_codes)
                val_reconstrcuted_x = val_reconstructed_data[:, 0:dim_x]
                val_reconstrcuted_y = val_reconstructed_data[:, dim_x:]

                # Model loss:

                # part 1 : reconstruction
                val_cx = torch.mean(
                    norm_fn_x(
                        val_x_batch,
                        val_reconstrcuted_x,
                        **{"p": self.norms_params["l_norm_p_x"]},
                    )
                )
                val_cy = torch.mean(
                    norm_fn_y(
                        val_y_batch,
                        val_reconstrcuted_y,
                        **{"p": self.norms_params["l_norm_p_y"]},
                    )
                )

                val_reconst_cost_batch = val_cx + val_cy

                # part 2 : regularization
                if not train_with_recon_only:
                    val_latent_dist_batch = spc.sinkhorn_normalized(
                        val_latent_vector,
                        val_real_codes,
                        epsilon_sk,
                        val_b_size,
                        niter_sk,
                        p_sk,
                        device=device,
                    )
                    if sink_scale_by_dim:
                        val_latent_dist_batch = val_latent_dist_batch / latent_dim
                else:
                    val_latent_dist_batch = torch.tensor(0.0, dtype=torch.float32)

                # total
                val_model_loss_batch = (
                    val_reconst_cost_batch + sink_lambda * val_latent_dist_batch
                ) * 2 ** (power_p - 1)

                val_loss += val_model_loss_batch.cpu().data.numpy()
                val_reconst += val_reconst_cost_batch.cpu().data.numpy()
                val_latent_dist += val_latent_dist_batch.cpu().data.numpy()

                if self.normalize:
                    val_rmse_x_batch = td.rmse_torch(
                        tdp.un_normalize(
                            val_reconstrcuted_x.view(val_b_size, dim_x),
                            self.normalization_dict_x,
                        ),
                        tdp.un_normalize(
                            val_x_batch.view(val_b_size, dim_x),
                            self.normalization_dict_x,
                        ),
                    )
                    val_rmse_y_batch = td.rmse_torch(
                        tdp.un_normalize(
                            val_reconstrcuted_y.view(val_b_size, dim_y),
                            self.normalization_dict_y,
                        ),
                        tdp.un_normalize(
                            val_y_batch.view(val_b_size, dim_y),
                            self.normalization_dict_y,
                        ),
                    )
                else:
                    val_rmse_x_batch = td.rmse_torch(val_reconstrcuted_x, val_x_batch)
                    val_rmse_y_batch = td.rmse_torch(val_reconstrcuted_y, val_y_batch)

                val_recons_rmse_x.append(val_rmse_x_batch.cpu().data.numpy())
                val_recons_rmse_y.append(val_rmse_y_batch.cpu().data.numpy())

            val_batches = val_size // batch_size
            avg_val_loss = val_loss / val_batches  # average loss on validation set
            avg_val_reconst = (
                val_reconst / val_batches
            )  # average reconstruction loss on validation set
            avg_val_latent_dist = (
                val_latent_dist / val_batches
            )  # average latent distance on validation set

            val_recons_rmse_x = np.array(val_recons_rmse_x).reshape(-1)
            val_recons_rmse_y = np.array(val_recons_rmse_y).reshape(-1)

            avg_val_recons_rmse_x = np.mean(
                val_recons_rmse_x
            )  # average reconstruction rmse on validation set for x
            avg_val_recons_rmse_y = np.mean(
                val_recons_rmse_y
            )  # average reconstruction rmse on validation set for y

            lower_val_recons_rmse_x = avg_val_recons_rmse_x - 1.96 * np.std(
                val_recons_rmse_x
            )
            lower_val_recons_rmse_y = avg_val_recons_rmse_y - 1.96 * np.std(
                val_recons_rmse_y
            )

            upper_val_recons_rmse_x = avg_val_recons_rmse_x + 1.96 * np.std(
                val_recons_rmse_x
            )
            upper_val_recons_rmse_y = avg_val_recons_rmse_y + 1.96 * np.std(
                val_recons_rmse_y
            )

            if best_model_metric == "recon":
                val_metric = avg_val_reconst
            elif best_model_metric == "full_loss":
                val_metric = avg_val_loss
            else:
                raise ValueError(
                    "Encountered unknown 'best_model_metric' value at model evaluation."
                )

            train_hist["validation_loss"].append(avg_val_loss)
            train_hist["validation_reconst"].append(avg_val_reconst)
            train_hist["validation_latent_dist"].append(avg_val_latent_dist)
            train_hist["avg_val_recons_rmse_x"].append(avg_val_recons_rmse_x)
            train_hist["avg_val_recons_rmse_y"].append(avg_val_recons_rmse_y)
            train_hist["lower_val_recons_rmse_x"].append(lower_val_recons_rmse_x)
            train_hist["lower_val_recons_rmse_y"].append(lower_val_recons_rmse_y)
            train_hist["upper_val_recons_rmse_x"].append(upper_val_recons_rmse_x)
            train_hist["upper_val_recons_rmse_y"].append(upper_val_recons_rmse_y)

            if val_metric < (best_val_metric - 1e-4):
                # store model checkpoint when improvement on validation metric is seen or every 100 epochs
                best_val_metric = val_metric
                train_hist["best_val_metric"] = best_val_metric
                self.best_training_epoch = epoch + epoch_offset

                # save model checkpoint
                self.total_training_epochs = nb_epochs + epoch_offset
                self.train_history = train_hist
                self.optimizer = optimizer  # save optimizer state
                self.sinkhorn_lambda_scheduling_params["sink_lambda"] = sink_lambda
                self.optim_lr_scheduler = optim_lr_scheduler

                self.save_checkpoint(epoch + epoch_offset, save_full_experiment=True)

            if not (epoch + epoch_offset) % 100:
                # save model checkpoint every 100 epochs regardless of improvement
                self.train_history = train_hist
                self.total_training_epochs = nb_epochs + epoch_offset
                self.optimizer = optimizer  # save optimizer state
                self.sinkhorn_lambda_scheduling_params["sink_lambda"] = sink_lambda
                self.optim_lr_scheduler = optim_lr_scheduler

                self.save_checkpoint(epoch + epoch_offset, save_full_experiment=True)

                """
                torch.save(
                    model.netG.state_dict(),
                    f"{model_training_dir}/models/netG_params_epoch_{epoch}.pth",
                )
                torch.save(
                    model.netD.state_dict(),
                    f"{model_training_dir}/models/netD_params_epoch_{epoch}.pth",
                )"""

            if best_val_metric < training_stop_metric_threshold:
                print(
                    f"Training finished at epoch {epoch + epoch_offset}, reaching loss of {best_val_metric:.5f}"
                )
                break

            if lr_scheduling:
                if (
                    optim_params["lr_scheduler"] == "on_plateau"
                ):  # after validation evaluation
                    optim_lr_scheduler.step(val_metric)

            epoch_end_time = (
                time.time()
            )  # end time of the current epoch. Includes validation time.
            per_epoch_ptime = epoch_end_time - epoch_start_time

            avg_training_loss = torch.mean(torch.FloatTensor(losses))
            avg_training_reconst = torch.mean(torch.FloatTensor(recons_losses))
            avg_training_latent_dist = torch.mean(torch.FloatTensor(latent_dist))

            print(
                f"Epoch {epoch + epoch_offset} - epoch time :{per_epoch_ptime:.2f} s, train_loss: {avg_training_loss:.3f}, "
                f"val_loss: {avg_val_loss:.3f}, train_recon: {avg_training_reconst:.3f}, "
                f"val_recon: {avg_val_reconst:.3f}, train_latent_dist: {avg_training_latent_dist:.3f},"
                f"val_latent_dist: {avg_val_latent_dist:.3f}, "
                f"avg val recon rmse x: {avg_val_recons_rmse_x:.3f}, avg val recon rmse y: {avg_val_recons_rmse_y:.3f}"
            )

            # log to tensorboard
            exp_logger.add_scalar("Loss/train", avg_training_loss, epoch + epoch_offset)
            exp_logger.add_scalar("Loss/val", avg_val_loss, epoch + epoch_offset)
            exp_logger.add_scalar(
                "Recons/train", avg_training_reconst, epoch + epoch_offset
            )
            exp_logger.add_scalar("Recons/val", avg_val_reconst, epoch + epoch_offset)
            exp_logger.add_scalar(
                "Latent distance/train", avg_training_latent_dist, epoch + epoch_offset
            )
            exp_logger.add_scalar(
                "Latent distance/val", avg_val_latent_dist, epoch + epoch_offset
            )

            exp_logger.add_scalar(
                f"Best val {best_model_metric} metric",
                best_val_metric,
                epoch + epoch_offset,
            )

            exp_logger.add_histogram(
                "Val recons. RMSE X", val_recons_rmse_x, epoch + epoch_offset, bins=30
            )
            exp_logger.add_histogram(
                "Val recons. RMSE Y", val_recons_rmse_y, epoch + epoch_offset, bins=30
            )
            exp_logger.add_scalar(
                "VAL RMSE/recons. RMSE X", avg_val_recons_rmse_x, epoch + epoch_offset
            )
            exp_logger.add_scalar(
                "VAL RMSE/recons. RMSE Y", avg_val_recons_rmse_y, epoch + epoch_offset
            )

            train_hist["train_loss"].append(avg_training_loss)
            train_hist["train_reconst"].append(avg_training_reconst)
            train_hist["train_latent_dist"].append(
                torch.mean(torch.FloatTensor(latent_dist))
            )
            train_hist["per_epoch_ptimes"].append(per_epoch_ptime)

            # check fixed noise results
            model.eval()
            fixed_results = model.netG(fixed_z.to(device)).detach().cpu()
            fixed_x = fixed_results[:, 0:dim_x].reshape(1, dim_x)
            fixed_y = fixed_results[:, dim_x:].reshape(1, dim_y)
            if self.normalize:
                if self.data_prep["function"] == "standardize":
                    fixed_x = tdp.un_standardize(fixed_x, self.x_mean, self.x_std)
                    fixed_y = tdp.un_standardize(fixed_y, self.y_mean, self.y_std)
                    # fixed_x = fixed_x * self.x_std + self.x_mean
                    # fixed_y = fixed_y * self.y_std + self.y_mean
                if self.data_prep["function"] == "min_max":
                    fixed_x = tdp.min_max_unnormalize(
                        fixed_x, self.x_lower, self.x_upper
                    )
                    fixed_y = tdp.min_max_unnormalize(
                        fixed_y, self.y_lower, self.y_upper
                    )
            fixed_x = fixed_x.squeeze().reshape(ny, nx)
            plot.plot_example(
                example_x=fixed_x,
                example_y=fixed_y.squeeze(),
                save_location=f"{self.model_training_dir}/training_plots/model_epoch_{epoch+epoch_offset}.png",
            )

        end_time = time.time()  # end time of the whole training
        total_ptime = end_time - start_time
        train_hist["total_ptime"] = total_ptime
        train_hist["best_val_metric"] = best_val_metric

        print("Training finished!... save training results \n")
        print("Last epoch stats: \n")
        print(
            f"Epoch {epoch + epoch_offset} - epoch time :{per_epoch_ptime:.2f} s, train_loss: {avg_training_loss:.3f}, "
            f"val_loss: {avg_val_loss:.3f}, train_recon: {avg_training_reconst:.3f}, "
            f"val_recon: {avg_val_reconst:.3f}, train_latent_dist: {avg_training_latent_dist:.3f},"
            f"val_latent_dist: {avg_val_latent_dist:.3f}, "
            f"avg val recon rmse x: {avg_val_recons_rmse_x:.3f}, avg val recon rmse y: {avg_val_recons_rmse_y:.3f} \n"
        )

        print(
            f"Avg epoch time: {torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])):.2f}, "
            f"total {nb_epochs} epochs, total training time: {total_ptime:.2f}"
        )

        # save models checkpoints/configurations
        train_hist["best_val_metric"] = best_val_metric

        self.total_training_epochs = nb_epochs + epoch_offset
        self.train_history = train_hist
        self.optimizer = optimizer  # save optimizer state
        self.sinkhorn_lambda_scheduling_params["sink_lambda"] = sink_lambda
        self.optim_lr_scheduler = optim_lr_scheduler

        self.save_checkpoint(epoch + epoch_offset, save_full_experiment=True)
        """
        torch.save(
            model.netG.state_dict(),
            f"{model_training_dir}/models/netG_params_epoch_{epoch}.pth",
        )
        torch.save(
            model.netD.state_dict(),
            f"{model_training_dir}/models/netD_params_epoch_{epoch}.pth",
        )"""

        with open(f"{model_training_dir}/train_hist", "wb") as f:
            pickle.dump(train_hist, f)

        exp_logger.flush()
        exp_logger.close()
