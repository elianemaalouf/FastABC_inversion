"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
This file allows to configure and run the cVAE (conditional Variational Autoencoder)
experiment for the Geophysics problems.
"""

import inspect
import os
import pickle
import random
import string
import time

import fastabc_inversion.geo_problems.utils.torch_data_prep as tdp
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
from fastabc_inversion.geo_problems.jGNN_SuS_exp import params_init
from fastabc_inversion.geo_problems.utils.config import Config
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


def cvae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    CVAE loss function combining reconstruction loss and KL divergence

    Args:
        recon_x: reconstructed input
        x: original input
        mu: mean of latent distribution
        logvar: log variance of latent distribution
        beta: weighting factor for KL divergence (beta-VAE)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    # KL divergence loss
    # Aggregation : sum across ALL dimensions and batches
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Aggregation : sum across latent dims, mean across batch
    # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_cvae(model, train_x, train_y, batch_size, optimizer, device, beta):
    """Training loop for one epoch of the cVAE model"""

    model.train()  # set networks in training mode

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    train_size = train_x.shape[0]
    dim_x = train_x.shape[1]
    dim_y = train_y.shape[1]

    cycle_steps = train_size // batch_size

    # shuffle training set
    suffle_idx = torch.randperm(train_size)
    x = train_x[suffle_idx, :]
    y = train_y[suffle_idx, :]

    for i in range(cycle_steps):
        # might miss some examples if batch size is not a divider of training size,
        # should not be very problematic since we shuffle data at each epoch

        # read databatch
        x_batch = x[i * batch_size : (i + 1) * batch_size, :]
        y_batch = y[i * batch_size : (i + 1) * batch_size, :]

        x_batch = x_batch.view(-1, dim_x).to(device)
        y_batch = y_batch.view(-1, dim_y).to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_x, mu, logvar = model(x_batch, y_batch)

        loss, recon_loss, kl_loss = cvae_loss(recon_x, x_batch, mu, logvar, beta=beta)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    avg_loss = total_loss / cycle_steps
    avg_recon_loss = total_recon_loss / cycle_steps
    avg_kl_loss = total_kl_loss / cycle_steps

    return avg_loss, avg_recon_loss, avg_kl_loss


def validate_cvae(model, val_x, val_y, batch_size, device, beta):
    """Validation loop for the cVAE model"""

    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    val_size = val_x.shape[0]
    dim_x = val_x.shape[1]
    dim_y = val_y.shape[1]

    cycle_steps = val_size // batch_size

    # shuffle training set
    suffle_idx = torch.randperm(val_size)
    x = val_x[suffle_idx, :]
    y = val_y[suffle_idx, :]

    with torch.no_grad():
        for i in range(cycle_steps):
            # might miss some examples if batch size is not a divider of training size,
            # should not be very problematic since we shuffle data at each epoch

            # read databatch
            x_batch = x[i * batch_size : (i + 1) * batch_size, :]
            y_batch = y[i * batch_size : (i + 1) * batch_size, :]

            x_batch = x_batch.view(-1, dim_x).to(device)
            y_batch = y_batch.view(-1, dim_y).to(device)

            # Forward pass
            recon_x, mu, logvar = model(x_batch, y_batch)

            loss, recon_loss, kl_loss = cvae_loss(
                recon_x, x_batch, mu, logvar, beta=beta
            )

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

    avg_loss = total_loss / cycle_steps
    avg_recon_loss = total_recon_loss / cycle_steps
    avg_kl_loss = total_kl_loss / cycle_steps

    return avg_loss, avg_recon_loss, avg_kl_loss


class cVAE_exp:
    def __init__(
        self,
        parameters_file,
        name,
        model=None,
        run_id=None,
        seed=None,
        load_pretrained_model=False,
        pretrained_model_path_netD=None,
        pretrained_model_path_netG=None,
        model_training_params=None,
        noise_dicts=None,
        use_cuda=True,
        gpu_id=0,
        name_suffix="",
    ):
        """
        Experiment base class
        :param name: name of the experiment
        :param parameters_file: mandatory parameter to specify the path to the configuration file.
        :param model: model to be used for the experiment
        :param run_id: run id of the experiment. Used to create a folder for the experiment in the data_rootdir.
        :param seed: seed for reproducibility
        :param load_pretrained_model: whether to load a pretrained model
        :param pretrained_model_path_netD: path to the pretrained netD model
        :param pretrained_model_path_netG: path to the pretrained netG model
        :param model_training_params: parameters for training the model
        :param noise_dict: dictionary containing the noise configuration
        :param use_cuda: whether to use gpu or cpu
        :param gpu_id: id of the gpu to use if use_cuda is True
        :param name_suffix: suffix to add to the experiment name. Start with '_'.
        """
        self.config = Config(parameters_file)
        data_rootdir = self.config.datadir + "/Geo_cVAE_experiments"
        os.makedirs(data_rootdir, exist_ok=True)

        self.latent_dim = None
        self.input_dim = None
        self.output_dim = None
        self.inversion_dir = None
        self.data_rootdir = data_rootdir
        self.model = model
        self.model_log_ref = None
        self.netG = self.model.netG if self.model is not None else None
        self.netD = self.model.netD if self.model is not None else None
        self.device = self.model.device if self.model is not None else None
        self.optimizer = None
        self.optim_params = None
        self.lr_scheduling = None
        self.optim_lr_scheduler = None
        self.load_pretrained_model = load_pretrained_model
        self.pretrained_model_netD_state_dict = pretrained_model_path_netD
        self.pretrained_model_netG_state_dict = pretrained_model_path_netG

        self.nn_params = None
        self.model_training_params = model_training_params
        self.best_training_epoch = None

        self.noise_dicts = noise_dicts
        self.obs_inference_dir_prefix = None
        self.inverted_obs_idx = None
        self.all_obs_inference_results = None

        self.dim_x = None
        self.dim_y = None
        self.normalize = False
        self.data_prep = None
        self.normalization_dict_x = None
        self.normalization_dict_y = None

        if run_id is None:
            # generate a random run_id
            self.run_id = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=5)
            )
        else:
            self.run_id = run_id

        self.name = f"{name}{name_suffix}_{self.run_id}"

        if seed is None:
            # generate a random seed
            self.seed = random.randint(1, 10000)
        else:
            self.seed = seed

        # fix seed for reproducibility
        # "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different
        # platforms. Furthermore, results may not be reproducible between CPU and GPU executions,
        # even when using identical seeds." See https://pytorch.org/docs/stable/notes/randomness.html
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # create experiment directory
        self.experiment_dir = f"{self.data_rootdir}/{self.name}"
        os.makedirs(self.experiment_dir, exist_ok=True)

        # create directories for storing models snapshots and logs
        self.model_training_dir = f"{self.experiment_dir}/model_training_data"
        os.makedirs(self.model_training_dir, exist_ok=True)
        os.makedirs(f"{self.model_training_dir}/training_plots", exist_ok=True)
        os.makedirs(f"{self.model_training_dir}/models", exist_ok=True)

        # create directories for storing inference results
        self.inference_dir = f"{self.experiment_dir}/inference_results"
        os.makedirs(self.inference_dir, exist_ok=True)

        # prepare for SuS obs inference directories creation
        self.obs_inference_dir_prefix = "test_SuS_obs_"

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
        # os.makedirs(f"{self.model_training_dir}/reconstruction_val", exist_ok=True)
        # os.makedirs(f"{self.model_training_dir}/reconstruction_train", exist_ok=True)

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

    def save_model_summary(self, input_dim):
        """
        Save the model summary to a file in the model training directory
        """
        result = torchinfo.summary(self.model, input_size=input_dim, device=self.device)

        with open(f"{self.model_training_dir}/model_summary.txt", "a+") as f:
            f.write(str(result))

    def save_experiment(self):
        """
        Save all experiments configurations and results to files in the experiment directory.
        Does not save training data and inverted latent samples.
        """

        # save experiment configurations
        with open(f"{self.experiment_dir}/experiment_configurations.txt", "a+") as f:
            f.write(f"Experiment name: {self.name}\n")
            f.write(f"Experiment run id: {self.run_id}\n")
            f.write(f"Experiment seed: {self.seed}\n")
            f.write(f"Experiment data rootdir: {self.data_rootdir}\n")
            f.write(f"Experiment data preparation: {self.data_prep}\n")
            f.write(f"Experiment model selector: {self.model_selector}\n")
            f.write(f"Experiment load pretrained model: {self.load_pretrained_model}\n")
            f.write(
                f"Experiment pretrained model path netD: {self.pretrained_model_netD_state_dict}\n"
            )
            f.write(
                f"Experiment pretrained model path netG: {self.pretrained_model_netG_state_dict}\n"
            )
            f.write(f"Experiment device: {self.device}\n")
            f.write(f"Experiment optimizer: {self.optimizer}\n")
            f.write(f"Experiment optimizer lr scheduler: {self.optim_lr_scheduler}\n")
            f.write(
                f"Experiment model training parameters: {self.model_training_params}\n"
            )
            f.write(f"Experiment dims: x, y: {self.dim_x}, {self.dim_y}\n")
            f.write(f"Experiment training best epoch: {self.best_training_epoch}\n")

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

                self.training_x = self.training_x.view(self.train_size, self.dim_x)
                self.training_y = self.training_y.view(self.train_size, self.dim_y)

                self.x_mean = self.training_x.mean(axis=0)
                self.x_std = self.training_x.std(axis=0)
                self.y_mean = self.training_y.mean(axis=0)
                self.y_std = self.training_y.std(axis=0)

                if data_prep is not None:
                    self.normalize = True
                    self.data_prep = data_prep  # store original configuration

                    if data_prep["function"] == "standardize":
                        print("Standardizing inputs")
                        self.training_x = tdp.standardize(
                            self.training_x, self.x_mean, self.x_std
                        )
                        self.training_y = tdp.standardize(
                            self.training_y, self.y_mean, self.y_std
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
                            self.training_x, **data_prep["kwargs"]
                        )
                        (
                            self.training_y,
                            self.y_lower,
                            self.y_upper,
                        ) = tdp.min_max_normalize(
                            self.training_y, **data_prep["kwargs"]
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
                self.validation_x = self.validation_x.view(self.val_size, self.dim_x)
                self.validation_y = self.validation_y.view(self.val_size, self.dim_y)

                print("Validation data loaded")
                print("Validation models shape:", self.validation_x.shape)
                print("Validation travel times shape:", self.validation_y.shape)

                if data_prep is not None:
                    if (
                        self.normalization_dict_x is None
                        or self.normalization_dict_y is None
                    ):
                        raise ValueError(
                            "Training data must be loaded before validation data"
                        )

                    self.validation_x = tdp.normalize(
                        self.validation_x, self.normalization_dict_x
                    )
                    self.validation_y = tdp.normalize(
                        self.validation_y, self.normalization_dict_y
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

    def construct_model_architecture(self, model_selector, nn_params):
        """
        Construct the model architecture.
        :param nn_params: dictionary of extra neural network parameters
        """
        print(f"Constructing cVAE model architecture ...")

        if model_selector == "cvae":
            import fastabc_inversion.geo_problems.nn.cVAE as nnm
        elif model_selector == "cvae_red_y":
            import fastabc_inversion.geo_problems.nn.cVAE_red_y as nnm
        else:
            raise ValueError(
                f"Invalid model selector: {model_selector}. Choose 'cvae' or 'cvae_red_y'."
            )

        self.input_dim = self.dim_x + self.dim_y
        self.output_dim = self.dim_x
        self.latent_dim = nn_params["latent_dim"]
        self.nn_params = nn_params
        self.model_selector = model_selector

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
        self.netG = nnm.netG(
            ngpu=self.ngpu,
            ndata=self.dim_y,
            height=self.ny,
            width=self.nx,
            **netG_params,
        )  # decoder

        self.model = nnm.cVAE(
            encoder=self.netD, decoder=self.netG, ngpu=self.ngpu
        )  # encoder - decoder

        self.save_model_summary(
            input_dim=[
                (self.model_training_params["batch_size"], self.dim_x),
                (self.model_training_params["batch_size"], self.dim_y),
            ]
        )

        _map_module_init_param = {
            "netD": ("prelu", None, True),
            "netG": ("prelu", None, True),
        }
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
            "model_log_ref": self.model_log_ref,
            "nn_params": self.nn_params,
            "netD_state_dict": self.model.netD.state_dict(),
            "netG_state_dict": self.model.netG.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optim_params": self.optim_params,
            "lr_scheduling": self.lr_scheduling,
            "total_training_epochs": self.total_training_epochs,
            "train_history": self.train_history,
            "model_training_params": self.model_training_params,
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

    def plot_reconstruction_check(self, epoch, save_dir=None):
        """
        Plot validation input images against their reconstructed outputs

        Args:
            epoch: Current training epoch
            save_dir: Directory to save plots (optional, defaults to model_training_dir)
        """
        if not (epoch % 30 == 0 or epoch == 1):
            return

        from fastabc_inversion.geo_problems.utils.visualization import \
            plot_reconstruction

        if save_dir is None:
            save_dir = self.model_training_dir

        # Create reconstruction check directory if it doesn't exist
        recon_check_dir = os.path.join(save_dir, "reconstruction_checks")
        os.makedirs(recon_check_dir, exist_ok=True)

        # Set model to evaluation mode
        self.model.eval()

        # Get a batch of validation data
        val_batch_size = min(3, len(self.validation_x))  # Plot up to 4 samples
        val_indices = np.random.choice(
            len(self.validation_x), val_batch_size, replace=False
        )

        val_x_batch = self.validation_x[val_indices]
        val_y_batch = self.validation_y[val_indices]

        # Convert to tensors and move to device
        val_x_tensor = (
            val_x_batch.clone().detach().to(dtype=torch.float32, device=self.device)
        )
        val_y_tensor = (
            val_y_batch.clone().detach().to(dtype=torch.float32, device=self.device)
        )

        # Generate reconstructions
        with torch.no_grad():
            reconstructed, _, _ = self.model(val_x_tensor, val_y_tensor)
            reconstructed = reconstructed.cpu().numpy()

        # Plot each sample
        for i in range(val_batch_size):
            # Stack original and reconstructed data
            comparison_data = np.stack([val_x_batch[i], reconstructed[i]], axis=0)

            # Determine dimensions for plotting
            if hasattr(self, "nx") and hasattr(self, "ny"):
                dims = (self.ny, self.nx)
            else:
                # Assume 1D data if dimensions not available
                dims = (1, len(val_x_batch[i]))

            # Create save path
            save_path = os.path.join(
                recon_check_dir, f"epoch_{epoch:04d}_sample_{i + 1}.png"
            )

            # Plot reconstruction
            plot_reconstruction(
                comparison_data,
                dims,
                title=f"Epoch {epoch}, Sample {i + 1}",
                save_location=save_path,
                dpi=300,
            )

        print(
            f"Reconstruction check plots saved for epoch {epoch} in {recon_check_dir}"
        )

    def train_model(
        self,
        model_selector,
        nn_params,
        lr_scheduling,
        optim_params,
        model_training_params,
        model_log_ref,
        continue_training_from_checkpoint,
    ):
        """
        Train the model.
        Parameters to pass to prep_model():
        :param model_selector: name of the model to use
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
        :param model_training_params: dictionary of model training parameters.
        The dictionary contains:
            - batch_size: batch size for training
            - nb_epochs: number of epochs for training
            - beta : weighting factor for KL divergence (beta-VAE)
            - training_stop_metric_threshold: threshold for the training stopping metric. If the metric is below this
        threshold, training stops.
        :param model_log_ref: short reference to be used for logging model performance in log file
        :param continue_training_from_checkpoint: whether to continue training from a checkpoint, contains the path to the checkpoint.
        """

        self.model_training_params = model_training_params

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

            if model_training_params is None:
                self.model_training_params = checkpoint["model_training_params"]

            print(f"Continuing training from checkpoint at epoch {epoch_offset}")

        beta = (
            self.model_training_params["beta"]
            if "beta" in self.model_training_params
            else 1.0
        )

        model = self.model
        nb_epochs = self.model_training_params["nb_epochs"]
        batch_size = self.model_training_params["batch_size"]
        device = self.device
        optimizer = self.optimizer
        latent_dim = self.latent_dim
        input_dim = self.input_dim
        dim_x = self.dim_x
        dim_y = self.dim_y
        nx = self.nx
        ny = self.ny
        model_training_dir = self.model_training_dir
        exp_logger = self.tsb_logger

        training_stop_metric_threshold = self.model_training_params[
            "training_stop_metric_threshold"
        ]
        optim_lr_scheduler = self.optim_lr_scheduler

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

            train_hist["per_epoch_ptimes"] = []  # per epoch training time duration
            train_hist["total_ptime"] = None  # total epochs training time
            train_hist["best_val_metric"] = None  # final best validation metric reached
            best_val_metric = float("inf")
        else:
            train_hist = self.train_history
            best_val_metric = train_hist["best_val_metric"]

        ## add network graph to tenserboard
        with torch.no_grad():
            model.eval()
            with torch.no_grad():
                dummy_x = torch.rand(10, dim_x).to(device)
                dummy_y = torch.rand(10, dim_y).to(device)
                exp_logger.add_graph(model, (dummy_x, dummy_y))

        print("Starting training loop ...")

        start_time = time.time()  # start time of the whole training

        for epoch in range(nb_epochs):
            epoch_start_time = time.time()  # start time of the current epoch

            train_total_loss, train_recon_loss, train_kl_loss = train_cvae(
                model,
                self.training_x,
                self.training_y,
                optimizer=optimizer,
                batch_size=batch_size,
                beta=beta,
                device=device,
            )

            if lr_scheduling:
                if optim_params["lr_scheduler"] == "one_cycle":  # after batch training
                    optim_lr_scheduler.step()

            val_total_loss, val_recon_loss, val_kl_loss = validate_cvae(
                model,
                self.validation_x,
                self.validation_y,
                batch_size=batch_size,
                beta=beta,
                device=device,
            )
            self.plot_reconstruction_check(epoch)

            train_hist["train_loss"].append(train_total_loss)
            train_hist["train_reconst"].append(train_recon_loss)
            train_hist["train_latent_dist"].append(train_kl_loss)

            train_hist["validation_loss"].append(val_total_loss)
            train_hist["validation_reconst"].append(val_recon_loss)
            train_hist["validation_latent_dist"].append(val_kl_loss)

            if val_total_loss < (best_val_metric - 1e-4):
                # store model checkpoint when improvement on validation metric is seen or every 100 epochs
                best_val_metric = val_total_loss
                train_hist["best_val_metric"] = best_val_metric
                self.best_training_epoch = epoch + epoch_offset

                # save model checkpoint
                self.total_training_epochs = nb_epochs + epoch_offset
                self.train_history = train_hist
                self.optimizer = optimizer  # save optimizer state
                self.optim_lr_scheduler = optim_lr_scheduler

                self.save_checkpoint(epoch + epoch_offset, save_full_experiment=True)

            if not (epoch + epoch_offset) % 100:
                # save model checkpoint every 100 epochs regardless of improvement
                self.train_history = train_hist
                self.total_training_epochs = nb_epochs + epoch_offset
                self.optimizer = optimizer  # save optimizer state
                self.optim_lr_scheduler = optim_lr_scheduler

                self.save_checkpoint(epoch + epoch_offset, save_full_experiment=True)

            if best_val_metric < training_stop_metric_threshold:
                print(
                    f"Training finished at epoch {epoch + epoch_offset}, reaching loss of {best_val_metric:.5f}"
                )
                break

            if lr_scheduling:
                if (
                    optim_params["lr_scheduler"] == "on_plateau"
                ):  # after validation evaluation
                    optim_lr_scheduler.step(val_total_loss)

            epoch_end_time = time.time()
            # end time of the current epoch. Includes validation time.
            per_epoch_ptime = epoch_end_time - epoch_start_time

            print(
                f"Epoch {epoch + epoch_offset} - epoch time :{per_epoch_ptime:.2f} s, train_loss: {train_total_loss:.3f}, "
                f"val_loss: {val_total_loss:.3f}, train_recon: {train_recon_loss:.3f}, "
                f"val_recon: {val_recon_loss:.3f}, train_latent_dist: {train_kl_loss:.3f},"
                f"val_latent_dist: {val_kl_loss:.3f}"
            )

            # log to tensorboard
            exp_logger.add_scalar("Loss/train", train_total_loss, epoch + epoch_offset)
            exp_logger.add_scalar("Loss/val", val_total_loss, epoch + epoch_offset)
            exp_logger.add_scalar(
                "Recons/train", train_recon_loss, epoch + epoch_offset
            )
            exp_logger.add_scalar("Recons/val", val_recon_loss, epoch + epoch_offset)
            exp_logger.add_scalar(
                "Latent distance/train", train_kl_loss, epoch + epoch_offset
            )
            exp_logger.add_scalar(
                "Latent distance/val", val_kl_loss, epoch + epoch_offset
            )

            exp_logger.add_scalar(
                f"Best val total loss metric", best_val_metric, epoch + epoch_offset
            )
            train_hist["per_epoch_ptimes"].append(per_epoch_ptime)

        end_time = time.time()  # end time of the whole training
        total_ptime = end_time - start_time
        train_hist["total_ptime"] = total_ptime
        train_hist["best_val_metric"] = best_val_metric

        print("Training finished!... save training results \n")
        print("Last epoch stats: \n")
        print(
            f"Epoch {epoch + epoch_offset} - epoch time :{per_epoch_ptime:.2f} s, train_loss: {train_total_loss:.3f}, "
            f"val_loss: {val_total_loss:.3f}, train_recon: {train_recon_loss:.3f}, "
            f"val_recon: {val_recon_loss:.3f}, train_latent_dist: {train_kl_loss:.3f},"
            f"val_latent_dist: {val_kl_loss:.3f}"
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
        self.optim_lr_scheduler = optim_lr_scheduler

        self.save_checkpoint(epoch + epoch_offset, save_full_experiment=True)

        with open(f"{model_training_dir}/train_hist", "wb") as f:
            pickle.dump(train_hist, f)

        exp_logger.flush()
        exp_logger.close()

    def run_inference(
        self, observation_vec, noise_label, sample_size=1000, on_test_set=False
    ):
        print("Running cVAE inference ...\n")

        inference_results = {}

        source_x = (
            self.test_x.view(-1, self.dim_x)
            if on_test_set
            else self.validation_x.view(-1, self.dim_x)
        )

        for observation_idx in observation_vec:
            print(f"Running inference for observation {observation_idx} ...")
            if observation_idx not in inference_results:
                inference_results[observation_idx] = {}

            # load observation
            observation = self.get_observation(observation_idx, noise_label).view(
                1, self.dim_y
            )
            if self.normalize:
                observation = tdp.normalize(observation, self.normalization_dict_y)

            # generate samples from the model
            inferred_sample = self.model.generate_samples(observation, sample_size)

            if self.normalize:
                inferred_sample = tdp.un_normalize(
                    inferred_sample, self.normalization_dict_x
                )

            inference_results[observation_idx]["samples"] = inferred_sample
            inference_results[observation_idx]["ground_truth"] = source_x[
                observation_idx, :
            ].view(1, self.dim_x)

        return inference_results

    def update_noise_dicts(self, noise_list):
        """
        Update the noise dictionary.
        :param noise_list: a list of noise labels to update the noise dictionary with.
        """
        _map_noise_to_dict_idx = {
            "small_gauss": 0,
            "large_gauss": 1,
            "gumbel": 2,
        }
        if self.noise_dicts is None:
            self.noise_dicts = {}
        for noise_label in noise_list:
            self.noise_dicts[noise_label] = self.config.noises_list[
                _map_noise_to_dict_idx[noise_label]
            ]

    def inference_diagnostics(
        self,
        observation_vec,
        noises_list,
        on_test_set=False,
        inf_sample_size=1000,
        bootstraps=10,
    ):
        """
        Run inference diagnostics for a list of noise labels.
        :param observation_vec: vector of observation indices to run inference for
        :param noises_list: list of noise labels to run inference for
        :param on_test_set: whether to run inference for observastion from the test set or from the validation set
        :return: dictionary with inference results for each noise label
        """
        from fastabc_inversion.geo_problems.Inference_Diagnostics import \
            save_to_disk

        m = inf_sample_size
        self.inverted_obs_idx = observation_vec
        self.update_noise_dicts(noises_list)

        if self.all_obs_inference_results is None:
            self.all_obs_inference_results = {}

        for noise_label in noises_list:
            self.all_obs_inference_results[noise_label] = self.run_inference(
                self.inverted_obs_idx,
                noise_label,
                sample_size=m,
                on_test_set=on_test_set,
            )

        save_to_disk(
            self.all_obs_inference_results,
            f"{self.inference_dir}/all_obs_inference_results.pkl",
        )
        save_to_disk(
            self.inverted_obs_idx,
            f"{self.inference_dir}/inverted_obs_idx.txt",
            _pickle=False,
            _text=True,
        )

        # compute metrics
        from fastabc_inversion.geo_problems.Inference_Diagnostics import \
            compute_array_stats
        from fastabc_inversion.geo_problems.utils.evaluation.scorers import (
            torch_es, torch_rmse, torch_vs)

        metrics = {}
        inference_metrics_stats = {}
        metrics["rmse"] = {}
        metrics["es"] = {}
        metrics["es"][1] = {}
        metrics["es"][2] = {}
        metrics["vs"] = {}
        metrics["vs"][0.5] = {}

        for noise_label in noises_list:
            metrics["rmse"][noise_label] = []
            metrics["es"][1][noise_label] = []
            metrics["es"][2][noise_label] = []
            metrics["vs"][0.5][noise_label] = []
            inference_metrics_stats[noise_label] = {}

            for observation_idx in self.inverted_obs_idx:
                print(
                    f"Computing metrics for observation {observation_idx} with noise label {noise_label} ..."
                )
                inverted_x = self.all_obs_inference_results[noise_label][
                    observation_idx
                ]["samples"]
                ref_x = self.all_obs_inference_results[noise_label][observation_idx][
                    "ground_truth"
                ]

                # compute RMSE
                metrics["rmse"][noise_label].extend(
                    torch_rmse(ref_x, inverted_x, on_gpu=True)
                )

                # Bootstrap ES (1,2) and VS(0.5)
                for _ in range(bootstraps):
                    if bootstraps == 1:
                        size = m
                        replace_cond = False
                    else:
                        size = m // 3
                        replace_cond = True

                    random_idx = np.random.choice(m, size=size, replace=replace_cond)

                    metrics["es"][1][noise_label].append(
                        torch_es(ref_x, inverted_x[random_idx], 1, on_gpu=True)
                    )
                    metrics["es"][2][noise_label].append(
                        torch_es(ref_x, inverted_x[random_idx], 2, on_gpu=True)
                    )
                    metrics["vs"][0.5][noise_label].append(
                        torch_vs(ref_x, inverted_x[random_idx], 0.5, on_gpu=True)
                    )

            print(f"Computing metrics summaries for noise label {noise_label}.")
            inference_metrics_stats[noise_label]["rmse"] = compute_array_stats(
                np.array(metrics["rmse"][noise_label])
            )
            inference_metrics_stats[noise_label]["es"] = {}
            inference_metrics_stats[noise_label]["es"][1] = compute_array_stats(
                np.array(metrics["es"][1][noise_label])
            )
            inference_metrics_stats[noise_label]["es"][2] = compute_array_stats(
                np.array(metrics["es"][2][noise_label])
            )
            inference_metrics_stats[noise_label]["vs"] = {}
            inference_metrics_stats[noise_label]["vs"][0.5] = compute_array_stats(
                np.array(metrics["vs"][0.5][noise_label])
            )

        save_to_disk(metrics, f"{self.inference_dir}/inference_metrics.pkl")
        save_to_disk(
            inference_metrics_stats,
            f"{self.inference_dir}/inference_metrics_stats.txt",
            _pickle=False,
            _text=True,
        )

        return self.all_obs_inference_results, metrics, inference_metrics_stats

    def plot_posterior_samples(self, obs_vec, k=3, dpi=600, show=False):
        """
        Function to plot posterior samples of the cVAE model from the inference results provided in `inferred_samples`.
        :param inferred_samples: dictionary with inference results for each observation index.
        :param obs_vec: list of observation indices to plot samples for.
        :param k: number of examples to plot for each observation index.
        :param dpi: dots per inch for the saved plots.
        :param show: whether to show the plots or not.
        :return:
        """
        import fastabc_inversion.geo_problems.utils.visualization.plotting_tools as plot
        from fastabc_inversion.geo_problems.utils.evaluation.scorers import \
            torch_rmse

        dim_x = self.dim_x

        nx = self.nx
        ny = self.ny

        if self.all_obs_inference_results is None:
            # verify if .pkl file with inference results exists
            if os.path.exists(f"{self.inference_dir}/all_obs_inference_results.pkl"):
                with open(
                    f"{self.inference_dir}/all_obs_inference_results.pkl", "rb"
                ) as f:
                    self.all_obs_inference_results = pickle.load(f)
            else:
                raise ValueError(
                    "Inference results are not available. Run inference diagnostics first."
                )

        noise_list = list(self.all_obs_inference_results.keys())
        for noise_label in noise_list:
            for obs_idx in obs_vec:
                if obs_idx not in self.all_obs_inference_results[noise_label]:
                    print(
                        f"Observation {obs_idx} is not available in inference results for noise label {noise_label}."
                    )
                    continue
                samples = self.all_obs_inference_results[noise_label][obs_idx][
                    "samples"
                ]
                ref_x = self.all_obs_inference_results[noise_label][obs_idx][
                    "ground_truth"
                ].reshape(1, dim_x)

                if samples.shape[0] < k:
                    print(
                        f"Not enough samples for observation {obs_idx} with noise label {noise_label}. "
                        f"Available: {samples.shape[0]}, requested: {k}."
                    )
                    continue
                else:
                    rand_idx = np.random.choice(samples.shape[0], k, replace=False)
                    selected_samples_x = samples[rand_idx].reshape(k, dim_x)

                    rmse_values = torch_rmse(ref_x, selected_samples_x, on_gpu=True)

                    all_examples_to_plot = np.concatenate(
                        (ref_x.cpu().numpy(), selected_samples_x.cpu().numpy()), axis=0
                    ).reshape(1, k + 1, -1)

                    file_name = f"{self.inference_dir}/obs_{obs_idx}_noise_{noise_label}_samples.pdf"

                    # plot the samples
                    plot.plot_samples(
                        all_examples_to_plot,
                        rmse_labels=[list(rmse_values)],
                        ssim_labels=None,
                        width=nx,
                        height=ny,
                        grd_truth=True,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                        figsize=(15, 6),
                    )
