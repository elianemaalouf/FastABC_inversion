"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
This file allows to configure and run the conditional generation experiments.
"""

import os
import pickle
import random
import string
import time
from abc import ABC, abstractmethod

import fastabc_inversion.conditional_generation.sinkhorn.sinkhorn_pointcloud as spc
import numpy as np
import torch
import torch.distributions as dists
import torch.optim as optim
import torchinfo
from fastabc_inversion.conditional_generation.nn.clr_torch import CLR
from fastabc_inversion.conditional_generation.utils.utilities import \
    norm_fn_selector
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


def inspect_data(
    dataloader, random_samples=False, grid_size=5, num_batch_to_inspect=1, **kwargs
):
    """
    Inspect a dataloader by printing some statistics and displaying some samples
    dataloader: PyTorch DataLoader object
    random_samples: if True, display random samples; if False, display one sample per class
    grid_size: size of the grid to display samples (grid_size x grid_size) when random_samples is True
    num_batch_to_inspect: number of batches to inspect
    """
    # get data from num_batch_to_inspect batches
    from fastabc_inversion.conditional_generation.utils.plotting import \
        plot_samples_inspections

    images = []
    labels = []
    for i, (img_batch, label_batch) in enumerate(dataloader):
        images.append(img_batch)
        labels.append(label_batch)
        if i + 1 >= num_batch_to_inspect:
            break
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Image shape: {images[0].shape}")
    print(f"Label shape: {labels[0].shape}")
    print(f"Image min/max: {images.min().item()}/{images.max().item()}")
    print(f"Label min/max: {labels.min().item()}/{labels.max().item()}")

    # revert labels to label instead of one_hot, just take max index
    if labels.ndim > 1 and labels.shape[1] > 1:
        labels = torch.argmax(labels, dim=1)

    plot_samples_inspections(
        images=images,
        labels=labels,
        random_samples=random_samples,
        grid_size=grid_size,
        **kwargs,
    )


def jsae_loss_fn(
    original_data,
    reconstructed_data,
    data_codes,
    latent_vector,
    sink_lambda,
    norm_fn_x_dict,
    norm_fn_y_dict,
    sinkhorn_params,
    device,
    cx_weight=1.0,
    cy_weight=1.0,
):
    """
    Computes the JSAE loss function, which includes reconstruction loss,
    and latent distribution loss combined with a scaling factor.

    original_data: Tuple of tensors (x, y) where `x` and `y` represent
                    the original input data for both modalities.
    reconstructed_data: Tuple of tensors (reconstructed_x, reconstructed_y)
                        representing the reconstructed data for the respective
                        modalities.
    data_codes: Tensor representing the encoded latent representations of the data.
    latent_vector: Tensor representing a random latent sample from the latent distribution
    sink_lambda: Weighting factor for the latent distribution loss in
                        the total loss computation.
    norm_fn_x_dict: Dictionary that specifies the norm function for the x modality
                           and its associated `p` parameter.
                           Keys include:
                           - fn: Callable, the normalization function
                           - p: Parameter for the norm function
    norm_fn_y_dict: Dictionary that specifies the norm function for the y modality
                           and its associated `p` parameter.
                           Keys include:
                           - fn: Callable, the normalization function
                           - p: Parameter for the norm function
    sinkhorn_params: Dictionary containing parameters related to Sinkhorn normalization.
                            Keys include:
                            - epsilon: Parameter for Sinkhorn normalization (float)
                            - niter: Number of iterations for Sinkhorn computation (int)
                            - p: Sinkhorn norm exponent (float)
    device: PyTorch device where the tensors will be allocated (e.g., "cpu", "cuda").
    cx_weight: Weighting factor for the reconstruction loss of modality x.
    cy_weight: Weighting factor for the reconstruction loss of modality y.
    :return: Tuple containing:
             - model_loss_batch: total loss combining reconstruction loss and
                                 latent distribution loss.
             - reconst_cost_batch: Reconstruction loss for both modalities.
             - latent_dist_batch: Latent distribution loss calculated using Sinkhorn normalization.
    """
    batch_size = data_codes.shape[0]

    x_batch = original_data[0].view(batch_size, -1)
    y_batch = original_data[1].view(batch_size, -1)

    recon_x = reconstructed_data[0].view(batch_size, -1)
    recon_y = reconstructed_data[1].view(batch_size, -1)

    # Reconstruction loss
    cx_batch = torch.mean(
        norm_fn_x_dict["fn"](
            x_batch,
            recon_x,
            p=norm_fn_x_dict["p"],
        )
    )
    cy_batch = torch.mean(
        norm_fn_y_dict["fn"](
            y_batch,
            recon_y,
            p=norm_fn_y_dict["p"],
        )
    )

    reconst_cost_batch = cx_batch * cx_weight + cy_batch * cy_weight

    # Latent distribution loss
    latent_dist_batch = spc.sinkhorn_normalized(
        latent_vector,
        data_codes,
        sinkhorn_params["epsilon"],
        batch_size,
        sinkhorn_params["niter"],
        sinkhorn_params["p"],
        device=device,
    )

    model_loss_batch = (reconst_cost_batch + sink_lambda * latent_dist_batch) * 2 ** (
        sinkhorn_params["p"] - 1
    )

    return model_loss_batch, reconst_cost_batch, latent_dist_batch, cx_batch, cy_batch


class Experiment(ABC):
    map_ERADist_to_torch_dist = {
        "standardnormal": dists.Normal,
        "normal": dists.Normal,
        "uniform": dists.Uniform,
    }

    def __init__(
        self,
        seed,
        data_dir,
        experiment_rootdir,
        name,
        image_size,
        dim_x,
        dim_y,
        latent_dist_name,
        latent_dist_params_list,
        use_cuda=True,
        gpu_id=0,
    ):
        """
        Initialize the experiment
        seed: random seed for reproducibility
        data_dir: directory where data is stored
        experiment_rootdir: directory where experiment results will be stored
        name: name of the experiment
        image_size: size of the input images (assumed square)
        dim_x: dimension of the input data
        dim_y: dimension of the conditioning data
        latent_dist_name: name of the latent distribution
        latent_dist_params_list: list of parameters for the latent distribution
        use_cuda: whether to use CUDA if available
        gpu_id: GPU id to use if CUDA is available
        """

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

        self.run_id = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=5)
        )

        self.name = f"{name}_{self.run_id}"

        # device

        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and use_cuda) else "cpu"
        )
        self.ngpu = 1 if torch.cuda.is_available() and use_cuda else 0

        self.data_dir = data_dir
        self.experiment_rootdir = experiment_rootdir

        # create experiment directory
        self.experiment_dir = f"{self.experiment_rootdir}/{self.name}"
        os.makedirs(self.experiment_dir, exist_ok=True)

        # create directories for storing models snapshots and logs
        self.model_training_dir = f"{self.experiment_dir}/model_training_data"
        os.makedirs(self.model_training_dir, exist_ok=True)
        os.makedirs(f"{self.model_training_dir}/training_plots", exist_ok=True)
        os.makedirs(f"{self.model_training_dir}/models", exist_ok=True)

        # create directories for storing inference results
        self.inference_dir = f"{self.experiment_dir}/inference_results"
        os.makedirs(self.inference_dir, exist_ok=True)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.train_size = None
        self.val_size = None
        self.test_size = None

        # tensorboard logger
        self.tensorboard_logging_dir_root = f"{experiment_rootdir}/runs"
        os.makedirs(self.tensorboard_logging_dir_root, exist_ok=True)

        # jGNN
        self.latent_dim = None
        self.image_size = image_size
        self.dim_x = image_size * image_size if dim_x is None else dim_x
        self.dim_y = dim_y
        self.input_dim = (
            self.dim_x + self.dim_y
            if self.dim_y is not None and self.dim_x is not None
            else None
        )

        self.model_selector = None
        self.model_log_ref = None
        self.model = None  # full model
        self.netD = None  # encoder
        self.netG = None  # decoder / generator
        self.nn_params = None
        self.model_training_params = None
        self.torch_clr_transformer = CLR()

        self.load_pretrained_model = False
        self.pretrained_model_netD_state_dict = None
        self.pretrained_model_netG_state_dict = None

        self.optimizer = None
        self.lr_scheduling = False
        self.optim_lr_scheduler = None

        self.norms_params = None

        self.sinkhorn_lambda_scheduling_params = None
        self.sinkhorn_params = None

        self.latent_dist_name = latent_dist_name
        self.latent_dist_params_list = latent_dist_params_list
        self.latent_dist = self.map_ERADist_to_torch_dist[self.latent_dist_name]

        self.train_history = None
        self.total_training_epochs = 0

        self.image_transform = None
        self.label_transform = None

        self.g_fun = None
        self.u2x = None

    def config_tensorboard_logging(self):
        """
        Configure tensorboard logging.
        """
        self.tensorboard_logging_dir = (
            f"{self.tensorboard_logging_dir_root}/{self.name}"
        )
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

    def save_checkpoint(self, checkpoint_dict, save_full_experiment=False):
        """
        Save a checkpoint of the experiment
        checkpoint_dict: dictionary containing the checkpoint data to be saved. Should contain at least the epoch number under the key 'epoch'.
        save_full_experiment: if True, save the full experiment object using pickle
        """
        epoch = checkpoint_dict.get("epoch", None)
        if epoch is None:
            raise ValueError(
                "checkpoint_dict must contain the epoch number under the key 'epoch'"
            )

        torch.save(
            checkpoint_dict,
            f"{self.model_training_dir}/models/checkpoint_epoch_{epoch}.pth",
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

    def save_model_summary(self, input_dim):
        """
        Save the model summary to a file in the model training directory
        """
        result = torchinfo.summary(self.model, input_size=input_dim, device=self.device)

        with open(f"{self.model_training_dir}/model_summary.txt", "a+") as f:
            f.write(str(result))

    def save_experiment(self, config_dict=None):
        """
        Saves the experiment configuration to a text file in the experiment directory
        """
        if config_dict is None:
            config_dict = {
                "name": self.name,
                "run_id": self.run_id,
                "seed": self.seed,
                "data_dir": self.data_dir,
                "experiment_rootdir": self.experiment_rootdir,
                "image_size": self.image_size,
                "dim_x": self.dim_x,
                "dim_y": self.dim_y,
                "latent_dim": self.latent_dim,
                "latent distribution name": self.latent_dist_name,
                "latent distribution parameters": self.latent_dist_params_list,
                "Image decoder network": self.model_selector,
                "load pretrained model": self.load_pretrained_model,
                "pretrained model netD state dict": self.pretrained_model_netD_state_dict,
                "pretrained model netG state dict": self.pretrained_model_netG_state_dict,
                "device": self.device,
                "optimizer": self.optimizer,
                "optimizer lr scheduler": self.optim_lr_scheduler,
                "sinkhorn parameters": self.sinkhorn_params,
                "sinkhorn lambda scheduling parameters": self.sinkhorn_lambda_scheduling_params,
                "model training parameters": self.model_training_params,
                "metric spaces norms parameters": self.norms_params,
                "best training epoch": self.best_training_epoch,
            }

        with open(f"{self.experiment_dir}/experiment_config.txt", "w") as f:
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")

    @abstractmethod
    def load_data(self):
        """
        Load the data
        """
        pass

    @abstractmethod
    def construct_model_architecture(self):
        """
        Construct the model architecture
        """
        pass

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

    def lambda_sinkhorn_scheduling(
        self,
        current_lambda,
        sink_lambda_sched_factor,
        running_recon_loss,
        running_latent_dist_loss,
        epoch,
        prec_nugget=1e-10,
    ):
        cycle_steps = len(self.train_loader)  # number of batches per epoch

        p_sk = self.sinkhorn_params["p"]
        sink_lambda_sched_epoch = self.sinkhorn_lambda_scheduling_params[
            "sink_lambda_scheduler_epoch"
        ]

        sink_lambda = current_lambda

        if current_lambda ** (1 / p_sk) > 1:
            old_sink_lambda = current_lambda

            if sink_lambda_sched_factor == -1:
                sink_lambda = (running_recon_loss / cycle_steps) // (
                    (running_latent_dist_loss / cycle_steps) + prec_nugget
                )  # don't use validation data here (risk of contaminating the training with the validation set)
                if sink_lambda == 0:
                    sink_lambda = 1

            if sink_lambda_sched_factor > 1:
                if epoch % sink_lambda_sched_epoch == 0:
                    sink_lambda = max(current_lambda // sink_lambda_sched_factor, 1)
            print(f"Sinkhorn lambda changed from {old_sink_lambda} to {sink_lambda}")

        else:
            # stop reducing sink_lambda below 1
            sink_lambda = 1
            sink_lambda_sched_factor = 1

        return sink_lambda, sink_lambda_sched_factor

    def inflation_parameter(self, metric_1, metric_2, inflation_factor=1.0):
        """
        Compute an inflation parameter based on two metrics and an inflation factor.
        metric_1: First metric value
        metric_2: Second metric value
        inflation_factor: Inflation factor
        :return: Inflation parameter
        """

        if (
            metric_2 > 0 and metric_2 > 1.0
        ):  # avoid division by zero and avoid over-inflating when metric_2 is small
            inflation_param = (metric_1 / metric_2) * inflation_factor
        else:
            inflation_param = 1.0

        if inflation_param < 1.0:
            inflation_param = 1.0

        return inflation_param

    def validate_model(self, sink_lambda, norm_fn_x_dict, norm_fn_y_dict):
        """
        This method evaluates the average values of total loss, reconstruction loss, and latent distribution loss
        over the validation dataset.

        sink_lambda: Regularization parameter for Sinkhorn loss in the jsae_loss_fn.
        norm_fn_x_dict: Dictionary that specifies the norm function for the x modality. See jsae_loss_fn for details.
        norm_fn_y_dict: Dictionary that specifies the norm function for the y modality. See jsae_loss_fn for details.
        :return: A tuple of three values - average total loss, average reconstruction loss,
                 and average latent distribution loss computed over the validation dataset.
        """
        self.model.eval()

        val_total_loss, val_recon_loss, val_latent_dist_loss, cx, cy = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(self.val_loader):
                b_size = x_batch.shape[0]

                # 1. Move data to the appropriate device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # data_batch = torch.cat((x_batch, y_batch), 1)

                # 2. Forward pass: Compute model output
                data_codes_batch = self.model.netD(x_batch, y_batch)

                latent_vector_batch = (
                    self.latent_dist(
                        torch.tensor(
                            self.latent_dist_params_list[0], dtype=torch.float32
                        ),
                        torch.tensor(
                            self.latent_dist_params_list[1], dtype=torch.float32
                        ),
                    )
                    .sample((b_size, self.latent_dim))
                    .to(self.device)
                )

                reconstrcuted_x_batch, reconstrcuted_y_batch = self.model.netG(
                    data_codes_batch
                )

                # 3. CLR transform y_batch and reconstrcuted_y_batch
                y_batch_clr = self.torch_clr_transformer(y_batch)
                reconstrcuted_y_batch_clr = self.torch_clr_transformer(
                    reconstrcuted_y_batch
                )

                # 4. Compute loss
                (
                    model_loss_batch,
                    reconst_cost_batch,
                    latent_dist_batch,
                    cx_batch,
                    cy_batch,
                ) = jsae_loss_fn(
                    [x_batch, y_batch_clr],
                    [reconstrcuted_x_batch, reconstrcuted_y_batch_clr],
                    data_codes_batch,
                    latent_vector_batch,
                    sink_lambda,
                    norm_fn_x_dict,
                    norm_fn_y_dict,
                    self.sinkhorn_params,
                    self.device,
                )

                val_total_loss += model_loss_batch.item()
                val_recon_loss += reconst_cost_batch.item()
                val_latent_dist_loss += latent_dist_batch.item()
                cx += cx_batch.item()
                cy += cy_batch.item()

            total_batches = i + 1

        # return averages
        return (
            val_total_loss / total_batches,
            val_recon_loss / total_batches,
            val_latent_dist_loss / total_batches,
            cx / total_batches,
            cy / total_batches,
        )

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
        Train the model
        """

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

        self.model_training_params = model_training_params
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

            train_hist = self.train_history
            best_val_metric = train_hist["best_val_metric"]
        else:
            train_hist = {}
            train_hist[
                "train_reconst"
            ] = []  # reconstruction part of the loss on training data
            train_hist["train_loss"] = []  # total loss on training data
            train_hist[
                "train_latent_dist"
            ] = []  # distribution distance in latent space - on training data
            train_hist["train_cx_recon"] = []  # X reconstruction error on training data
            train_hist["train_cy_recon"] = []  # Y reconstruction error on training data

            train_hist["validation_loss"] = []  # total loss on validation data
            train_hist[
                "validation_reconst"
            ] = []  # reconstruction part of the loss on validation data
            train_hist[
                "validation_latent_dist"
            ] = []  # distribution distance in latent space - on validation data
            train_hist[
                "validation_cx_recon"
            ] = []  # X reconstruction error on validation data
            train_hist[
                "validation_cy_recon"
            ] = []  # Y reconstruction error on validation

            train_hist["per_epoch_ptimes"] = []  # per epoch training time duration
            train_hist["total_ptime"] = None  # total epochs training time
            train_hist["best_val_metric"] = None  # final best validation metric reached
            best_val_metric = float("inf")

        norm_fn_x = norm_fn_selector(self.norms_params["norm_fct_type_x"])
        norm_fn_x_dict = {"fn": norm_fn_x, "p": self.norms_params["l_norm_p_x"]}

        norm_fn_y = norm_fn_selector(self.norms_params["norm_fct_type_y"])
        norm_fn_y_dict = {"fn": norm_fn_y, "p": self.norms_params["l_norm_p_y"]}

        inflate_cy_recon = self.model_training_params["inflate_recon_y"]

        # setup plotting tools
        import fastabc_inversion.conditional_generation.utils.plotting as plot

        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config(mpl)

        # train NN
        ## generate a fixed vector to plot visual evolution during training
        fixed_z = (
            self.latent_dist(
                torch.tensor(self.latent_dist_params_list[0], dtype=torch.float32),
                torch.tensor(self.latent_dist_params_list[1], dtype=torch.float32),
            )
            .sample((1, self.latent_dim))
            .to(self.device)
        )
        print(f"Fixed latent tensor shape: {fixed_z.shape}")
        print(
            f"Latent distribution: {self.latent_dist.__name__}, "
            f"param 1: {self.latent_dist_params_list[0]}, param 2: {self.latent_dist_params_list[1]}"
        )

        ## add network graph to tenserboard
        with torch.no_grad():
            self.model.eval()
            dummy_input = (
                torch.rand(10, 1, self.image_size, self.image_size).to(
                    self.device
                ),  # Tensor for input_images: (Batch, Channels, Height, Width)
                torch.rand(10, self.dim_y)
                .float()
                .to(
                    self.device
                ),  # Tensor for input_labels: (Batch, Features) with shape (10, 10)
            )
            self.tsb_logger.add_graph(self.model, dummy_input)

        print("Starting training loop ...")

        start_time = time.time()  # start time of the whole training

        sink_lambda = self.sinkhorn_lambda_scheduling_params["sink_lambda"]
        sink_lambda_sched_factor = self.sinkhorn_lambda_scheduling_params[
            "sink_lambda_scheduler_factor"
        ]

        cy_weight = (
            1.0
            if not inflate_cy_recon
            else self.model_training_params["inflation_param_start"]
        )

        for epoch in range(self.model_training_params["nb_epochs"]):
            self.model.train()

            #### Inflation parameters ####
            if epoch > 0:
                sink_lambda, sink_lambda_sched_factor = self.lambda_sinkhorn_scheduling(
                    sink_lambda,
                    sink_lambda_sched_factor,
                    epoch_recon_loss,
                    epoch_latent_dist_loss,
                    epoch,
                )

                if inflate_cy_recon:
                    cy_weight = self.inflation_parameter(
                        avg_training_cx_recon, avg_training_cy_recon
                    )

            ###################################

            (
                epoch_total_loss,
                epoch_recon_loss,
                epoch_latent_dist_loss,
                epoch_cx_recon,
                epoch_cy_recon,
            ) = (0.0, 0.0, 0.0, 0.0, 0.0)

            # choose what to store when storing a checkpoint for the whole experiment
            checkpoint_data = {
                "epoch": epoch + epoch_offset,
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

            epoch_start_time = time.time()  # start time of the current epoch

            for i, (x_batch, y_batch) in enumerate(self.train_loader):
                b_size = x_batch.shape[0]

                # 1. Move data to the appropriate device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # data_batch = torch.cat((x_batch, y_batch), 1)

                # 2. Zero the gradients from the previous iteration
                self.model.zero_grad()

                # 3. Forward pass: Compute model output
                data_codes_batch = self.model.netD(x_batch, y_batch)

                latent_vector_batch = (
                    self.latent_dist(
                        torch.tensor(
                            self.latent_dist_params_list[0], dtype=torch.float32
                        ),
                        torch.tensor(
                            self.latent_dist_params_list[1], dtype=torch.float32
                        ),
                    )
                    .sample((b_size, self.latent_dim))
                    .to(self.device)
                )

                reconstrcuted_x_batch, reconstrcuted_y_batch = self.model.netG(
                    data_codes_batch
                )

                # 3. CLR transform y_batch and reconstrcuted_y_batch
                y_batch_clr = self.torch_clr_transformer(y_batch)
                reconstrcuted_y_batch_clr = self.torch_clr_transformer(
                    reconstrcuted_y_batch
                )

                # 4. Compute loss
                (
                    model_loss_batch,
                    reconst_cost_batch,
                    latent_dist_batch,
                    cx_batch,
                    cy_batch,
                ) = jsae_loss_fn(
                    [x_batch, y_batch_clr],
                    [reconstrcuted_x_batch, reconstrcuted_y_batch_clr],
                    data_codes_batch,
                    latent_vector_batch,
                    sink_lambda,
                    norm_fn_x_dict,
                    norm_fn_y_dict,
                    self.sinkhorn_params,
                    self.device,
                    cy_weight=cy_weight,
                )

                # 5. Backward pass: Compute gradients
                model_loss_batch.backward()

                # 6. Update weights
                self.optimizer.step()

                if self.lr_scheduling:
                    if (
                        self.optim_params["lr_scheduler"] == "one_cycle"
                    ):  # after batch training
                        self.optim_lr_scheduler.step()

                # 7. Update running losses
                epoch_total_loss += model_loss_batch.item()
                epoch_recon_loss += reconst_cost_batch.item()
                epoch_latent_dist_loss += latent_dist_batch.item()
                epoch_cx_recon += cx_batch.item()
                epoch_cy_recon += cy_batch.item()

            # average training losses per epoch (over all batches)
            avg_training_loss = epoch_total_loss / (i + 1)
            avg_training_reconst = epoch_recon_loss / (i + 1)
            avg_training_latent_dist = epoch_latent_dist_loss / (i + 1)
            avg_training_cx_recon = epoch_cx_recon / (i + 1)
            avg_training_cy_recon = epoch_cy_recon / (i + 1)

            train_hist["train_loss"].append(avg_training_loss)
            train_hist["train_reconst"].append(avg_training_reconst)
            train_hist["train_latent_dist"].append(avg_training_latent_dist)
            train_hist["train_cx_recon"].append(avg_training_cx_recon)
            train_hist["train_cy_recon"].append(avg_training_cy_recon)

            # Run evaluation on validation set at end of every epoch
            (
                avg_val_loss,
                avg_val_reconst,
                avg_val_latent_dist,
                avg_val_cx,
                avg_val_cy,
            ) = self.validate_model(sink_lambda, norm_fn_x_dict, norm_fn_y_dict)

            train_hist["validation_loss"].append(avg_val_loss)
            train_hist["validation_reconst"].append(avg_val_reconst)
            train_hist["validation_latent_dist"].append(avg_val_latent_dist)
            train_hist["validation_cx_recon"].append(avg_val_cx)
            train_hist["validation_cy_recon"].append(avg_val_cy)

            # store checkpoint when improvement on validation metric is seen or every 100 epochs
            if avg_val_loss < (best_val_metric - 1e-4):
                best_val_metric = avg_val_loss
                train_hist["best_val_metric"] = best_val_metric
                self.best_training_epoch = epoch + epoch_offset

                # update experiment attributes
                self.total_training_epochs = (
                    epoch + epoch_offset
                )  # actual number of training epochs completed
                self.train_history = train_hist
                self.sinkhorn_lambda_scheduling_params["sink_lambda"] = sink_lambda
                self.sinkhorn_lambda_scheduling_params[
                    "sink_lambda_scheduler_factor"
                ] = sink_lambda_sched_factor

                self.save_checkpoint(checkpoint_data, save_full_experiment=True)

            if not (epoch + epoch_offset) % 100:
                # update experiment attributes
                self.total_training_epochs = (
                    epoch + epoch_offset
                )  # actual number of training epochs completed
                self.train_history = train_hist
                self.sinkhorn_lambda_scheduling_params["sink_lambda"] = sink_lambda
                self.sinkhorn_lambda_scheduling_params[
                    "sink_lambda_scheduler_factor"
                ] = sink_lambda_sched_factor

                self.save_checkpoint(checkpoint_data, save_full_experiment=True)

            if (
                best_val_metric
                < self.model_training_params["training_stop_metric_threshold"]
            ):
                print(
                    f"Training finished at epoch {epoch + epoch_offset}, reaching loss of {best_val_metric:.5f}"
                )
                break

            if self.lr_scheduling:
                if (
                    self.optim_params["lr_scheduler"] == "on_plateau"
                ):  # after validation evaluation
                    self.optim_lr_scheduler.step(avg_val_loss)

            epoch_end_time = (
                time.time()
            )  # end time of the current epoch. Includes validation time.
            per_epoch_ptime = epoch_end_time - epoch_start_time

            train_hist["per_epoch_ptimes"].append(per_epoch_ptime)

            print(
                f"Epoch {epoch + epoch_offset} - epoch time :{per_epoch_ptime:.2f} s, train_loss: {avg_training_loss:.3f}, "
                f"val_loss: {avg_val_loss:.3f}, train_recon: {avg_training_reconst:.3f}, "
                f"val_recon: {avg_val_reconst:.3f}, train_latent_dist: {avg_training_latent_dist:.3f},"
                f"val_latent_dist: {avg_val_latent_dist:.3f}, cx_train:{avg_training_cx_recon:.3f}, "
                f"cy_train:{avg_training_cy_recon:.3f}, cx_val:{avg_val_cx:.3f}, cy_val:{avg_val_cy:.3f}. \n"
            )

            # log to tensorboard
            self.tsb_logger.add_scalar(
                "Loss/train", avg_training_loss, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar("Loss/val", avg_val_loss, epoch + epoch_offset)
            self.tsb_logger.add_scalar(
                "Recons/train", avg_training_reconst, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar(
                "Recons/val", avg_val_reconst, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar(
                "Latent distance/train", avg_training_latent_dist, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar(
                "Latent distance/val", avg_val_latent_dist, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar(
                "Recons/Cx_recon_train", avg_training_cx_recon, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar(
                "Recons/Cx_recon_val", avg_val_cx, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar(
                "Recons/Cy_recon_train", avg_training_cy_recon, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar(
                "Recons/Cy_recon_val", avg_val_cy, epoch + epoch_offset
            )
            self.tsb_logger.add_scalar(
                "Sinkhorn_lambda", sink_lambda, epoch + epoch_offset
            )
            # log the current learning rate if lr scheduling is used
            if self.lr_scheduling:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.tsb_logger.add_scalar(
                    "Learning_rate", current_lr, epoch + epoch_offset
                )

            # check fixed noise results
            self.model.eval()
            with torch.no_grad():
                output = self.model.netG(fixed_z.to(self.device))
                fixed_x, fixed_y = output[0].detach().cpu(), output[1].detach().cpu()

                fig, axes = plt.subplots(1, 1)

                axes.imshow(fixed_x.squeeze(), cmap="gray")
                axes.set_title(f"Label: {fixed_y.numpy()}")
                axes.axis("off")
                plt.savefig(
                    f"{self.model_training_dir}/training_plots/model_epoch_{epoch+epoch_offset}.png"
                )
                plt.close()

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
            f"val_latent_dist: {avg_val_latent_dist:.3f} , cx_train:{avg_training_cx_recon:.3f}, "
            f"cy_train:{avg_training_cy_recon:.3f}, cx_val:{avg_val_cx:.3f}, cy_val:{avg_val_cy:.3f}. \n"
        )

        print(
            f"Avg epoch time: {torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])):.2f}, "
            f"total {self.model_training_params['nb_epochs']} epochs, total training time: {total_ptime:.2f}"
        )

        # save models checkpoints/configurations
        train_hist["best_val_metric"] = best_val_metric

        # update experiment attributes
        self.total_training_epochs = (
            epoch + epoch_offset
        )  # actual number of training epochs completed
        self.train_history = train_hist
        self.sinkhorn_lambda_scheduling_params["sink_lambda"] = sink_lambda
        self.sinkhorn_lambda_scheduling_params[
            "sink_lambda_scheduler_factor"
        ] = sink_lambda_sched_factor

        self.save_checkpoint(checkpoint_data, save_full_experiment=True)

        with open(f"{self.model_training_dir}/train_hist", "wb") as f:
            pickle.dump(train_hist, f)

        self.tsb_logger.flush()
        self.tsb_logger.close()
