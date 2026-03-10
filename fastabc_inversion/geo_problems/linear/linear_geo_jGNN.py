"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Script to run jGNN training for the linear geophysics problem.
"""

from fastabc_inversion.geo_problems.jGNN_SuS_exp import (
    jGNN_SuS_exp, load_experiment_from_file)

################################# Configure hyperparameters #################################
exp_id = 80

gpu_id = 1

## Loss
p = 2
batch_size = 300
latent_dist_name = "normal"  # "standardnormal" # "uniform"#
latent_dist_params_list = [0, 0.6]  #  [0, 1] # [-1, 1] #

sinkhorn_epsilon = 100
sinkhorn_regularization_starting_value = 150
sinkhorn_regularization_scaling_factor = (
    -1
)  # 1 means no scaling. 2 (means divide by 2).
# -1 means equilibrate with the reconstruction loss dynamically (recon_loss // sink_loss)
scale_sink_by_dim = False

## Architecture
latent_space_dimension = 30
smooth_encoder = False
encoder_activation = {
    "name": "prelu",
    "num_parameters": 1,
}  # set num_parameters to -1 if num. of channels to be used
decoder_activation = {
    "name": "constrained_prelu",
    "num_parameters": 1,
}  # set num_parameters to -1 if num. of channels to be used

## Training
epochs = 2000
lr_scheduler = "on_plateau"  # None#     'one_cycle'#     # learning rate scheduling
inflate_recon_y = (
    False  # whether to inflate the reconstruction loss by recon_x // recon_y
)
scale_by_dim = False  # whether to scale the norms by the dimension of the data
best_model_metric = "full_loss"  # "recon"#
subset_train = None  # training set size. None : takes all training data (7k or 10k)

# data_prep = None
data_prep = {
    "function": "min_max",  # "min_max" or "standardize"
    "kwargs": {
        "lower_p": 1,
        "upper_p": 99,
    },  # for min_max, provide lower and upper percentiles
    # for standardize, mean and std tensors will be computed in load_data function
}

run_training = True  # Do training once model is created or loaded from checkpoint
## provide an epoch number to import checkpoint from; if set to None, no checkpoint is imported and a new model is trained
checkpoint_epoch = None
########################################################################################

model_selector = "jUnet"  # "jointUnet_noSpectralNorm" #"jointWAE_convolutions"
nn_params = {
    "encoder_input_image_channels": 1,
    "encoder_init_conv_channels": 64,
    "latent_dim": latent_space_dimension,
    "decoder_output_image_channels": 1,
    "decoder_init_conv_channels": 64,
    "use_conv_bias": True,
    "spectral_norm_encoder": smooth_encoder,
    "spectral_norm_decoder": True,
    "activation_dict_encoder": encoder_activation,  # set num_parameters to -1 if num. of channels to be used
    "activation_dict_decoder": decoder_activation,  # set num_parameters to -1 if num. of channels to be used
}

sinkhorn_params = {
    "p": p,
    "epsilon": sinkhorn_epsilon,
    "niter": 100,
    "scale_by_dim": scale_sink_by_dim,
}
sinkhorn_lambda_scheduling_params = {
    "sink_lambda": sinkhorn_regularization_starting_value,  # initial value of lambda, multiplying the sinkhorn part of the loss
    "sink_lambda_scheduler_factor": sinkhorn_regularization_scaling_factor,  # 1 means no scheduling
    "sink_lambda_scheduler_epoch": 100,  #  number of epochs after which to update lambda
}
train_with_recon_only = False  # Whether to train with only reconstruction loss
# or with full loss including latent distribution loss
norms_params = {
    "l_norm_p_x": p,
    "l_norm_p_y": p,
    "norm_fct_type_x": "lpp" if not scale_by_dim else "mse",
    "norm_fct_type_y": "lpp" if not scale_by_dim else "mse",
}
model_training_params = {
    "power_p": p,
    "batch_size": batch_size,
    "nb_epochs": epochs,
    "best_model_metric": best_model_metric,
    "training_stop_metric_threshold": 1e-3,
    "train_with_recon_only": train_with_recon_only,  # whether to train the model with only the reconstruction part of the loss
    "inflate_recon_y": inflate_recon_y,
}

# initialize experiment
parameters_file = "FastABC/data/geo_linear/parameters_matern32_Mu10_Var1p96_CorH30_CorV15_linear_81_10k.txt"
experiment_id = f"_id_{exp_id}"
seed = None  # int(time.time()) # if None, the seed is taken from the parameters file
geo_jGNN_exp = jGNN_SuS_exp(
    parameters_file,
    name="Lin_Geo_jGNN_SuS_Matern_81",
    seed=seed,
    run_id=None,
    gpu_id=gpu_id,
    latent_dist_name=latent_dist_name,
    latent_dist_params_list=latent_dist_params_list,
    name_suffix=experiment_id,
)

print(f"Running experiment: {geo_jGNN_exp.name}")

# load datasets
datasets_to_load = ["train", "validation", "test"]
geo_jGNN_exp.load_data(
    datasets_to_load=datasets_to_load, data_prep=data_prep, subset_train=subset_train
)

# train or import pretrained model
lr_scheduler_verbosity = True
lr = 0.001
betas = (0.9, 0.999)

if lr_scheduler is not None:
    if lr_scheduler == "on_plateau":
        optim_params = {
            "lr": lr,
            "betas": betas,
            "lr_scheduler": lr_scheduler,
            "lr_factor": 0.9,
            "lr_patience": 30,
            "lr_threshold": 1e-4,
            "lr_threshold_mode": "abs",
            "lr_eps": 1e-10,
            "verbose": lr_scheduler_verbosity,
        }
    else:
        optim_params = {
            "lr": lr,
            "betas": betas,
            "lr_scheduler": lr_scheduler,
            "max_lr": lr,
            "epochs": epochs,
            "steps_per_epoch": geo_jGNN_exp.train_size // batch_size,
            "verbose": lr_scheduler_verbosity,
        }
else:
    optim_params = {
        "lr": lr,
        "betas": betas,
    }

lr_scheduling = True if lr_scheduler is not None else False

from_checkpoint = (
    f"{geo_jGNN_exp.model_training_dir}/models/checkpoint_epoch_{checkpoint_epoch}.pth"
    if checkpoint_epoch is not None
    else None
)

model_log_ref = f"linGeo_{model_selector}_{geo_jGNN_exp.run_id}"  # f"linGeo_jointWAE_convolutions_{geo_jGNN_exp.run_id}"

if run_training:
    print("Training model ...")
    geo_jGNN_exp.config_tensorboard_logging(tensorboard_logging_dir_root="../runs")
    geo_jGNN_exp.train_model(
        model_selector=model_selector,
        nn_params=nn_params,
        lr_scheduling=lr_scheduling,
        optim_params=optim_params,
        sinkhorn_params=sinkhorn_params,
        sinkhorn_lambda_scheduling_params=sinkhorn_lambda_scheduling_params,
        model_training_params=model_training_params,
        norms_params=norms_params,
        model_log_ref=model_log_ref,
        continue_training_from_checkpoint=from_checkpoint,
    )

else:
    print("Loading experiment from file...")
    # load pre-trained model from checkpoint
    geo_jGNN_exp = load_experiment_from_file(
        f"{geo_jGNN_exp.model_training_dir}/models/experiment_epoch_{checkpoint_epoch}.pkl"
    )
    geo_jGNN_exp.config_tensorboard_logging(tensorboard_logging_dir_root="../runs")

geo_jGNN_exp.save_experiment()
