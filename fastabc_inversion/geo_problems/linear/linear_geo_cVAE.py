"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Script to run cVAE training and diagnostics for the linear geophysics problem.
"""
from fastabc_inversion.geo_problems.cVAE_exp import (cVAE_exp,
                                                     load_experiment_from_file)

################################# Configure hyperparameters #################################

exp_id = 4

gpu_id = 0

## Loss
batch_size = 300

beta_kl = 1  # KL divergence weight

## Architecture
latent_space_dimension = 60
model_selector = "cvae"  # "cvae_red_y" #

## Training
epochs = 300
lr_scheduler = "on_plateau"  # None#     'one_cycle'#     # learning rate scheduling
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

nn_params = {
    "encoder_input_image_channels": 1,
    "encoder_init_conv_channels": 64,
    "latent_dim": latent_space_dimension,
    "decoder_output_image_channels": 1,
    "decoder_init_conv_channels": 64,
    "use_conv_bias": True,
}

model_training_params = {
    "batch_size": batch_size,
    "nb_epochs": epochs,
    "training_stop_metric_threshold": 1e-3,
    "beta": beta_kl,
}


# initialize experiment
parameters_file = "FastABC/data/geo_linear/parameters_matern32_Mu10_Var1p96_CorH30_CorV15_linear_81_7k.txt"
experiment_id = f"_id_{exp_id}"
seed = None  # int(time.time()) # if None, the seed is taken from the parameters file
geo_cvae_exp = cVAE_exp(
    parameters_file,
    name="Lin_Geo_cVAE_Matern_81",
    seed=seed,
    run_id=None,
    gpu_id=gpu_id,
    name_suffix=experiment_id,
)

print(f"Running experiment: {geo_cvae_exp.name}")

# load datasets
datasets_to_load = ["train", "validation", "test"]
geo_cvae_exp.load_data(
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
            "steps_per_epoch": geo_cvae_exp.train_size // batch_size,
            "verbose": lr_scheduler_verbosity,
        }
else:
    optim_params = {
        "lr": lr,
        "betas": betas,
    }

lr_scheduling = True if lr_scheduler is not None else False

from_checkpoint = (
    f"{geo_cvae_exp.model_training_dir}/models/checkpoint_epoch_{checkpoint_epoch}.pth"
    if checkpoint_epoch is not None
    else None
)

model_log_ref = f"linGeo_{model_selector}_{geo_cvae_exp.run_id}"

if run_training:
    print("Training model ...")
    geo_cvae_exp.config_tensorboard_logging(tensorboard_logging_dir_root="../runs")
    geo_cvae_exp.train_model(
        model_selector=model_selector,
        nn_params=nn_params,
        lr_scheduling=lr_scheduling,
        optim_params=optim_params,
        model_training_params=model_training_params,
        model_log_ref=model_log_ref,
        continue_training_from_checkpoint=from_checkpoint,
    )

    geo_cvae_exp.save_experiment()


###########################################################################################

# %%
# run inferences
from fastabc_inversion.geo_problems.cVAE_exp import load_experiment_from_file

# run diagnostics
epoch = 299
experiment_id = "id_4"
experiment_folder_name = f"Lin_Geo_cVAE_Matern_81_{experiment_id}_XAJI0"
dir_root = "../Geo_cVAE_experiments"
model_training_dir = f"{dir_root}/{experiment_folder_name}/model_training_data"
geo_cvae_exp = load_experiment_from_file(
    f"{model_training_dir}/models/experiment_epoch_{epoch}.pkl"
)

observation_idx_vec = [
    102,
    106,
    270,
    435,
    860,
    154,
    253,
    309,
    548,
    966,
    385,
    498,
    583,
    608,
    836,
    900,
    10,
    18,
    19,
    20,
    1,
    3,
    45,
    96,
    140,
    157,
    179,
    191,
    204,
    223,
    262,
    269,
    283,
    304,
    305,
    347,
    363,
    379,
    506,
    517,
    521,
    546,
    573,
    607,
    656,
    664,
    671,
    680,
    792,
    801,
]  # 50 obs for inference (final results)

# validation_idx_vec = [143, 310, 489, 729, 118, 134, 732, 195, 280, 104, 750, 73, 186, 558, 384, 291, 234, 494, 58,
#                       775, 685, 754, 248, 643, 36, 952, 325, 230, 129, 877,
#                       35, 644, 337, 417, 341, 727, 116, 586, 317, 410, 9, 669, 431, 905, 802, 824, 774, 177, 284, 336,
#                      163, 142, 52, 11, 519, 353, 554, 301, 396, 653] # 60 obs for validation (inference diagnostics/sensitivity analysis)

(
    all_obs_inference_results,
    metrics,
    inference_metrics_stats,
) = geo_cvae_exp.inference_diagnostics(
    observation_vec=observation_idx_vec,
    noises_list=["small_gauss", "large_gauss", "gumbel"],
    on_test_set=True,
    inf_sample_size=500,
    bootstraps=1,
)
# bootstraps=10 when assessing uncertainty in metrics with validation_idx_vec;
# else bootstraps=1 for the final inference results with observation_idx_vec

geo_cvae_exp.plot_posterior_samples(
    [379, 583, 385]
)  # pick the same examples as the other models for comparison
