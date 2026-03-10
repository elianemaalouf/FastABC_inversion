"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Script to generate data for SuS inference sensitivity assessments
"""
import os
import pickle
import time

import fastabc_inversion.geo_problems.Inference_Diagnostics as diags
import numpy as np
import scipy.stats as ss
import torch.distributions as dists
from fastabc_inversion.geo_problems.jGNN_SuS_exp import \
    load_experiment_from_file


def make_new_noisy_test_data(idx_vec, noiseless_y, noise_dict, noisy_tt_folder):
    """
    Create new noisy test data and save it to disk
    :param idx_vec: indices of test samples to create noisy data for
    :param noiseless_y: noiseless test data (tensor)
    :param noise_dict: dictionary defining the noise distribution (keys: 'distribution', 'location', 'scale')
    :param noisy_tt_folder: folder to save noisy data to.
    :return:
    """
    distribution = noise_dict["distribution"]
    location = noise_dict["location"]
    scale = noise_dict["scale"]

    if distribution == "Gaussian":
        if scale != 0.0:
            dist = dists.Normal(loc=location, scale=scale)
    else:
        raise ValueError("Unsupported noise distribution: {}".format(distribution))

    for idx in idx_vec:
        noiseless_sample = noiseless_y[idx].numpy()
        noise = dist.sample(noiseless_sample.shape).numpy() if scale != 0.0 else 0.0
        noisy_sample = noiseless_sample + noise

        with open(noisy_tt_folder + "/noisy_tt_vec{}".format(idx), "wb") as f:
            pickle.dump(noisy_sample, f)

        with open(noisy_tt_folder + "/noise_configuration", "w") as f:
            f.write(str(noise_dict))


#### Configure SuS validation procedure parameters here ####
# Load jGNN experiment
epoch = 1999  # 1291
experiment_folder_name = "NonLin_Geo_jGNN_SuS_Matern_81_id_91_5OV4I"  # "Lin_Geo_jGNN_SuS_Matern_81_id_8010k_5OV4I"
dir_root = "../Geo_jGNN_SuS_experiments"
model_training_dir = f"{dir_root}/{experiment_folder_name}/model_training_data"
geo_jGNN_exp = load_experiment_from_file(
    f"{model_training_dir}/models/experiment_epoch_{epoch}.pkl"
)

test_x = geo_jGNN_exp.test_x
test_y = geo_jGNN_exp.test_y  # noiseless data

data_folder_root = geo_jGNN_exp.config.data_folder_location

validation_idx_vec = [
    143,
    310,
    489,
    729,
    118,
    134,
    732,
    195,
    280,
    104,
    750,
    73,
    186,
    558,
    384,
    291,
    234,
    494,
    58,
    775,
    685,
    754,
    248,
    643,
    36,
    952,
    325,
    230,
    129,
    877,
    35,
    644,
    337,
    417,
    341,
    727,
    116,
    586,
    317,
    410,
    9,
    669,
    431,
    905,
    802,
    824,
    774,
    177,
    284,
    336,
    163,
    142,
    52,
    11,
    519,
    353,
    554,
    301,
    396,
    653,
]  # sensitivity assessment observations

inference_params = {
    "N": 1000,
    "p0": 0.1,
    "epsilon_vec": [0.01],
    "norm_fct": "l1",  #'l2' (SSE) or 'l1' (SAE)
    "max_it": 20,  # 30, # 50
}
sus_runs = 1
geo_jGNN_exp.inference_params = {}

# make validation noises
# Prior : Gaussian
# three cases: noiseless, median scale noise and 95th percentile scale noise from invgamma distribution
alpha = 1  # 1.5
beta = 5  # 2.0
noise_scales = {
    "noiseless": 0.0,
    "median": np.sqrt(ss.invgamma.ppf(0.5, a=alpha, scale=beta)),  # ~ 2.7
    "p95": np.sqrt(ss.invgamma.ppf(0.95, a=alpha, scale=beta)),
}  # ~ 9.9
noise_list = list(noise_scales.keys())

run_inference = (
    False  # if True, run inference, otherwise reload previous results from file
)
make_diagnostics_data = (
    True  # whether to create new diagnostics data (otherwise reload from file)
)
load_inverted_x_y = True  # whether to load inverted x and y from file, if False it runs z samples through the decoder to get x and y
make_vs_training_refs = False  # whether to make validation examples vs training reference statistics (only runs if make_diagnostics_data is True)

assess_diagnostics = True  # whether to run diagnostics on the inference results
diags_to_run = [
    "pf",
    "scores",
    "rmse",
    "agg_diags",
]  # diagnostics to run, if empty runs all diagnostics
diags_to_load = (
    []
)  # diagnostics to load if not running them (only used if assess_diagnostics is True)

assess_composite_sample = False  # whether to make a sample at a given threshold and assess it with various metrics
load_composite_sample = False  # whether to load a previously made composite sample, otherwise makes a new one
plot_posterior_samples = True  # runs only when assess_composite_sample is True

assess_threshold_posterior_percentile = False
#####################################################################################

all_obs_results = {}
for noise_label, noise_scale in noise_scales.items():
    geo_jGNN_exp.noise_dicts[noise_label] = {
        "distribution": "Gaussian",
        "location": 0,
        "scale": noise_scale,
    }
    geo_jGNN_exp.update_inference_params(inference_params, noise_label)

    if run_inference:
        # check if noisy data already exists, otherwise create it.
        # treat noiseless case in the same way for simplicity
        noisy_tt_folder = (
            data_folder_root
            + "/noisy_ttvec_Gaussian_loc0_scale{}".format(
                str(noise_scale).replace(".", "p")
            )
        )
        if not os.path.isdir(noisy_tt_folder):
            print("Creating noisy data in folder: {}".format(noisy_tt_folder))
            os.makedirs(noisy_tt_folder, exist_ok=True)
            make_new_noisy_test_data(
                validation_idx_vec,
                test_y,
                geo_jGNN_exp.noise_dicts[noise_label],
                noisy_tt_folder,
            )

        time_start = time.time()
        ## run inference
        all_obs_results = geo_jGNN_exp.run_sus_inference_all_observations(
            observation_vec=validation_idx_vec,
            sus_runs=sus_runs,
            noise_label=noise_label,
            return_full_results=True,
        )
        end_time = time.time()
        print(
            f"Inference for {len(validation_idx_vec)} observation ran in {end_time - time_start} seconds."
        )

    else:
        print("Reloading previous inference results from file.")
        all_obs_results = geo_jGNN_exp.load_inference_results(
            validation_idx_vec, noise_label, all_obs_results
        )
        print(f"Loaded inference results for observations {validation_idx_vec}")

inference_diags = diags.InferenceDiagnostics(geo_jGNN_exp)

p = 2 if inference_params["norm_fct"] == "l2" else 1

if make_diagnostics_data:
    if len(all_obs_results) != 0:
        inference_diags.all_obs_results = all_obs_results
        if inference_diags.all_thresholds_inverted_z is None:
            inference_diags.build_diagnostics_data(
                load_inverted_x_y=load_inverted_x_y, p=p
            )
    else:
        raise ValueError("Please load inference results first.")

    if make_vs_training_refs:
        # make vs trainig references -- will overwrite previous ones if they exist
        metrics = {
            "rmse": {},
            "es_1": {"power": 1},
            "vs": {"power": 0.5},
            "es_2": {"power": 2},
        }
        inference_diags.make_vs_training_refs(
            validation_idx_vec, metrics, load_existing=False
        )

if assess_diagnostics:
    # run listed diagnostics in diags_to_run

    for noise_label in noise_scales.keys():
        for epsilon in inference_params["epsilon_vec"]:
            if "pf" in diags_to_run:
                print("Running probability of failure plottings...")
                inference_diags.get_all_obs_P_f_vs_threshold(
                    noise_label,
                    epsilon,
                    log_y=False,
                    log_x=False,
                    use_true_thresh=True,
                    prob_thresh=0.1,
                    smoothness=0.3,
                )

    # make ES and VS stats and plots
    if "scores" in diags_to_run:
        print("Running scores diagnostics...")
        inference_diags.make_scores_stats(
            es_p=[1, 2],
            vs_p=0.5,
            bootstraps=10,
            bootstrap_replace=True,
            make_plots=True,
        )
    else:
        if "scores" in diags_to_load:
            inference_diags.load_diags(["scores"])

    if "rmse" in diags_to_run:
        print("Running RMSE diagnostics...")
        inference_diags.make_rmse_stats(
            plot_samples=False,
            random_samples=False,
            make_boxplots=False,
            compute_ssim=False,
        )
    else:
        if "rmse" in diags_to_load:
            inference_diags.load_diags(["rmse"])

    if "agg_diags" in diags_to_run:
        print("Running aggregated diagnostics...")

        # vs training references
        rmse_x_refs_val = {
            "train": {"lower": 1.546, "center": 1.864, "upper": 2.266},
        }
        es_1_x_refs_val = {
            "train": {"lower": 1198.761, "center": 1524.401, "upper": 1874.821},
        }
        es_2_x_refs_val = {
            "train": {"lower": 2133.927, "center": 3601.982, "upper": 5279.313},
        }
        vs_05_x_refs_val = {
            "train": {"lower": 642842.875, "center": 685893.437, "upper": 916489.156},
        }

        refs_dict_x_val = {
            "rmse": rmse_x_refs_val,
            "es": {1: es_1_x_refs_val, 2: es_2_x_refs_val},
            "vs": {0.5: vs_05_x_refs_val},
        }

        inference_diags.agg_obs_metrics(
            min_prob_thresh=-13, make_plots=True, refs_dict_x=refs_dict_x_val
        )

# make composite samples
if assess_composite_sample:
    if load_composite_sample:
        # load previously made composite sample
        (
            composite_samples,
            threshold_ids,
            composite_sample_metrics,
        ) = inference_diags.load_composite_samples()
        if composite_sample_metrics is None:
            (
                composite_sample_metrics,
                composite_sample_metrics_summaries,
            ) = inference_diags.compute_composite_sample_metrics(
                composite_samples, noise_list, es_p=[1, 2], vs_p=0.5, m=500
            )
    else:
        # make new composite sample
        threshold_ids = [3]
        composite_samples = inference_diags.make_composite_samples(
            validation_idx_vec, noise_list, threshold_ids
        )
        (
            composite_sample_metrics,
            composite_sample_metrics_summaries,
        ) = inference_diags.compute_composite_sample_metrics(
            composite_samples, noise_list, es_p=[1, 2], vs_p=0.5, m=500
        )

    if plot_posterior_samples:
        inference_diags.plot_posterior_samples(
            composite_samples,
            composite_sample_metrics,
            validation_idx_vec,
            pick_from_min_rmse_x=True,
            pick_from_min_rmse_y=True,
            min_rmse_number=50,
        )  # will pick from min_rmse_number* 3 examples at min rmse for x and/or y

if assess_threshold_posterior_percentile:
    threshold_id = None  # corresponding to log_10(P_f), None takes all the list of thresholds that were used
    (
        percentiles,
        percentils_stats,
    ) = inference_diags.assess_threshold_posterior_percentiles(threshold_id)
