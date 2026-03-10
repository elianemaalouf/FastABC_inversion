"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Script to run SuS inference on non linear geophysics problem
"""

import time

import fastabc_inversion.geo_problems.Inference_Diagnostics as diags
from fastabc_inversion.geo_problems.jGNN_SuS_exp import \
    load_experiment_from_file

epoch = 1999
# experiment_id = 91
experiment_folder_name = "NonLin_Geo_jGNN_SuS_Matern_81_id_91_5OV4I"
dir_root = "../Geo_jGNN_SuS_experiments"
model_training_dir = f"{dir_root}/{experiment_folder_name}/model_training_data"
geo_jGNN_exp = load_experiment_from_file(
    f"{model_training_dir}/models/experiment_epoch_{epoch}.pkl"
)

# correct the latent distribution name (mainly relevant for old experiments using a normal that is not standard normal)
if geo_jGNN_exp.latent_dist_name == "standardnormal":
    geo_jGNN_exp.latent_dist_name = "normal"

old_experiment = False  # to correct for old experiments params

noise_list = [
    "small_gauss",
    "large_gauss",
]  # "gumbel"] ,  the noise types to evaluate the inference on

load_inference_results = (
    True  # whether to load inference results from file. Runs new inference if False.
)
run_diagnostics = True  # whether to run diagnostics on the inference results
load_inverted_x_y = True  # whether to load inverted x and y from file, if False it runs z samples through the decoder to get x and y
load_composite_samples = True  # whether to load composite samples from file, if False it recomposes the samples from inference results and thresholds_ids
benchmark_inference = True  # whether to benchmark inference against other methods

if not load_inference_results or old_experiment:
    geo_jGNN_exp.noise_dicts = {}  # to correct for old experiments
    geo_jGNN_exp.noise_dict = None  # to correct for old experiments
    geo_jGNN_exp.noise = noise_list

geo_jGNN_exp.update_noise_dicts(noise_list)
geo_jGNN_exp.inference_params = {}

# run inference for one or multiple observations and evaluate for each observation
## parameters for inference
inference_root_dir = geo_jGNN_exp.inference_dir

inference_params = {
    "N": 1000,
    "p0": 0.1,
    "epsilon_vec": [0.01],
    "norm_fct": "l1",  #'l2' (SSE) or 'l1' (SAE)
    "max_it": 20,  # 30, # 50
}
sus_runs = 1

return_full_results = True

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
]  # final inference observations

# references from CCA files : reference_inv_metrics_es1_vs05_rmse.json
rmse_x_refs_inv = {
    "train": {"lower": 1.765, "center": 1.917, "upper": 2.076},
}
es_1_x_refs_inv = {
    "train": {"lower": 1839.868, "center": 2101.633, "upper": 2351.541},
}
es_2_x_refs_inv = {
    "train": {"lower": 2231.428, "center": 3441.199, "upper": 4522.294},
}  # computed separately
vs_05_x_refs_inv = {
    "train": {"lower": 1124808.615, "center": 1164289.579, "upper": 1272192.490},
}

refs_dict_x = {
    "rmse": rmse_x_refs_inv,
    "es": {1: es_1_x_refs_inv, 2: es_2_x_refs_inv},
    "vs": {0.5: vs_05_x_refs_inv},
}

all_obs_results = {}
for noise_label in noise_list:
    # set noise label and update noise dictionary

    geo_jGNN_exp.update_inference_params(inference_params, noise_label)

    if not load_inference_results:
        time_start = time.time()
        ## run inference
        all_obs_results = geo_jGNN_exp.run_sus_inference_all_observations(
            observation_vec=observation_idx_vec,
            sus_runs=sus_runs,
            noise_label=noise_label,
            return_full_results=return_full_results,
        )
        end_time = time.time()
        print(
            f"Inference for {len(observation_idx_vec)} observation ran in {end_time - time_start} seconds."
        )
    else:
        # load inference results
        all_obs_results = geo_jGNN_exp.load_inference_results(
            observation_idx_vec, noise_label, all_obs_results
        )
        print(f"Loaded inference results for observations {observation_idx_vec}")


inference_diags = diags.InferenceDiagnostics(geo_jGNN_exp)

to_load = [
    "rmse",
    "scores",
    "agg_diags",
    "pf",
]  # 'rmse', 'scores', 'agg_diags', 'pf', 'post_samples'

if run_diagnostics:
    p = 2 if inference_params["norm_fct"] == "l2" else 1
    if len(all_obs_results) != 0:
        inference_diags.all_obs_results = all_obs_results
        if inference_diags.all_thresholds_inverted_z is None:
            inference_diags.build_diagnostics_data(
                load_inverted_x_y=load_inverted_x_y, p=p
            )
    else:
        raise ValueError("Please load inference results first.")

    for noise_label in noise_list:
        for epsilon in inference_params["epsilon_vec"]:
            # make plot of probability of failure evolution vs threshold and find max curvature
            if "pf" not in to_load:
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
    if "scores" in to_load:
        inference_diags.load_diags(["scores"])
    else:
        inference_diags.make_scores_stats(
            es_p=[1, 2],
            vs_p=0.5,
            bootstraps=10,
            bootstrap_replace=True,
            make_plots=True,
        )

    # makes RMSE stats and plot samples
    if "rmse" in to_load:
        inference_diags.load_diags(["rmse"])
    else:
        inference_diags.make_rmse_stats(
            plot_samples=False,
            random_samples=True,
            make_boxplots=True,
            compute_ssim=False,
        )

    if "agg_diags" not in to_load:
        inference_diags.agg_obs_metrics(
            min_prob_thresh=-13, make_plots=True, refs_dict_x=refs_dict_x
        )

    # make composite samples
    if load_composite_samples:
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
        threshold_ids = [3]
        composite_samples = inference_diags.make_composite_samples(
            observation_idx_vec, noise_list, threshold_ids
        )
        (
            composite_sample_metrics,
            composite_sample_metrics_summaries,
        ) = inference_diags.compute_composite_sample_metrics(
            composite_samples, noise_list, es_p=[1, 2], vs_p=0.5, m=500
        )
    if "post_samples" not in to_load:
        inference_diags.plot_posterior_samples(
            composite_samples,
            composite_sample_metrics,
            observation_idx_vec,
            pick_from_min_rmse_x=True,
            min_rmse_number=100,
        )

# %%
if benchmark_inference:
    # expect composite_samples, threshold_ids, composite_sample_metrics to be already computed by running diagnostics
    # composite_samples, threshold_ids, composite_sample_metrics = inference_diags.load_composite_samples()
    n_obs = len(observation_idx_vec)

    from fastabc_inversion.geo_problems.Inference_Diagnostics import \
        load_from_disk

    methods_metrics = {}

    # load CCA methods results and format them
    cca_metrics = {"CCA": {}}
    cca_metrics_file = "FastABC/data/geo_nonlinear/other_methods_inference_metrics/CCA/inversion_metrics_formatted.pkl"
    cca_metrics["CCA"] = load_from_disk(cca_metrics_file)
    methods_metrics.update(cca_metrics)

    # benchmark inference against other methods
    inference_diags.bench_sus_metrics(
        methods_metrics=methods_metrics,
        n_obs=n_obs,
        refs_dict=refs_dict_x,
        make_plots=True,
    )
