"""
ANALYTICAL INVERSION:
--------------------
Performs inversion based on knowing the subsurface slowness field Gaussian covariance, the linear forward and gaussian
uncertainties (measurments uncertainties)
"""
import os
import pickle

import fastabc_inversion.geo_problems.data_simulation.solver_setup as ss
import h5py
import numpy as np
import torch
from fastabc_inversion.geo_problems.utils.config import Config
from fastabc_inversion.geo_problems.utils.generic_fn import (load_from_disk,
                                                             save_to_disk)


def setup_posterior(
    noise_scale,
    d_obs,
    solver_matrix,
    ndata,
    prior_covMat,
    prior_mean,
    ntot,
    verbose=False,
    jitter=0,
):
    """setup_posterior
    Function to setup the Gaussian posterior, the solution to the linear inverse problem.
    It assumes a Gaussian prior, a linear forward and Gaussian noise (only measurement uncertainties, no model uncertainties)
    Based on Tarantola - 2005 - "Inverse Problem Theory and Methods for Model Parameter estimation" (p.64-66)

    :param noise_scale: gaussian noise standard deviation
    :param d_obs: the observation data vector, contaminate with noise
    :param solver_matrix: the linear forward matrix
    :param ndata: the number of rays (sources * receivers)
    :param prior_covMat: Gaussian prior covariance matrix
    :param prior_mean: Gaussian prior mean vector
    :param ntot: 2D field grid total size (width * height)
    :param jitter: a jitter value to add to the covariance matrix diagonal
    :return: a dictionary with the Gaussian posterior distribution configuration : covariance matrix,
    square root of the covariance matrix and mean
    """
    error = noise_scale
    cov_prior = prior_covMat
    m_prior = prior_mean

    cov_noise = np.identity(ndata) * (error**2)
    F_cov_prior = np.dot(solver_matrix, cov_prior)
    cov_prior_F_T = np.dot(cov_prior, np.transpose(solver_matrix))

    cov_y = np.dot(solver_matrix, cov_prior_F_T) + cov_noise
    cov_y_inv = np.linalg.inv(cov_y)

    cov_post = (
        cov_prior
        - np.dot(np.dot(cov_prior_F_T, cov_y_inv), F_cov_prior)
        + np.identity(ntot) * jitter
    )
    L_post = np.linalg.cholesky(cov_post)

    m_post = m_prior + np.dot(
        np.dot(cov_prior_F_T, cov_y_inv), (d_obs - np.dot(solver_matrix, m_prior))
    )

    if verbose:
        print("Refer to [Tarantola, 2005] p. 64-66 for references")
        print("d_obs:", d_obs.shape)
        print("Noise covariance shape: ", cov_noise.shape)
        print("F.Cov_prior shape:", F_cov_prior.shape)
        print("Cov_prior.F^T shape:", cov_prior_F_T.shape)
        print("cov_y shape:", cov_y.shape)
        print("Prior covariace tilde shape:", cov_post.shape)
        print("Posterior mean shape:", m_post.shape)
        print(np.diag(cov_post))
        print(
            "number of negative eigenValues for posterior covariance:",
            sum(np.linalg.eigvalsh(cov_post) < 0),
        )
        print(min(np.linalg.eigvalsh(cov_post)))
        print("Posterior covariace square root shape:", L_post.shape)

    return {
        "posterior_covariance": cov_post,
        "postCov_squareRoot": L_post,
        "posterior_mean": m_post,
    }


def generate_Gauss_samples(set_size, mean, cov, dims):
    """generate_Gauss_samples
    Function to run a loop to generate samples from a gaussian distribution.

    :param set_size: number of samples to generate from the desired distribution
    :param mean: mean of the desired distribution
    :param cov: covariance matrix of the desired distribution
    :param dims: list containing the shape of the desired 3D sample, it assumes the format (number of channels, width, height, width*height*nc)
    :return: returns a 4D numpy array (number of samples, number of channels, width, height)
    """

    if len(dims):
        nc = dims[0]
        nx = dims[1]
        ny = dims[2]
    else:
        return None

    samples = np.random.multivariate_normal(mean, cov, set_size, check_valid="raise")

    # expand dimensions to match the desired shape
    samples = samples.reshape(set_size, nc, ny, nx, order="C")

    return samples


def compute_rmse_grdTruth(samples, ref_model):
    """
    TODO: replace with a function in scorers
    Function to compute RMSE between each element in samples with the reference.
    :param samples: should be an array. Format as [number of samples, size].
    :param ref_model: reference model. Should be a numpy array. If format as [1, size], the reference is broadcasted.
    :return: a 1D numpy array with the RMSE values for each sample. Format as [number of samples, 1].
    """
    rmse_values = np.sqrt(np.mean((samples - ref_model) ** 2, axis=1))

    return rmse_values


def resimulate(samples, solver_type, args, so_file=None):
    """
    Function to run model samples through a forward simulator.
    :param samples: samples of models to run through the forward simulator. Takes in a 4D numpy array (number of samples, number of channels, width, height)
    :param solver_type: specify if 'linear' or 'eikonal-nl'
    :param args: list of all paramters necessary for the function prep_solver()
    :param so_file: if solver_type = 'eikonal-nl', then provide the so-file that goes with it.
    :return: a 2D numpy array (number of samples, ndata)
    """
    size = samples.shape[0]
    width = samples.shape[1]
    height = samples.shape[2]

    solver_setup_dict = ss.prep_solver(solver_type, args)

    if solver_type == "linear":
        solver_matrix = solver_setup_dict["solver_matrix"]
        ndata = solver_setup_dict["ndata"]
        linear_solver = True
        samples = np.squeeze(samples)
        samples = samples.reshape(
            size, width * height
        )  # remove the channel dimension when it is 1
        resimulations = np.dot(solver_matrix, samples.T)

    elif solver_type == "eikonal-nl":
        linear_solver = False
        ndata = solver_setup_dict["ndata"]
    else:
        print("Unknown solver type!")

    if linear_solver:
        return resimulations.T

    else:
        resimulations = np.zeros((size, ndata))

        for i in range(samples.shape[0]):
            model_i = np.squeeze(samples[i, :, :])

            resimulation_i = ss.call_eikonal(solver_setup_dict, model_i, so_file)

            resimulations[i, :] = resimulation_i.reshape(1, -1)

        return resimulations


class AnalyticalInversion:
    """
    Class to perform analytical inversion based on the Gaussian posterior distribution.
    It assumes a linear forward and Gaussian noise (only measurement uncertainties, no model uncertainties).
    """

    def __init__(self, parameters_file):
        self.config = Config(parameters_file)
        self.data_rootdir = self.config.datadir + "/Analytical_Inversions"
        os.makedirs(self.data_rootdir, exist_ok=True)

        self.inversion_dir = f"{self.data_rootdir}/{self.config.cov_kernel}"
        os.makedirs(self.inversion_dir, exist_ok=True)

        self.seed = self.config.parameters["FixedSeed"]  # set by Config

        self.nx = self.config.nx
        self.ny = self.config.ny
        self.nc = self.config.nc
        self.dim_x = self.nx * self.ny
        self.dim_y = self.config.ndata
        self.cov_prior = self.config.CM
        self.m_prior = self.config.m_prior
        self.Forward = self.config.solver_matrix

        _map_noise_to_dict_idx = {
            "small_gauss": 0,
            "large_gauss": 1,
            "gumbel": 2,
        }
        self.noise_dicts = {}

        for noise_label in _map_noise_to_dict_idx.keys():
            self.noise_dicts[noise_label] = self.config.noises_list[
                _map_noise_to_dict_idx[noise_label]
            ]

        self.inverted_obs = None
        self.all_obs_results = None

    def load_data(self):
        test_models_file = h5py.File(
            self.config.data_folder_location + "/test_models.h5"
        )
        test_models = torch.tensor(
            test_models_file.get("test_models"), dtype=torch.float64
        ).numpy()
        self.test_x = test_models.reshape(-1, self.config.nx * self.config.ny)
        test_models_file.close()
        test_truett_file = h5py.File(
            self.config.data_folder_location + "/test_truett_noNoise.h5"
        )
        test_truett_noiseless = torch.tensor(
            test_truett_file.get("test_truett_noNoise"), dtype=torch.float64
        ).numpy()
        self.test_y = test_truett_noiseless.reshape(-1, self.config.rays)
        test_truett_file.close()

        self.test_size = self.test_x.shape[0]

        print("Reference test data loaded")
        print("Test models shape:", self.test_x.shape)
        print("Test travel times shape:", self.test_y.shape)

    def get_observations_folder(self, noise_label):
        """
        Get the folder path where noisy observations are stored based on the noise label.
        :param noise_label: Label of the noise type (e.g., 'small_gauss', 'large_gauss').
        :return:
        """
        noise_distribution = self.noise_dicts[noise_label]["distribution"]
        noise_loc = self.noise_dicts[noise_label]["location"]
        noise_scale = self.noise_dicts[noise_label]["scale"]

        return f"{self.config.data_folder_location}/noisy_ttvec_{noise_distribution}_loc{noise_loc}_scale{str(noise_scale).replace('.', 'p')}"

    def get_observation(self, observation_idx, noise_label):
        """
        Load observation from file.
        :param observation_idx: Index of the observation to load.
        :param noise_label: Label of the noise type (e.g., 'small_gauss', 'large_gauss').
        """
        print(f"Loading noisy observation {observation_idx} from file...")

        observation_path = (
            f"{self.get_observations_folder(noise_label)}/noisy_tt_vec{observation_idx}"
        )

        with open(observation_path, "rb") as f:
            y_obs = pickle.load(f)
        return y_obs

    def invert_obs(self, obs_idx, noise_label, sample_size=500):
        # Run the analytical inversion for the given observation index and noise label.
        print(f"Inverting observation {obs_idx} with noise label '{noise_label}'...")

        y_obs = self.get_observation(obs_idx, noise_label).numpy()
        noise_scale = self.noise_dicts[noise_label]["scale"]
        post_dict = setup_posterior(
            noise_scale,
            y_obs,
            self.Forward,
            self.config.ndata,
            self.cov_prior,
            self.m_prior,
            self.config.ntot,
            jitter=0,
        )
        # Generate samples from posterior distribution
        post_samples = generate_Gauss_samples(
            sample_size,
            mean=post_dict["posterior_mean"],
            cov=post_dict["posterior_covariance"],
            dims=[self.nc, self.nx, self.ny],
        ).reshape(sample_size, self.dim_x)
        # save_to_disk(post_samples, self.inversion_dir, _pickle=True)
        ref_x = self.test_x[obs_idx, :].reshape(
            1, self.dim_x
        )  # Ground truth subsurface model
        return post_samples, ref_x

    def run_inversion(self, obs_idx_vec, resimulate=True):
        """Run the inversion for a vector of observation indices.
        :param
        """
        self.inverted_obs = obs_idx_vec
        self.all_obs_results = {}
        for noise_label in self.noise_dicts.keys():
            if not (noise_label == "small_gauss" or noise_label == "large_gauss"):
                print(
                    f"Skipping noise label '{noise_label}' as it is not supported with analytical inversion."
                )
                continue
            else:
                self.all_obs_results[noise_label] = {}
                for obs_idx in obs_idx_vec:
                    print(
                        f"Inverting observation index {obs_idx} for noise label '{noise_label}'..."
                    )
                    self.all_obs_results[noise_label][obs_idx] = {}
                    post_samples, ref_x = self.invert_obs(obs_idx, noise_label)
                    self.all_obs_results[noise_label][obs_idx]["samples"] = post_samples
                    self.all_obs_results[noise_label][obs_idx]["ground_truth"] = ref_x
                    if resimulate:
                        self.all_obs_results[noise_label][obs_idx]["resim_x"] = np.dot(
                            self.Forward, post_samples.T
                        ).T

        save_to_disk(
            self.all_obs_results,
            f"{self.inversion_dir}/all_obs_results.pkl",
            _pickle=True,
        )
        save_to_disk(
            obs_idx_vec,
            f"{self.inversion_dir}/obs_idx_vec.txt",
            _text=True,
            _pickle=True,
        )
        return self.all_obs_results

    def compute_metrics(self, metrics):
        """
        Compute metrics for the inverted observations.
        :param metrics: Dictionary of metrics to compute. Supported metrics: 'rmse', 'es', 'vs'.
                e.g., = {'rmse':None, 'es':[1,2], 'vs':[0.5]} the values for the keys are the parameters to the metric
        """
        from ThesisCodes.Utiles.evaluation.scorers import (torch_es,
                                                           torch_rmse,
                                                           torch_vs)

        metrics_list = list(metrics.keys())
        if self.all_obs_results is None:
            self.all_obs_results = load_from_disk(
                f"{self.inversion_dir}/all_obs_results.pkl"
            )

        noise_list = list(self.all_obs_results.keys())
        obs_idx_vec = list(self.all_obs_results[noise_list[0]].keys())

        metrics_results = {}

        for noise_label in noise_list:
            metrics_results[noise_label] = {}

            for metric in metrics_list:
                if metric == "rmse":
                    print(f"Computing RMSE for noise label '{noise_label}'...")
                    metrics_results[noise_label][metric] = []
                    for obs_idx in obs_idx_vec:
                        samples = self.all_obs_results[noise_label][obs_idx]["samples"]
                        ref_x = self.all_obs_results[noise_label][obs_idx][
                            "ground_truth"
                        ]
                        rmse_values = torch_rmse(ref_x, samples, on_gpu=True)
                        metrics_results[noise_label][metric].extend(rmse_values)
                elif metric == "es":
                    print(f"Computing ES for noise label '{noise_label}'...")
                    if not isinstance(metrics["es"], list):
                        es_p = [metrics["es"]]
                    else:
                        es_p = metrics["es"]
                    metrics_results[noise_label][metric] = {}
                    for p in es_p:
                        metrics_results[noise_label][metric][p] = []
                        for obs_idx in obs_idx_vec:
                            samples = self.all_obs_results[noise_label][obs_idx][
                                "samples"
                            ]
                            ref_x = self.all_obs_results[noise_label][obs_idx][
                                "ground_truth"
                            ]
                            es_values = torch_es(ref_x, samples, power=p, on_gpu=True)
                            metrics_results[noise_label][metric][p].append(es_values)
                elif metric == "vs":
                    print(f"Computing VS for noise label '{noise_label}'...")
                    if not isinstance(metrics["vs"], list):
                        vs_p = [metrics["vs"]]
                    else:
                        vs_p = metrics["vs"]
                    metrics_results[noise_label][metric] = {}
                    for p in vs_p:
                        metrics_results[noise_label][metric][p] = []
                        for obs_idx in obs_idx_vec:
                            samples = self.all_obs_results[noise_label][obs_idx][
                                "samples"
                            ]
                            ref_x = self.all_obs_results[noise_label][obs_idx][
                                "ground_truth"
                            ]
                            vs_values = torch_vs(ref_x, samples, power=p, on_gpu=True)
                            metrics_results[noise_label][metric][p].append(vs_values)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

        save_to_disk(
            metrics_results,
            f"{self.inversion_dir}/inversion_metrics_results.pkl",
            _pickle=True,
        )
        return metrics_results


if __name__ == "__main__":
    parameters_file = "/media/dl-rookie/Data/Final_thesis_results/Data/matern32_Mu10_Var1p96_CorH30_CorV15_linear_81/parameters.txt"
    analytical_inversion = AnalyticalInversion(parameters_file)

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
    ]
    analytical_inversion.load_data()
    # inversion_results = analytical_inversion.run_inversion(observation_idx_vec, resimulate=True)
    # metrics = {'rmse': None, 'es': [1, 2], 'vs': [0.5]}
    # metrics_results = analytical_inversion.compute_metrics(metrics)


"""
if __name__ == "__main__":
    # what to plot? - set to True
    plot_example = False  # plot the selected test example (plots the subsurface model along the noisy and noiseless measurement)
    plot_post_MeanStd = False  # plot posterior distribution pixel wise mean and std
    plot_samples_Diffrmse_x = False  # plot samples from posterior distribution at min, max and median rmse values between posterior samples and ground truth
    plot_samples_Diffrmse_y = False  # plot samples from posterior distribution at min, max and median rmse values between posterior resimulations and ground truth
    plot_randpost_samples = False  # plot random samples from posterior distribution
    plot_prior_vs_post_summary = (
        False  # plot prior distribution mean and std along side posterior mean and std
    )
    plot_PostSamples_varios = False  # plot posterior samples variograms (horiz., vert.) along side ground truth vario.
    plot_rmseSamplevsGrdTruth_boxplots = (
        False  # plot RMSE boxplots: posterior samples vs ground truth
    )
    plot_rmse_resimulatioPostSample_boxplots = False  # plot RMSE boxplots: posterior samples resimulation vs noiseless measurement
    plot_priorCovvspostCov = False
    plot_resimulations = False
    plot_all_samples = (
        True  # to plot images of the generated posterior samples. Useful for movie.
    )

    number_inversions = 1  # number of inversions to perform
    samples = 1000  # number of samples to generate

    agg_stats = {}
    # read configuration
    parameters_file = "/media/dl-rookie/Data/Final_thesis_results/Data/exponential_Mu14_Var0p16_CorH25_CorV25_linear_81/parameters.txt"

    config = Config(parameters_file)

    ## setup directories and import necessary files
    datadir = config.datadir
    data_folder_location = config.data_folder_location

    agg_plots_folder = (
        data_folder_location + "/AnalyticalPost_agg_plots"
    )  # create folder where aggregate plots can be stored
    try:  # create data folder
        os.mkdir(agg_plots_folder)
        print("Aggregated plots folder created...")
    except OSError:
        print("Unable to create analytical aggregated plots folder...")

    agg_stats_backup_file = agg_plots_folder + "/agg_stats.txt"
    # read test data from files
    test_models_file = h5py.File(config.data_folder_location + "/test_models.h5")
    test_models = torch.FloatTensor(test_models_file.get("test_models"))
    test_models_file.close()
    test_truett_file = h5py.File(
        config.data_folder_location + "/test_truett_noNoise.h5"
    )
    test_truett = torch.FloatTensor(test_truett_file.get("test_truett_noNoise"))
    test_truett_file.close()

    test_set_size = int(config.set_size - config.set_size * config.val_split)

    # randomly select y_obs vectors to invert
    idx_vec = np.random.randint(0, test_set_size, number_inversions)

    if config.solver_type == "eikonal-nl":
        print("solver type = {}... No analytical solution".format(config.solver_type))

    if config.solver_type == "linear":
        for idx in idx_vec:
            rmse_vec_x = []
            rmse_vec_y = []
            rmse_vec_y_obs = []
            for noise_i in config.noises_list:
                ## Read noise configuration
                noise_distribution = noise_i["distribution"]
                noise_loc = noise_i["location"]
                noise_scale = noise_i["scale"]

                noisy_tt_folder = (
                    data_folder_location
                    + "/noisy_ttvec_{}_loc{}_scale{}".format(
                        noise_distribution,
                        noise_loc,
                        str(noise_scale).replace(".", "p"),
                    )
                )

                if noise_distribution.lower() == "Gaussian".lower():
                    # create folder to store posterior samples/results
                    idx_analytical_inversion_folder = (
                        noisy_tt_folder + "/analytical_noisyttvec_{}".format(idx)
                    )

                    try:  # create data folder
                        os.mkdir(idx_analytical_inversion_folder)
                        print(
                            "Analytical solutions folder created for noisy tt vector {}...".format(
                                idx
                            )
                        )
                    except OSError:
                        print(
                            "Unable to create analytical solutions folder for noisy tt vector {}...".format(
                                idx
                            )
                        )

                    # read y_obs
                    with open(
                        noisy_tt_folder + "/noisy_tt_vec{}".format(idx), "rb"
                    ) as f:
                        y_obs = pickle.load(f)

                    # generate posterior mean and covariance
                    post_dict = setup_posterior(
                        noise_scale,
                        y_obs,
                        config.solver_matrix,
                        config.ndata,
                        config.CM,
                        config.m_prior,
                        config.ntot,
                        jitter=0,
                    )

                    # generate samples from posterior distribution
                    post_samples = generate_Gauss_samples(
                        samples,
                        mean=post_dict["posterior_mean"],
                        cov=post_dict["posterior_covariance"],
                        dims=[config.nc, config.nx, config.ny, config.ntot],
                    )

                    # calculate RMSE values between posterior samples and ground truth
                    ref_x = test_models[idx, :, :, :].view(1, config.nx * config.ny)
                    ref_x = ref_x.numpy()

                    rmse_vec_x_i = compute_rmse_grdTruth(
                        np.reshape(post_samples, (samples, config.nx * config.ny)),
                        ref_x,
                    )
                    rmse_vec_x.append(rmse_vec_x_i)

                    agg_stats[
                        "ex{}_{}_{}_RMSE_posteriorMeanVsGrdTruth".format(
                            idx, noise_distribution, str(noise_scale).replace(".", "p")
                        )
                    ] = compute_rmse_grdTruth(
                        ref_x, post_dict["posterior_mean"].reshape(1, -1)
                    )[
                        0
                    ]

                    # resimulate posterior samples through forward solver
                    resimulations = resimulate(
                        post_samples,
                        config.solver_type,
                        [
                            config.rays,
                            config.nx,
                            config.ny,
                            config.spacing,
                            config.sources_x,
                        ],
                    )

                    # calculate RMSE values between posterior resimulations and ground truth (i.e. noiseless travel time)
                    ref_y = test_truett[idx, :].view(1, -1)  # noisless
                    ref_y = ref_y.numpy()

                    rmse_vec_y_i = compute_rmse_grdTruth(resimulations, ref_y)
                    rmse_vec_y.append(rmse_vec_y_i)

                    rmse_vec_y_obs_i = compute_rmse_grdTruth(
                        resimulations, y_obs.view(1, -1).numpy()
                    )
                    rmse_vec_y_obs.append(rmse_vec_y_obs_i)

                    # plots
                    ## read noise free measurement
                    # example_x = test_models[idx, :, :, :].numpy().reshape(config.ny, config.nx)
                    # example_y = test_truett[idx, :].squeeze().numpy()

                    ## plot the example
                    if plot_example:
                        example_x = ref_x.reshape(config.ny, config.nx)
                        example_y = ref_y.reshape(-1)
                        save_location_file = (
                            idx_analytical_inversion_folder
                            + "/plot_ex_{}.pdf".format(idx)
                        )
                        plot.plot_example(
                            example_x, example_y, y_obs, save_location_file, dpi=600
                        )

                    ## calculate posterior means and std and plot them
                    if plot_post_MeanStd:
                        save_location_file = (
                            idx_analytical_inversion_folder
                            + "/posteriorSummaryStats_ex_{}.pdf".format(idx)
                        )
                        plot.plot_summary_models(
                            post_samples.reshape(samples, config.nx * config.ny),
                            width=config.nx,
                            height=config.ny,
                            save_location=save_location_file,
                        )

                    ## plot samples against ground truth - at different RMSE reference values with x grd truth (min , median, max)
                    if plot_samples_Diffrmse_x:
                        example_x = ref_x.reshape(config.ny, config.nx)
                        example_y = ref_y.reshape(-1)

                        median_rmse_loc = np.where(
                            rmse_vec_x_i.reshape(-1)
                            == np.quantile(
                                rmse_vec_x_i.reshape(-1), 0.5, interpolation="nearest"
                            )
                        )[0][0]

                        sort_order = np.argsort(rmse_vec_x_i)
                        k = 3

                        k_min_idx = sort_order[0:k]
                        k_min_RMSE = rmse_vec_x_i[sort_order[k_min_idx]]

                        k_max_idx = sort_order[-1 - k : -1]
                        k_max_RMSE = rmse_vec_x_i[sort_order[k_max_idx]]

                        median_rmse_loc_arg = np.where(sort_order == median_rmse_loc)[
                            0
                        ][0]
                        k_median_idx = sort_order[
                            median_rmse_loc_arg - 1 : median_rmse_loc_arg + 2
                        ]
                        k_median_RMSE = rmse_vec_x_i[sort_order[k_median_idx]]

                        examples_min_rmse = np.concatenate(
                            (
                                example_x.reshape(1, config.ny * config.nx),
                                post_samples[k_min_idx, :, :, :].reshape(
                                    -1, config.ny * config.nx
                                ),
                            ),
                            axis=0,
                        ).reshape(1, k + 1, -1)
                        examples_median_rmse = np.concatenate(
                            (
                                example_x.reshape(1, config.ny * config.nx),
                                post_samples[k_median_idx, :, :, :].reshape(
                                    -1, config.ny * config.nx
                                ),
                            ),
                            axis=0,
                        ).reshape(1, k + 1, -1)
                        examples_max_rmse = np.concatenate(
                            (
                                example_x.reshape(1, config.ny * config.nx),
                                post_samples[k_max_idx, :, :, :].reshape(
                                    -1, config.ny * config.nx
                                ),
                            ),
                            axis=0,
                        ).reshape(1, k + 1, -1)

                        examples = np.concatenate(
                            (
                                examples_min_rmse,
                                examples_median_rmse,
                                examples_max_rmse,
                            ),
                            axis=0,
                        )

                        examples = examples.reshape(-1, k + 1, config.ny * config.nx)
                        save_location_file = (
                            idx_analytical_inversion_folder
                            + "/posteriorExamples_MinMeanMaxRMSE_x_ex_{}.pdf".format(
                                idx
                            )
                        )
                        plot.plot_samples(
                            examples,
                            width=config.nx,
                            height=config.ny,
                            grd_truth=True,
                            save_location=save_location_file,
                        )

                        ## plot samples against ground truth - at different RMSE reference values with x grd truth (min , median, max)
                        if plot_samples_Diffrmse_y:
                            example_x = ref_x.reshape(config.ny, config.nx)
                            example_y = ref_y.reshape(-1)

                            median_rmse_loc_y = np.where(
                                rmse_vec_y_i.reshape(-1)
                                == np.quantile(
                                    rmse_vec_y_i.reshape(-1),
                                    0.5,
                                    interpolation="nearest",
                                )
                            )[0][0]
                            median_rmse_loc_y_obs = np.where(
                                rmse_vec_y_obs_i.reshape(-1)
                                == np.quantile(
                                    rmse_vec_y_obs_i.reshape(-1),
                                    0.5,
                                    interpolation="nearest",
                                )
                            )[0][0]

                            sort_order_y = np.argsort(rmse_vec_y_i)
                            sort_order_y_obs = np.argsort(rmse_vec_y_obs_i)
                            k = 3

                            k_min_idx_y = sort_order_y[0:k]
                            k_min_RMSE_y = rmse_vec_y_i[sort_order_y[k_min_idx_y]]

                            k_min_idx_y_obs = sort_order_y_obs[0:k]
                            k_min_RMSE_y_obs = rmse_vec_y_obs_i[
                                sort_order_y_obs[k_min_idx_y_obs]
                            ]

                            k_max_idx_y = sort_order_y[-1 - k : -1]
                            k_max_RMSE_y = rmse_vec_y_i[sort_order_y[k_max_idx_y]]

                            k_max_idx_y_obs = sort_order_y_obs[-1 - k : -1]
                            k_max_RMSE_y_obs = rmse_vec_y_obs_i[
                                sort_order_y_obs[k_max_idx_y_obs]
                            ]

                            median_rmse_loc_arg_y = np.where(
                                sort_order_y == median_rmse_loc_y
                            )[0][0]
                            k_median_idx_y = sort_order_y[
                                median_rmse_loc_arg_y - 1 : median_rmse_loc_arg_y + 2
                            ]
                            k_median_RMSE_y = rmse_vec_y_i[sort_order_y[k_median_idx_y]]

                            median_rmse_loc_arg_y_obs = np.where(
                                sort_order_y_obs == median_rmse_loc_y_obs
                            )[0][0]
                            k_median_idx_y_obs = sort_order_y_obs[
                                median_rmse_loc_arg_y_obs
                                - 1 : median_rmse_loc_arg_y_obs
                                + 2
                            ]
                            k_median_RMSE_y_obs = rmse_vec_y_obs_i[
                                sort_order_y_obs[k_median_idx_y_obs]
                            ]

                            examples_min_rmse_y = np.concatenate(
                                (
                                    example_x.reshape(1, config.ny * config.nx),
                                    post_samples[k_min_idx_y, :, :, :].reshape(
                                        -1, config.ny * config.nx
                                    ),
                                ),
                                axis=0,
                            ).reshape(1, k + 1, -1)
                            examples_median_rmse_y = np.concatenate(
                                (
                                    example_x.reshape(1, config.ny * config.nx),
                                    post_samples[k_median_idx_y, :, :, :].reshape(
                                        -1, config.ny * config.nx
                                    ),
                                ),
                                axis=0,
                            ).reshape(1, k + 1, -1)
                            examples_max_rmse_y = np.concatenate(
                                (
                                    example_x.reshape(1, config.ny * config.nx),
                                    post_samples[k_max_idx_y, :, :, :].reshape(
                                        -1, config.ny * config.nx
                                    ),
                                ),
                                axis=0,
                            ).reshape(1, k + 1, -1)

                            examples_y = np.concatenate(
                                (
                                    examples_min_rmse_y,
                                    examples_median_rmse_y,
                                    examples_max_rmse_y,
                                ),
                                axis=0,
                            )

                            examples_y = examples_y.reshape(
                                -1, k + 1, config.ny * config.nx
                            )
                            save_location_file_y = (
                                idx_analytical_inversion_folder
                                + "/posteriorExamples_MinMeanMaxRMSE_y_ex_{}.pdf".format(
                                    idx
                                )
                            )
                            plot.plot_samples(
                                examples_y,
                                width=config.nx,
                                height=config.ny,
                                grd_truth=True,
                                save_location=save_location_file_y,
                            )

                            examples_min_rmse_y_obs = np.concatenate(
                                (
                                    example_x.reshape(1, config.ny * config.nx),
                                    post_samples[k_min_idx_y_obs, :, :, :].reshape(
                                        -1, config.ny * config.nx
                                    ),
                                ),
                                axis=0,
                            ).reshape(1, k + 1, -1)
                            examples_median_rmse_y_obs = np.concatenate(
                                (
                                    example_x.reshape(1, config.ny * config.nx),
                                    post_samples[k_median_idx_y_obs, :, :, :].reshape(
                                        -1, config.ny * config.nx
                                    ),
                                ),
                                axis=0,
                            ).reshape(1, k + 1, -1)
                            examples_max_rmse_y_obs = np.concatenate(
                                (
                                    example_x.reshape(1, config.ny * config.nx),
                                    post_samples[k_max_idx_y_obs, :, :, :].reshape(
                                        -1, config.ny * config.nx
                                    ),
                                ),
                                axis=0,
                            ).reshape(1, k + 1, -1)

                            examples_y_obs = np.concatenate(
                                (
                                    examples_min_rmse_y_obs,
                                    examples_median_rmse_y_obs,
                                    examples_max_rmse_y_obs,
                                ),
                                axis=0,
                            )

                            examples_y_obs = examples_y_obs.reshape(
                                -1, k + 1, config.ny * config.nx
                            )
                            save_location_file_y_obs = (
                                idx_analytical_inversion_folder
                                + "/posteriorExamples_MinMeanMaxRMSE_y_obs_ex_{}.pdf".format(
                                    idx
                                )
                            )
                            plot.plot_samples(
                                examples_y_obs,
                                width=config.nx,
                                height=config.ny,
                                grd_truth=True,
                                save_location=save_location_file_y_obs,
                            )

                            if plot_resimulations:
                                references = np.concatenate(
                                    (ref_y, y_obs.view(1, -1).numpy()), axis=0
                                )
                                save_location_file = (
                                    idx_analytical_inversion_folder
                                    + "/postResim_MaxRMSEy_obs_ex_{}.pdf".format(idx)
                                )
                                plot.plot_resimulations(
                                    resimulations[k_max_idx_y_obs, :],
                                    references,
                                    save_location=save_location_file,
                                )

                    # plot random posterior samples against ground truth
                    if plot_randpost_samples:
                        example_x = ref_x.reshape(config.ny, config.nx)
                        example_y = ref_y.reshape(-1)

                        k = 3
                        examples_rand = np.concatenate(
                            (
                                example_x.reshape(1, config.ny * config.nx),
                                post_samples[0:k, :, :, :].reshape(
                                    -1, config.ny * config.nx
                                ),
                            ),
                            axis=0,
                        ).reshape(1, k + 1, -1)
                        examples_rand = examples_rand.reshape(
                            -1, k + 1, config.ny * config.nx
                        )
                        save_location_file = (
                            idx_analytical_inversion_folder
                            + "/posteriorRandSamplesVsGrdTruth_ex_{}.pdf".format(idx)
                        )
                        plot.plot_samples(
                            examples_rand,
                            width=config.nx,
                            height=config.ny,
                            grd_truth=True,
                            save_location=save_location_file,
                        )

                    # plot prior vs posterior summary stats
                    if plot_prior_vs_post_summary:
                        test_models_vecs = (
                            test_models[:, :, :, :]
                            .numpy()
                            .reshape(test_set_size, config.ny * config.nx)
                        )
                        prior_stats = plot.plot_summary_models(
                            test_models_vecs,
                            width=config.nx,
                            height=config.ny,
                            save_location=save_location_file,
                            gen_plots=False,
                        )
                        posterior_stats = plot.plot_summary_models(
                            post_samples.reshape(-1, config.ny * config.nx),
                            width=config.nx,
                            height=config.ny,
                            save_location=save_location_file,
                            gen_plots=False,
                        )
                        save_location_file = (
                            idx_analytical_inversion_folder
                            + "/priorVsposteriorSummaryStats_ex_{}.pdf".format(idx)
                        )
                        plot.plot_prior_vs_posterior_stats(
                            prior_stats,
                            posterior_stats,
                            width=config.nx,
                            height=config.ny,
                            save_location=save_location_file,
                        )

                    # plot variograms - post samples vs grd truth
                    if plot_PostSamples_varios:
                        example_x = ref_x.reshape(config.ny, config.nx)
                        example_y = ref_y.reshape(-1)

                        models = np.concatenate(
                            (
                                post_samples.reshape(
                                    samples, config.nx * config.ny
                                ).reshape(1, samples, -1),
                                np.tile(
                                    example_x.reshape(config.nx * config.ny),
                                    (samples, 1),
                                ).reshape(1, samples, -1),
                            ),
                            axis=0,
                        )
                        save_location_file_h = (
                            idx_analytical_inversion_folder
                            + "/horizVariogramPostSolutions_ex_{}.pdf".format(idx)
                        )
                        save_location_file_v = (
                            idx_analytical_inversion_folder
                            + "/vertVariogramPostSolutions_ex_{}.pdf".format(idx)
                        )
                        plot.plot_hv_variograms(
                            models,
                            [config.nx, config.ny],
                            [1, 0],
                            ["Post-samples", "Ground truth"],
                            save_location=[save_location_file_h, save_location_file_v],
                        )

                    # plot prior Covariance matrix vs Posterior covariance matrix
                    if plot_priorCovvspostCov:
                        prior_cov = (config.CM).reshape(1, config.ntot, config.ntot)
                        post_cov = post_dict["posterior_covariance"].reshape(
                            1, config.ntot, config.ntot
                        )
                        matrices = np.concatenate((prior_cov, post_cov), axis=0)
                        save_location_file = (
                            idx_analytical_inversion_folder
                            + "/priorVsposteriorCovariances_ex_{}.pdf".format(idx)
                        )
                        plot.plot_matrices(
                            matrices,
                            ["Prior covariance", "Posterior covariance"],
                            save_location=save_location_file,
                        )

                        ref_points = [0, 979, 1999]
                        for ref_point in ref_points:
                            save_location_file = (
                                idx_analytical_inversion_folder
                                + "/priorVsposteriorCorrelations_RefPoint{}_ex_{}.pdf".format(
                                    ref_point, idx
                                )
                            )
                            plot.plot_cov_to_corr(
                                matrices,
                                ref_point,
                                [config.ny, config.nx],
                                save_location=save_location_file,
                            )

                    if plot_all_samples:
                        example_x = ref_x.reshape(config.ny, config.nx)
                        post_samples_img_dir = (
                            idx_analytical_inversion_folder + "/post_samples_images"
                        )
                        try:
                            os.mkdir(post_samples_img_dir)
                            print("Created posterior samples images directory...")
                        except OSError:
                            print(
                                "Could not create posterior samples images directory..."
                            )
                        plot.plot_all_samples(
                            example_x,
                            post_samples,
                            [config.ny, config.nx],
                            200,
                            np.min(post_samples),
                            np.max(post_samples),
                            save_location=post_samples_img_dir,
                        )
                else:
                    print(
                        "Noise distribution = {} ... No analytical solution".format(
                            noise_distribution
                        )
                    )

            # plot RMSE boxplots - Slowness posterior samples vs ground truth
            if plot_rmseSamplevsGrdTruth_boxplots:
                save_location_file = (
                    agg_plots_folder
                    + "/boxplot_rmse_postSamplesVsGrdTruth_ex_{}.pdf".format(idx)
                )
                plot.plot_boxplots(
                    np.array(rmse_vec_x).reshape((1, 2, samples)),
                    labels=["Small noise", "Large noise"],
                    axes_plot_titles=[
                        "",
                        "",
                        "Slowness posterior samples RMSE with ground truth (ns/m)",
                    ],
                    save_location=save_location_file,
                )

            if plot_rmse_resimulatioPostSample_boxplots:
                # plot RMSE boxplots - Travel times posterior samples resimulation vs ground truth travel times
                save_location_file = (
                    agg_plots_folder
                    + "/boxplot_rmse_postTTresimVsTTGrdTruth_ex_{}.pdf".format(idx)
                )
                plot.plot_boxplots(
                    np.array(rmse_vec_y).reshape((1, 2, samples)),
                    labels=["Small noise", "Large noise"],
                    axes_plot_titles=[
                        "",
                        "",
                        "Travel times resimulation RMSE with ground truth (ns)",
                    ],
                    save_location=save_location_file,
                )

                # plot RMSE boxplots - Travel times posterior samples resimulation vs y_obs
                save_location_file = (
                    agg_plots_folder
                    + "/boxplot_rmse_postTTresimVsy_obs_ex_{}.pdf".format(idx)
                )
                plot.plot_boxplots(
                    np.array(rmse_vec_y_obs).reshape((1, 2, samples)),
                    labels=["Small noise", "Large noise"],
                    axes_plot_titles=[
                        "",
                        "",
                        "Travel times resimulation RMSE with noisy measurements (ns)",
                    ],
                    save_location=save_location_file,
                )

            with open(agg_stats_backup_file, "w") as data:
                data.write(str(agg_stats))
"""
