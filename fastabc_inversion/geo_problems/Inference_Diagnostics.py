"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Base class for inference diagnostics after running SuS experiments.
"""
import os
import pickle
import time

import fastabc_inversion.geo_problems.utils.evaluation.curve_approximation as ca
import fastabc_inversion.geo_problems.utils.visualization.plotting_tools as plot
import numpy as np
import torch
from fastabc_inversion.geo_problems.linear.analytical_inversion import \
    resimulate
from fastabc_inversion.geo_problems.utils import torch_distances as torch_dist
from fastabc_inversion.geo_problems.utils.torch_data_prep import un_normalize


def save_to_disk(data, file_path, _pickle=True, _text=False):
    # pickle the data
    if _pickle:
        # add .pkl extension if not present
        if not file_path.endswith(".pkl"):
            file_path += ".pkl"
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    if _text:
        # add .txt extension if not present
        if not file_path.endswith(".txt"):
            file_path += ".txt"

        with open(file_path, "w") as f:
            if not isinstance(data, str):
                data = str(data)
            f.write(data)


def load_from_disk(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def plot_stats(stats_vec, b_vec, titles, file_name, bootstraps=1, dpi=600, show=False):
    """Helper function to plot statistics vs b values.
    :param stats_vec:
    :param b_vec:
    :param titles:
    :param file_name:
    :param bootstraps:
    :param dpi:
    :param show:
    :return:
    """
    b_vec = np.array(b_vec)
    sort_args = np.argsort(b_vec)
    sorted_b = b_vec[sort_args]

    if bootstraps == 1:
        # line plot
        stats_vec = np.array(stats_vec).flatten()

        # concatenate sorted_b and stats_vec vertically
        sorted_b = sorted_b.reshape(-1, 1)
        stats_vec = stats_vec[sort_args].reshape(-1, 1)
        data = np.concatenate((sorted_b, stats_vec), axis=1).reshape(
            1, len(sorted_b), 2
        )

        labels = ["diag"]
        plot.plot_scatters(
            data,
            labels=labels,
            make_scatter=True,
            axis_labels={"x_label": titles[1], "y_label": titles[2]},
            plot_title=titles[0],
            save_location=file_name,
            show=show,
            dpi=dpi,
        )
    else:
        # boxplot
        num_thresholds = len(sorted_b)

        labels = [f"{i}" for i in sort_args]

        stats_vec = np.array(stats_vec)[sort_args]
        stats_vec = stats_vec.reshape(1, num_thresholds, bootstraps)

        plot.plot_boxplots(
            stats_vec,
            labels,
            axes_plot_titles=titles,
            lower_lim=None,
            save_location=file_name,
            dpi=dpi,
            show=show,
        )


def find_percentile_of_value_in_array(array, value):
    """
    Find the percentile of a value in an array.
    :param value: value to find the percentile of
    :param array: 1D numpy array
    :return: percentile of the value in the array
    """
    from scipy import stats

    array = np.array(array)
    if len(array) == 0:
        return None
    else:
        percentile = stats.percentileofscore(array, value, kind="weak")
        return percentile


def call_sinkhorn(data_1, data_2, sinkhorn_params, on_gpu=True):
    """
    Helper function to prepare the data for sinkhorn distance computation.
    :param data_1: first set of samples of shape (n_samples, n_features)
    :param data_2: second set of samples of shape (n_samples, n_features)
    :param sinkhorn_params: dictionary with sinkhorn parameters containing one or more of the following keys: 'epsilon',
    'niter', 'p'
    :return:
    """
    import fastabc_inversion.geo_problems.utils.sinkhorn.sinkhorn_pointcloud as spc

    # get sinkhorn parameters from the input dict
    if not isinstance(sinkhorn_params, dict):
        raise ValueError("sinkhorn_params should be a dictionary.")

    if on_gpu:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # read and fill missing parameters with default values
    spc_params = {
        "epsilon": sinkhorn_params.get("epsilon", 100),
        "n": sinkhorn_params.get("n", 500),
        "niter": sinkhorn_params.get("niter", 100),
        "p": sinkhorn_params.get("p", 2),
        "device": device,
    }

    # prepare the data
    ref_1_n = data_1.shape[0]
    ref_2_n = data_2.shape[0]

    # verify that only one of the two sizes is larger than one if they are not equal
    if ref_1_n != ref_2_n and ref_1_n > 1 and ref_2_n > 1:
        raise ValueError(
            "Only one of the two input arrays can have more than one sample if their sizes are not equal."
        )

    n = max(ref_1_n, ref_2_n)

    if n < spc_params["n"]:
        spc_params["n"] = n

    data_1_torch = torch.tensor(data_1, dtype=torch.float32, device=device)
    data_2_torch = torch.tensor(data_2, dtype=torch.float32, device=device)

    # align the shapes of the two tensors
    if ref_1_n == 1:
        data_1_torch = data_1_torch.repeat(n, 1)
    if ref_2_n == 1:
        data_2_torch = data_2_torch.repeat(n, 1)

    sinkhorn = spc.sinkhorn_normalized(data_1_torch, data_2_torch, **spc_params)

    return sinkhorn


def compute_array_stats(np_array):
    """
    Compute statistics for an array.
    Statistics computed are mean, median, 25th and 75th percentiles, 2.5th and 97.5th percentiles.
    :param np_array: 1D numpy array
    """
    if np_array is not None:
        mean = np.mean(np_array)
        std = np.std(np_array)
        median = np.median(np_array)
        q25 = np.quantile(np_array, q=0.25)
        q75 = np.quantile(np_array, q=0.75)
        q025 = np.quantile(np_array, q=0.025)
        q975 = np.quantile(np_array, q=0.975)

        return {
            "mean": mean,
            "median": median,
            "q25": q25,
            "q75": q75,
            "q025": q025,
            "q975": q975,
            "std": std,
        }
    else:
        return {
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "q025": None,
            "q975": None,
            "std": None,
        }


def format_cca_metrics(cca_metrics, n_obs):
    """
    Format the CCA metrics to be used in the inference benchmark uses ES (1) and VS(0.5), 'small_gauss' and 'large_gauss'.
    Only has values for metrics on 'x'.
    The output format is :
    {'CCA':
        {
        'rmse':{
                'small_gauss':[],
                'large_gauss':[]
                },
        'es':{
            1:
                {
                'small_gauss':[],
                'large_gauss':[]
                },
            }
        'vs':{
            0.5:
                {
                'small_gauss':[],
                'large_gauss':[]
                },
            },
        }
    }
    :param cca_metrics: dictionary with metrics computed with probabilistic cca
    :return: formatted dictionary with metrics
    """
    _map_noise = {"small_noise": "small_gauss", "large_noise": "large_gauss"}

    noise_list = list(cca_metrics.keys())

    metrics = list(cca_metrics[noise_list[0]].keys())

    m = 500

    if "es" in metrics:
        es_powers = [1]
    if "vs" in metrics:
        vs_powers = [0.5]

    formatted_metrics = {"CCA": {}}
    for metric in metrics:
        if metric == "es":
            formatted_metrics["CCA"]["es"] = {}
            for p in es_powers:
                formatted_metrics["CCA"]["es"][p] = {}
                for noise in noise_list:
                    if len(cca_metrics[noise][metric]) == n_obs:
                        formatted_metrics["CCA"]["es"][p][
                            _map_noise[noise]
                        ] = cca_metrics[noise][metric]
                    else:
                        raise ValueError(
                            f"Expected {n_obs} observations for metric {metric} but got {len(cca_metrics[noise][metric])} for noise {noise}."
                        )
        elif metric == "vs":
            formatted_metrics["CCA"]["vs"] = {}
            for p in vs_powers:
                formatted_metrics["CCA"]["vs"][p] = {}
                for noise in noise_list:
                    if len(cca_metrics[noise][metric]) == n_obs:
                        formatted_metrics["CCA"]["vs"][p][
                            _map_noise[noise]
                        ] = cca_metrics[noise][metric]
                    else:
                        raise ValueError(
                            f"Expected {n_obs} observations for metric {metric} but got {len(cca_metrics[noise][metric])} for noise {noise}."
                        )
        elif metric == "rmse":
            formatted_metrics["CCA"]["rmse"] = {}
            for noise in noise_list:
                if len(cca_metrics[noise][metric]) == int(n_obs * m):
                    formatted_metrics["CCA"]["rmse"][_map_noise[noise]] = cca_metrics[
                        noise
                    ][metric]
                else:
                    raise ValueError(
                        f"Expected {int(n_obs * m)} observations for metric {metric} but got {len(cca_metrics[noise][metric])} for noise {noise}."
                    )
        else:
            raise ValueError(
                f"Unknown metric {metric} in CCA metrics. Expected 'es', 'vs' or 'rmse'."
            )
    return formatted_metrics


def format_cvae_metrics(cvae_metrics, n_obs, noise_list):
    """

    :param cvae_metrics:
    :param n_obs:
    :param noise_list:
    :return:
    """
    metrics = ["rmse", "es", "vs"]
    es_p = [1, 2]
    vs_p = [0.5]
    m = 500

    formatted_metrics = {"cVAE": {}}
    for metric in metrics:
        if metric == "es":
            formatted_metrics["cVAE"]["es"] = {}
            for p in es_p:
                formatted_metrics["cVAE"]["es"][p] = {}
                for noise in noise_list:
                    if len(cvae_metrics[metric][p][noise]) == n_obs:
                        formatted_metrics["cVAE"]["es"][p][noise] = cvae_metrics[
                            metric
                        ][p][noise]
                    else:
                        raise ValueError(
                            f"Expected {n_obs} observations for metric {metric} but got {len(cvae_metrics[metric][p][noise])} for noise {noise}."
                        )
        elif metric == "vs":
            formatted_metrics["cVAE"]["vs"] = {}
            for p in vs_p:
                formatted_metrics["cVAE"]["vs"][p] = {}
                for noise in noise_list:
                    if len(cvae_metrics[metric][p][noise]) == n_obs:
                        formatted_metrics["cVAE"]["vs"][p][noise] = cvae_metrics[
                            metric
                        ][p][noise]
                    else:
                        raise ValueError(
                            f"Expected {n_obs} observations for metric {metric} but got {len(cvae_metrics[metric][p][noise])} for noise {noise}."
                        )
        elif metric == "rmse":
            formatted_metrics["cVAE"]["rmse"] = {}
            for noise in noise_list:
                if len(cvae_metrics[metric][noise]) == int(n_obs * m):
                    formatted_metrics["cVAE"]["rmse"][noise] = cvae_metrics[metric][
                        noise
                    ]
                else:
                    raise ValueError(
                        f"Expected {int(n_obs * m)} observations for metric {metric} but got {len(cvae_metrics[metric][noise])} for noise {noise}."
                    )
        else:
            raise ValueError(
                f"Unknown metric {metric} in cVAE metrics. Expected 'es', 'vs' or 'rmse'."
            )
    return formatted_metrics


class InferenceDiagnostics:
    def __init__(self, experiment_obj, round_digits=3):
        self.experiment_obj = experiment_obj
        self.round_digits = round_digits
        self.inference_root_dir = self.experiment_obj.inference_dir

        self.inference_params = (
            self.experiment_obj.inference_params
            if hasattr(self.experiment_obj, "inference_params")
            else None
        )
        self.all_obs_results = None

        self.train_x = None
        self.train_y = None
        self.test_x = None  # Ground truth test data
        self.test_y = None  # Ground truth test data

        self.inverted_obs = None
        self.inverted_obs_vectors = {}  # list of inverted observations
        self.noise_list = None  # list of noise types that contaminated the observations used during SuS inference
        self.epsilon_values = None  # explicit epsilon values used during SuS inference
        self.all_obs_max_curvature = None

        self.all_thresholds = None  # all thresholds reached during SuS inference
        self.all_thresholds_inverted_z = (
            None  # z values retrieved by the SuS experiment
        )
        self.all_thresholds_inverted_x = (
            None  # inverted_z transformed back to x space via the decoder
        )
        self.all_thresholds_inverted_y = (
            None  # inverted_z transformed back to y space via the decoder
        )
        self.all_thresholds_resimulated_inverted_x = None  # resimulated inverted_x

        self.final_threshold = None  # final threshold reached during SuS inference
        self.final_threshold_inverted_z = None  # z values retrieved by the SuS experiment for the final threshold only
        self.final_threshold_inverted_x = None  # inverted_z transformed back to x space via the decoder for the final threshold only
        self.final_threshold_inverted_y = None  # inverted_z transformed back to y space via the decoder for the final threshold only

        self.es_stats = None
        self.vs_stats = None
        self.es_stats_y = None
        self.vs_stats_y = None
        self.es_stats_resims = None
        self.vs_stats_resims = None

        self.all_obs_min_es = None
        self.all_obs_min_vs = None
        self.all_obs_min_es_y = None
        self.all_obs_min_vs_y = None
        self.all_obs_min_es_resims = None
        self.all_obs_min_vs_resims = None

    def get_true_thresholds(
        self,
        all_obs_results,
        obs_idx,
        noise_label,
        epsilon,
        use_true_thresh=False,
        SuS_run_id=0,
    ):
        """
        Get the true thresholds for a given observation and noise type from the b_line data (useful when the b vector
        was not returned during SuS inference).
        :param all_obs_results: dictionary, results of the SuS inversion for all observations
        :param obs_idx: int, observation id
        :param noise_label: str, noise type
        :param epsilon: float, epsilon value
        :param use_true_thresh: bool, whether to use actual vector of thresholds that were reached (True)
            or the b_line computed data (False)
        :param SuS_run_id: int, SuS run id (default is 0)
        :return:
        """
        true_thresholds = []
        p0 = self.inference_params[noise_label]["p0"]
        N = self.inference_params[noise_label]["N"]

        if not use_true_thresh:
            width = int(N * p0)  # Nc (number of chains of the SuS alg.)
            b_line = np.array(
                all_obs_results[obs_idx][noise_label][epsilon]["b_line"][SuS_run_id]
            )
            height = int(
                b_line.shape[1] / width
            )  # number of thresholds reached during SuS inference (number of iterations)
            b_line = b_line.reshape(height, width)

            true_thresholds.append(
                b_line[0, -1] + 1000
            )  # initial threshold value ~ infinity
            # we add this infinite threshold in order to align the values of the thresholds with the samples,
            # the first sample being from the prior distribution of the latent space (hence, prob. of failure is 1)
            # and the last one being from the subspace where the final threshold was reached (hence, prob. of failure is the
            # final estimate of the prob. of failure P_f)

            # append to true_thresholds the values in b_line[:, 0] (first column of b_line)
            true_thresholds.extend(b_line[:, 0])
        else:
            b_vector = all_obs_results[obs_idx][noise_label][epsilon]["all_thresholds"][
                SuS_run_id
            ]
            true_thresholds.append(
                b_vector[0] + 1000
            )  # initial threshold value ~ infinity
            true_thresholds.extend(b_vector)

        return true_thresholds

    def get_noise_lp_norm(self, obs_idx, noise_label, p=2, use_gpu=True):
        """
        Get the Lp norm of the noise that contaminated the observation.
        :param obs_idx: int, observation index
        :param noise_label: str, noise type
        :param p: int, p-norm
        :return:
        """
        y_obs = self.inverted_obs_vectors[obs_idx][noise_label]["obs"].reshape(1, -1)
        y_ref = self.inverted_obs_vectors[obs_idx]["ground_truth"].reshape(1, -1)

        if use_gpu:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

            if not isinstance(y_obs, torch.Tensor):
                y_obs = torch.tensor(y_obs, dtype=torch.float32, device=device)
            if not isinstance(y_ref, torch.Tensor):
                y_ref = torch.tensor(y_ref, dtype=torch.float32, device=device)

            noise_lp_norm = torch_dist.lpp_torch(y_obs, y_ref, p=p).cpu().numpy()
        else:
            noise_lp_norm = torch_dist.lpp_torch(y_obs, y_ref, p=p).numpy()

        return noise_lp_norm

    def make_vs_b_data(
        self, obs_idx, noise_label, epsilon, align_with_samples=True, SuS_run_id=0
    ):
        """
        Make intermediate values of prob of failure and Coefficient of variations of the prob of failure vs b values.
        :param obs_idx: observation index
        :param noise_label: noise type
        :param epsilon: epsilon value
        :param align_with_samples: bool, whether to add an infinite threshold value, a prob. of failure of 1 and a CoV of 0
        :param SuS_run_id: int, SuS run id
        :return:
        """

        b = np.array(
            self.all_obs_results[obs_idx][noise_label][epsilon]["all_thresholds"][
                SuS_run_id
            ]
        )
        all_prob = np.array(
            self.all_obs_results[obs_idx][noise_label][epsilon]["all_prob"][SuS_run_id]
        )
        all_delta = (
            np.array(
                self.all_obs_results[obs_idx][noise_label][epsilon]["all_delta"][
                    SuS_run_id
                ]
            )
            ** 2
        )

        # for each b value, compute prob. of failure as the product of all probs. of failure up to that b value index
        # and the CoV of the prob. of failure as the sum of all deltas up to that b value index
        all_p_f = np.cumprod(all_prob)
        all_cov_2 = np.cumsum(
            all_delta
        )  # keep the square of the CoV (= Var(ln(Pf_hat)))

        if align_with_samples:
            order = int(np.log10(b[0]))  # assess the order of b[0]
            b = np.insert(
                b, 0, b[0] + 10**order
            )  # add an initial value to be that is of the same order of the b[0]
            all_p_f = np.insert(all_p_f, 0, 1)  # initial prob. of failure ~ 1
            all_cov_2 = np.insert(all_cov_2, 0, 0)  # initial CoV ~ 0

        return b, all_p_f, all_cov_2

    def get_inverted_xy(self, z_values, **kwargs):
        """
        Get the inverted x and y values from the z values as well as resimulations of inverted x.
        :param z_values: a list of numpy arrays where each element is of shape (N, dim_z) to decode to x and y.
            N is the number of samples from inference_params[noise_label]['N'].
        :return:
        """
        self.experiment_obj.model.eval()

        ny = self.experiment_obj.ny
        nx = self.experiment_obj.nx

        inverted_x_all = []
        inverted_y_all = []
        resimulate_inverted_x_all = []

        if not isinstance(z_values, list):
            z_values = [z_values]

        for z_value_i in z_values:
            # make tensor from z_value_i
            z_value_i = torch.FloatTensor(
                z_value_i
            )  # one element of the list of shape (N, dim_z)
            N = z_value_i.shape[0]

            inverted_x, inverted_y = self.experiment_obj.get_inverted_x_y(
                z_value_i
            )  # tensors of shape (N, dim_x) and (N, dim_y)

            if self.experiment_obj.normalize:
                inverted_x = un_normalize(
                    inverted_x, self.experiment_obj.normalization_dict_x
                )
                inverted_y = un_normalize(
                    inverted_y, self.experiment_obj.normalization_dict_y
                )

            # TODO : check for kwargs existence before calling resimulate
            resimulate_inverted_x = resimulate(
                inverted_x.numpy().reshape(N, ny, nx), **kwargs
            )

            inverted_x_all.append(
                inverted_x
            )  # list of tensors of size similar to z_values
            inverted_y_all.append(
                inverted_y
            )  # list of tensors of size similar to z_values
            resimulate_inverted_x_all.append(
                resimulate_inverted_x
            )  # list of tensors of size similar to z_values

        return inverted_x_all, inverted_y_all, resimulate_inverted_x_all

    def build_diagnostics_data(
        self,
        all_obs_inference_results=None,
        all_thresholds_data=True,
        load_inverted_x_y=False,
        p=2,
    ):
        """
        Prepares the data necessary for running the diagnostics.
        :param all_obs_inference_results: list of dictionaries, each containing the results of the SuS inversion for one observation
        :param all_thresholds_data: whether to get data at the final threshold data (False) or for each intermediate threshold reached during SuS inference (True).
        :param load_inverted_x_y: bool, whether to load the inverted x and y from disk (True) or to compute them (False).
               Use True when the inverted x and y were already computed and saved to disk (e.g., when build_diagnostics_data was run before).
        :return:
        """
        print("Building diagnostics data...")

        # TODO : account for multiple SuS runs. Now assuming only one run.
        SuS_run_id = 0

        dim_x = self.experiment_obj.dim_x

        solver_type = self.experiment_obj.config.solver_type
        solver_args = [
            self.experiment_obj.config.rays,
            self.experiment_obj.config.nx,
            self.experiment_obj.config.ny,
            self.experiment_obj.config.spacing,
            self.experiment_obj.config.sources_x,
        ]
        so_file = (
            self.experiment_obj.config.so_file if solver_type == "eikonal-nl" else None
        )

        if all_obs_inference_results is None:
            if self.all_obs_results is None:
                raise ValueError(
                    "Please provide the results of the SuS inversion for all observations."
                )
            else:
                all_obs_inference_results = self.all_obs_results

        train_size = self.experiment_obj.train_size

        if self.experiment_obj.normalize:
            self.train_x = un_normalize(
                self.experiment_obj.training_x.view(train_size, dim_x),
                self.experiment_obj.normalization_dict_x,
            )
            self.train_y = un_normalize(
                self.experiment_obj.training_y, self.experiment_obj.normalization_dict_y
            )

        if self.experiment_obj.test_x is not None:
            self.test_x = self.experiment_obj.test_x
            self.test_y = self.experiment_obj.test_y
        else:
            raise ValueError(
                "Please first load test data into the experiment object by calling the load_data method."
            )

        # get observations and noise types
        self.inverted_obs = list(all_obs_inference_results.keys())
        self.noise_list = list(
            all_obs_inference_results[self.inverted_obs[0]].keys()
        )  # assumes that all observations have the same noise types

        for obs_idx in self.inverted_obs:
            self.inverted_obs_vectors[obs_idx] = {}
            self.inverted_obs_vectors[obs_idx]["ground_truth"] = self.test_y[obs_idx, :]
            for noise_label in self.noise_list:
                self.inverted_obs_vectors[obs_idx][noise_label] = {}
                self.inverted_obs_vectors[obs_idx][noise_label][
                    "obs"
                ] = self.experiment_obj.get_observation(obs_idx, noise_label)
                self.inverted_obs_vectors[obs_idx][noise_label][
                    "noise_norm"
                ] = self.get_noise_lp_norm(obs_idx, noise_label, p=p)

                self.inverted_obs_vectors[obs_idx][noise_label][
                    "obs_inference_dir"
                ] = f"{self.inference_root_dir}/{noise_label}/{self.experiment_obj.obs_inference_dir_prefix}{obs_idx}"

        self.epsilon_values = {
            noise_label: list(
                all_obs_inference_results[self.inverted_obs[0]][noise_label].keys()
            )
            for noise_label in self.noise_list
        }  # explicit (the ones selected by the user) threshold values

        # get inverted z values
        if all_thresholds_data:
            # load data from all thresholds reached during SuS inference (true thresholds not epsilon values)
            self.all_thresholds = {}
            self.all_thresholds_inverted_z = {}
            self.all_thresholds_inverted_x = {}
            self.all_thresholds_inverted_y = {}
            self.all_thresholds_resimulated_inverted_x = {}

            for obs_idx in self.inverted_obs:
                for noise_label in self.noise_list:
                    for epsilon in self.epsilon_values[noise_label]:
                        # N = self.inference_params[noise_label]['N']

                        print(
                            "getting inversion data for obs_idx: ",
                            obs_idx,
                            " noise_label: ",
                            noise_label,
                            " epsilon: ",
                            epsilon,
                        )
                        self.all_thresholds[
                            (obs_idx, noise_label, epsilon)
                        ] = self.get_true_thresholds(
                            all_obs_inference_results,
                            obs_idx,
                            noise_label,
                            epsilon,
                            use_true_thresh=True,
                        )

                        self.all_thresholds_inverted_z[
                            (obs_idx, noise_label, epsilon)
                        ] = all_obs_inference_results[obs_idx][noise_label][epsilon][
                            "samples_per_thresh"
                        ][
                            SuS_run_id
                        ]  # assuming one SuS run

                        start_time = time.time()
                        if not load_inverted_x_y:
                            (
                                self.all_thresholds_inverted_x[
                                    (obs_idx, noise_label, epsilon)
                                ],
                                self.all_thresholds_inverted_y[
                                    (obs_idx, noise_label, epsilon)
                                ],
                                self.all_thresholds_resimulated_inverted_x[
                                    (obs_idx, noise_label, epsilon)
                                ],
                            ) = self.get_inverted_xy(
                                self.all_thresholds_inverted_z[
                                    (obs_idx, noise_label, epsilon)
                                ],
                                solver_type=solver_type,
                                args=solver_args,
                                so_file=so_file,
                            )

                            save_to_disk(
                                self.all_thresholds_inverted_x[
                                    (obs_idx, noise_label, epsilon)
                                ],
                                f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/all_thresholds_inverted_x_{epsilon}_SuSRun_{SuS_run_id}.pkl",
                            )
                            save_to_disk(
                                self.all_thresholds_inverted_y[
                                    (obs_idx, noise_label, epsilon)
                                ],
                                f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/all_thresholds_inverted_y_{epsilon}_SuSRun_{SuS_run_id}.pkl",
                            )
                            save_to_disk(
                                self.all_thresholds_resimulated_inverted_x[
                                    (obs_idx, noise_label, epsilon)
                                ],
                                f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/all_resimulated_inverted_x_{epsilon}_SuSRun_{SuS_run_id}.pkl",
                            )
                        else:
                            (
                                self.all_thresholds_inverted_x[
                                    (obs_idx, noise_label, epsilon)
                                ],
                                self.all_thresholds_inverted_y[
                                    (obs_idx, noise_label, epsilon)
                                ],
                                self.all_thresholds_resimulated_inverted_x[
                                    (obs_idx, noise_label, epsilon)
                                ],
                            ) = self.load_invereted_x_y(
                                obs_idx, noise_label, epsilon, SuS_run_id=SuS_run_id
                            )
                        end_time = time.time()
                        print(
                            f"Time to {'load' if load_inverted_x_y else 'invert'} x and y: {end_time-start_time}"
                        )
        else:
            # load only final threshold data
            self.final_threshold = {}
            self.final_threshold_inverted_z = {}
            self.final_threshold_inverted_x = {}
            self.final_threshold_inverted_y = {}

            for obs_idx in self.inverted_obs:
                for noise_label in self.noise_list:
                    for epsilon in self.epsilon_values[noise_label]:
                        self.final_threshold[
                            (obs_idx, noise_label, epsilon)
                        ] = all_obs_inference_results[obs_idx][noise_label][epsilon][
                            "final_epsilon"
                        ]
                        self.final_threshold_inverted_z[
                            (obs_idx, noise_label, epsilon)
                        ] = all_obs_inference_results[obs_idx][noise_label][epsilon][
                            "final_inverted_latent"
                        ]

                        print(
                            "getting inverted x and y for obs_idx: ",
                            obs_idx,
                            " noise_label: ",
                            noise_label,
                            " epsilon: ",
                            epsilon,
                        )
                        (
                            self.final_threshold_inverted_x[
                                (obs_idx, noise_label, epsilon)
                            ],
                            self.final_threshold_inverted_y[
                                (obs_idx, noise_label, epsilon)
                            ],
                        ) = self.get_inverted_xy(
                            [
                                self.final_threshold_inverted_z[
                                    (obs_idx, noise_label, epsilon)
                                ]
                            ]
                        )

                        # TODO : resimulate inverted x ?
                        save_to_disk(
                            self.final_threshold_inverted_x[
                                (obs_idx, noise_label, epsilon)
                            ],
                            f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/final_threshold_inverted_x_{epsilon}.pkl",
                        )
                        save_to_disk(
                            self.final_threshold_inverted_y[
                                (obs_idx, noise_label, epsilon)
                            ],
                            f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/final_threshold_inverted_y_{epsilon}.pkl",
                        )

    def compose_sample(self, obs_key, threshold_ids, final_sample_size):
        """
        Compose a sample by randomly selecting a subset of samples from the thresholds specified by threshold_ids.
        :param obs_key: tuple (obs_idx, noise_label, epsilon) where obs_idx is the observation index,
                        noise_label is the noise label, epsilon is the epsilon value.
        :param threshold_ids: ids of the thresholds to sample from, e.g. [0, 1, 2] for the first three thresholds.
                            These ids conveniently correspond to the -(log_10(P_f)) values, hence, they could be
                            considered as probability thresholds.
                            e.g. value 3 would correspond to a probability threshold of 10^-3.
        :param final_sample_size: final sample size to be composed from the selected thresholds. Each threshold will contribute
                                equally to the final sample size as much as possible.
        :return: x, y and resimulated_x samples of shape (final_sample_size, dim_x),
                (final_sample_size, dim_y) and (final_sample_size, dim_x) respectively.
        """
        # check that ids are ints
        if not all(isinstance(id_, int) for id_ in threshold_ids):
            raise ValueError("All threshold_ids must be integers.")

        # if threshold_ids are negative integers, convert them to positive integers and sort them
        threshold_ids = [abs(id_) for id_ in threshold_ids]
        threshold_ids = sorted(threshold_ids)

        # check that threshold_ids are in range of the number of thresholds
        if any(
            id_ >= len(self.all_thresholds_inverted_x[obs_key]) for id_ in threshold_ids
        ):
            raise ValueError(
                "Some threshold_ids are out of range of the number of thresholds."
            )

        # compute the number of samples to take from each threshold
        num_thresholds = len(threshold_ids)
        samples_per_threshold = final_sample_size // num_thresholds
        if samples_per_threshold == 0:
            raise ValueError(
                "final_sample_size is too small to sample from the specified thresholds."
            )

        samples_per_threshold_list = [samples_per_threshold] * num_thresholds

        # if final_sample_size is not divisible by num_thresholds, compute the remainder
        remainder = final_sample_size % num_thresholds
        if remainder > 0:
            # pick a threshold at random to take the remaining samples from
            random_threshold_id = np.random.choice(len(threshold_ids))
            samples_per_threshold_list[random_threshold_id] += remainder

        # check that the sum of samples_per_threshold_list is equal to final_sample_size
        assert (
            sum(samples_per_threshold_list) == final_sample_size
        ), f"Sum of samples_per_threshold_list {sum(samples_per_threshold_list)} does not equal final_sample_size {final_sample_size}."

        # compose the sample
        x_samples = []
        y_samples = []
        resimulated_x_samples = []

        x_total_sample_size = self.all_thresholds_inverted_x[obs_key][0].shape[0]

        for i, th_id in enumerate(threshold_ids):
            # make a random selection of ids to read from the inverted x and y vectors
            random_ids = np.random.choice(
                x_total_sample_size, samples_per_threshold_list[i]
            )
            x_samples.append(self.all_thresholds_inverted_x[obs_key][th_id][random_ids])
            y_samples.append(self.all_thresholds_inverted_y[obs_key][th_id][random_ids])
            resimulated_x_samples.append(
                self.all_thresholds_resimulated_inverted_x[obs_key][th_id][random_ids]
            )

        # concatenate the samples in order to return a single sample
        x_samples = np.concatenate(x_samples, axis=0)
        y_samples = np.concatenate(y_samples, axis=0)
        resimulated_x_samples = np.concatenate(resimulated_x_samples, axis=0)

        return x_samples, y_samples, resimulated_x_samples

    def load_invereted_x_y(self, obs_idx, noise_label, epsilon, SuS_run_id=0):
        print(
            "loading inverted x and y for obs_idx: ",
            obs_idx,
            " noise_label: ",
            noise_label,
            " epsilon: ",
            epsilon,
        )
        inverted_x = pickle.load(
            open(
                f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/all_thresholds_inverted_x_{epsilon}_SuSRun_{SuS_run_id}.pkl",
                "rb",
            )
        )
        inverted_y = pickle.load(
            open(
                f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/all_thresholds_inverted_y_{epsilon}_SuSRun_{SuS_run_id}.pkl",
                "rb",
            )
        )

        resimulated_inverted_x = pickle.load(
            open(
                f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/all_resimulated_inverted_x_{epsilon}_SuSRun_{SuS_run_id}.pkl",
                "rb",
            )
        )

        return inverted_x, inverted_y, resimulated_inverted_x

    def load_composite_samples(self):
        """
        Load composite samples from disk.
        :return: composite_samples: dictionary where keys are tuples (obs_idx, noise_label, epsilon) and values are
                                    dictionaries with keys 'x', 'y' and 'resim_x' containing the samples.
        """
        print("Loading composite samples from disk...")

        composite_samples = load_from_disk(
            f"{self.inference_root_dir}/composite_samples.pkl"
        )

        # load threshold ids used to compose the samples
        threshold_ids = load_from_disk(f"{self.inference_root_dir}/threshold_ids.pkl")

        # load composite samples metrics if the file exists
        composite_sample_metrics = None
        metrics_file = f"{self.inference_root_dir}/composite_sample_metrics.pkl"
        if os.path.exists(metrics_file):
            print("Loading composite sample metrics from disk...")
            composite_sample_metrics = load_from_disk(metrics_file)

        self.composite_samples = composite_samples
        self.composite_samples_thresholds = threshold_ids
        self.composite_sample_metrics = composite_sample_metrics

        return composite_samples, threshold_ids, composite_sample_metrics

    def make_composite_samples(self, observation_idx_vec, noise_list, threshold_ids):
        """
        Make composite samples from the observations and noise types specified.
        :param observation_idx_vec: list of observation indices to sample from
        :param noise_list: list of noise types to sample from
        :param threshold_ids: ids of the thresholds to sample from, e.g. [0, 1, 2] for the first three thresholds.
                            These ids conveniently correspond to the -(log_10(P_f)) values, hence, they could be
                            considered as probability thresholds.
                            e.g. value 3 would correspond to a probability threshold of 10^-3.
        :return: composite_samples: dictionary where keys are tuples (obs_idx, noise_label, epsilon) and values are
                                    dictionaries with keys 'x', 'y' and 'resim_x' containing the samples.
        """
        print("Making composite samples with thresholds: ", threshold_ids)

        composite_samples = {}
        for obs_idx in observation_idx_vec:
            for noise_label in noise_list:
                for epsilon in self.inference_params[noise_label]["epsilon_vec"]:
                    final_sample_size = self.inference_params[noise_label]["N"]
                    obs_key = (obs_idx, noise_label, epsilon)
                    composite_samples[obs_key] = {}
                    (
                        composite_samples[obs_key]["x"],
                        composite_samples[obs_key]["y"],
                        composite_samples[obs_key]["resim_x"],
                    ) = self.compose_sample(
                        obs_key, threshold_ids, final_sample_size=final_sample_size
                    )

        # save composite samples to disk
        save_to_disk(
            composite_samples,
            f"{self.inference_root_dir}/composite_samples.pkl",
            _pickle=True,
            _text=False,
        )

        # backup the threholds used to compose the samples
        save_to_disk(
            threshold_ids,
            f"{self.inference_root_dir}/threshold_ids.pkl",
            _pickle=True,
            _text=True,
        )

        self.composite_samples = composite_samples
        self.composite_samples_thresholds = threshold_ids

        return composite_samples

    def get_max_curvature_point(
        self,
        x,
        y,
        obs_idx,
        noise_label,
        epsilon,
        log_y=True,
        smoothness=0.2,
        plot_curvature=True,
        prob_thresh=0.1,
        suffix=None,
        dpi=600,
        show=False,
    ):
        sorted_args = np.argsort(x)
        sorted_x = x[sorted_args]
        sorted_y = y[sorted_args]

        # verify if sorted_y contains values > prob_thresh and then pick the stop_id as the first index where sorted_y > prob_thresh
        stop_id = (
            np.argmax(sorted_y > prob_thresh)
            if np.any(sorted_y > prob_thresh)
            else len(sorted_y)
        )
        sorted_x = sorted_x[:stop_id]
        sorted_y = sorted_y[:stop_id]

        if log_y:
            sorted_y = np.log10(sorted_y)

        approximation = ca.curve_smoother(sorted_x, sorted_y, s=smoothness)
        x_new = approximation["x_new"]
        y_new = approximation["y_new"]
        curvature = approximation["curv_y"]
        id_max = approximation["id_max"]
        # first_derivative = approximation['first_derivative']
        # second_derivative = approximation['second_derivative']
        # id_max_origin = approximation['id_max_origin']
        # approx_fn = approximation['approx_fn']

        if plot_curvature:
            mpl, plt, make_axes_locatable, tick = plot.plots_imports()
            plot.base_config(mpl)

            fig, ax1 = plt.subplots()

            limit = (
                id_max + 100
            )  # limit the plot to the max curvature point + 100 for clarity

            ax1.plot(sorted_x[:limit], sorted_y[:limit], label="Probability of failure")
            ax1.plot(x_new[:limit], y_new[:limit], "--", label="Smoothed curve")

            ax2 = ax1.twinx()
            ax2.plot(x_new[:limit], curvature[:limit], "-.", label="Curvature")

            ax1.set_xlabel("Threshold")
            ax1.set_ylabel("Probability of failure")
            ax2.set_ylabel("Curvature")
            ax1.title.set_text(
                f"Max curvature point: {x_new[id_max]:.2f}, {y_new[id_max]:.2f}"
            )

            ax1.legend()
            ax2.legend()

            # remove spines
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)

            if show:
                plt.show()
            else:
                filename_suffix = f"{'_logY' if log_y else ''}"
                plt.savefig(
                    f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/Curvature_{epsilon}{filename_suffix}_Smoothness_{smoothness}{'_'+suffix if suffix is not None else ''}.pdf",
                    dpi=dpi,
                )
                plt.close()

        return x_new[id_max], y_new[id_max]

    def plot_pf_vs_threshold(
        self,
        obs_idx,
        noise_label,
        epsilon,
        log_y=True,
        log_x=True,
        prob_thresh=0.5,
        ci_level=95,
        dpi=600,
        show=False,
    ):
        """
        Plot the probability of failure vs the threshold reached during SuS inference with CI.
        :return:
        """
        print("Plotting Pf vs threshold...")

        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config(mpl)

        b, all_p_f, all_cov_2 = self.make_vs_b_data(
            obs_idx, noise_label, epsilon, align_with_samples=True
        )

        argsort_id = np.argsort(b)
        b = b[argsort_id]
        all_p_f = all_p_f[argsort_id]
        all_cov_2 = all_cov_2[argsort_id]

        stop_id = np.argmax(
            all_p_f > prob_thresh
        )  # stop at the first threshold where Pf > prob_thresh for clarity

        b = b[:stop_id]
        all_p_f = all_p_f[:stop_id]
        all_cov_2 = all_cov_2[:stop_id]

        # compute 95%-CI for \hat{Pf_b} = Prod(\hat{p}_k) for k=1 to b
        # \hat{Pf_b} is log-normal
        # Based on the delta method: note that the SuS code already returns the corrected c.o.v. of the log(\hat{p}_b)
        # for each k, \hat{p}_k = P(x \in A_k| A_{k-1})~=p0.
        # cov^2(\hat{Pf_b}) = Sum of cov^2(\hat{p}_k) = Sum of delta_k^2 for k=1 to b (stored in all_cov_2).

        alpha = 1 - ci_level / 100

        from scipy.stats import norm

        z = norm.ppf(1 - alpha / 2)

        margin = z * np.sqrt(all_cov_2)

        if log_y:  # in log scale
            lower = np.log(all_p_f) - margin
            upper = np.log(all_p_f) + margin
            y_ = np.log(all_p_f)
        else:
            lower = all_p_f * np.exp(-margin)
            upper = all_p_f * np.exp(margin)
            y_ = all_p_f

        x_ = b if not log_x else np.log(b)

        # make plots of Pf vs threshold with 95% CI limits for Pf
        fig, ax = plt.subplots()

        ax.plot(x_, y_, c="blue", label="Probability of failure")
        ax.fill_between(x_, lower, upper, color="blue", alpha=0.2)

        ax.set_xlabel(f"Threshold {'(Nat. log scale)' if log_x else ''}")
        ax.set_ylabel(f"Probability of failure {'(Nat. log scale)' if log_y else ''}")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show:
            plt.show()
        else:
            filename_suffix = f"{'_logX' if log_x else ''}{'_logY' if log_y else ''}"
            plt.savefig(
                f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/Pf_vs_threshold_{epsilon}{filename_suffix}.pdf",
                dpi=dpi,
            )
            plt.close()

    def get_all_obs_P_f_vs_threshold(
        self,
        noise_label,
        epsilon,
        log_y=True,
        log_x=True,
        smoothness=0.2,
        use_true_thresh=False,
        prob_thresh=0.5,
        SuS_run_id=0,
    ):
        """
        Get the max curvature point for all observations for a given noise type and epsilon value and plot the Pf vs threshold.
        :param noise_label: noise type
        :param epsilon: epsilon value
        :param log_y: bool, whether to plot the y axis in log scale in Pf_vs_threshold plot
        :param log_x: bool, whether to plot the x axis in log scale in Pf_vs_threshold plot
        :param smoothness: float, smoothness parameter for the curve approximation
        :param use_true_thresh: bool, whether to use actual vector of thresholds that was reached (True) or the b_line data (False)
        :param prob_thresh: float, probability threshold to stop the plot at
        :param SuS_run_id: int, SuS run id
        :return:
        """
        print("Getting all observations Pf vs threshold and max curvature...")

        all_obs_results = self.all_obs_results
        all_obs_max_curvature = {}
        noise_norm_vs_max_curvature = []
        max_curv_pf = []
        suffix = "trueB" if use_true_thresh else "bLine"

        for obs_idx in all_obs_results.keys():
            if not use_true_thresh:
                x = all_obs_results[obs_idx][noise_label][epsilon]["b_line"][SuS_run_id]
                y = all_obs_results[obs_idx][noise_label][epsilon]["Pf_line"][
                    SuS_run_id
                ]
            else:
                x = all_obs_results[obs_idx][noise_label][epsilon]["all_thresholds"][
                    SuS_run_id
                ]
                y = np.cumprod(
                    all_obs_results[obs_idx][noise_label][epsilon]["all_prob"][
                        SuS_run_id
                    ]
                )

            max_curvature_x, max_curvature_y = self.get_max_curvature_point(
                x,
                y,
                obs_idx,
                noise_label,
                epsilon,
                log_y=True,
                smoothness=smoothness,
                plot_curvature=True,
                prob_thresh=prob_thresh,
                suffix=suffix,
            )

            all_obs_max_curvature[(obs_idx, noise_label, epsilon)] = (
                max_curvature_x,
                max_curvature_y,
            )

            max_curv_pf.append(max_curvature_y)

            # ||noise|| - max curvature locatios
            noise_norm_vs_max_curvature.append(
                np.abs(
                    max_curvature_x
                    - self.inverted_obs_vectors[obs_idx][noise_label]["noise_norm"]
                )[0]
            )

            self.plot_pf_vs_threshold(
                obs_idx,
                noise_label,
                epsilon,
                log_y=log_y,
                log_x=log_x,
                prob_thresh=prob_thresh,
            )

        self.all_obs_max_curvature = all_obs_max_curvature
        save_to_disk(
            self.all_obs_max_curvature,
            f"{self.inference_root_dir}/{noise_label}/all_obs_max_curvature_{suffix}.pkl",
        )
        save_to_disk(
            self.all_obs_max_curvature,
            f"{self.inference_root_dir}/{noise_label}/all_obs_max_curvature_{suffix}.txt",
            _pickle=False,
            _text=True,
        )

        noise_vs_curv_stats = compute_array_stats(np.array(noise_norm_vs_max_curvature))
        save_to_disk(
            noise_norm_vs_max_curvature,
            f"{self.inference_root_dir}/{noise_label}/noise_norm_vs_max_curvature.pkl",
        )
        save_to_disk(
            noise_vs_curv_stats,
            f"{self.inference_root_dir}/{noise_label}/noise_vs_curv_stats.txt",
            _text=True,
            _pickle=False,
        )

        max_curv_pf_stats = compute_array_stats(np.array(max_curv_pf))
        save_to_disk(
            max_curv_pf_stats,
            f"{self.inference_root_dir}/{noise_label}/max_curv_pf_stas.txt",
            _text=True,
            _pickle=False,
        )

        return all_obs_max_curvature

    def find_min_median_max_values(self, values):
        """
        Find the examples with min, median and max values.
        :param values: numpy array of values to find min, median and max from.
        """
        k = 3  # number of examples to return for each level

        median_loc = np.where(
            values.reshape(-1)
            == np.quantile(values.reshape(-1), 0.5, interpolation="nearest")
        )[0][
            0
        ]  # ids of the k examples at median value

        sort_order = np.argsort(values)

        k_min_idx = sort_order[0:k]  # ids of the k examples with min values
        k_min_values = values[k_min_idx]  # the k values at minimum

        k_max_idx = sort_order[-k:]  # ids of the k examples with max values
        k_max_values = values[k_max_idx]  # the k values at maximum

        median_value_loc_arg = np.where(sort_order == median_loc)[0][
            0
        ]  # location of the median value in the sorted array
        k_median_idx = sort_order[median_value_loc_arg - 1 : median_value_loc_arg + 2]
        k_median_values = values[k_median_idx]

        return {
            "min_ids": k_min_idx,
            "min": k_min_values,
            "median_ids": k_median_idx,
            "median": k_median_values,
            "max_ids": k_max_idx,
            "max": k_max_values,
        }

    def compute_ssim(self, samples, ref, height, width):
        """
        Compute the SSIM between the samples and the reference.
        :param samples: numpy array of samples of shape (sample_size, dim)
        :param ref: numpy array of reference of shape (1, dim)
        :return:
        """
        from skimage.metrics import structural_similarity as ssim

        ssim_values = []
        range_min = min(np.min(ref), np.min(samples))
        range_max = max(np.max(ref), np.max(samples))
        data_range = range_max - range_min

        for sample in samples:
            ssim_values.append(
                ssim(
                    ref.reshape(height, width),
                    sample.reshape(height, width),
                    data_range=data_range,
                )
            )

        return ssim_values

    def make_rmse_stats(
        self,
        plot_samples=True,
        random_samples=False,
        make_boxplots=True,
        compute_ssim=False,
        dpi=600,
        show=False,
    ):
        """
        Make RMSE statistics for all observations and noise types and epsilon values. Compute for inverted_x vs x^*,
        inverted_y vs y_obs and resimulated_inverted_x vs y_obs.
        :param plot_samples: whether to plot samples of inverted_x, inverted_y and resimulated_inverted_x at min, median and max RMSE values for
        different thresholds
        :param make_boxplots: whether to make boxplots of the RMSE values for different thresholds
        :param compute_ssim: whether to compute the SSIM between each x sample and the ground truth x
        :param dpi: dpi for the plots
        :param show: whether to show the plots or save them to disk
        """

        print("Making RMSE statistics for all observations...")

        from fastabc_inversion.geo_problems.utils.evaluation.scorers import \
            torch_rmse

        dim_x = self.experiment_obj.dim_x
        dim_y = self.experiment_obj.dim_y
        nx = self.experiment_obj.nx
        ny = self.experiment_obj.ny

        self.all_obs_rmse_threshold = {}
        self.all_obs_rmse_y_threshold = {}
        self.all_obs_rmse_resims_threshold = {}

        # find examples and prepare to plot
        def plot_examples(
            rmse_vec,
            inverted_x,
            ref_x,
            file_name,
            random_samples=False,
            return_rmse_x=False,
            rmse_x_vec=None,
            ssim_vec=None,
            dpi=dpi,
            show=show,
        ):
            """
            Plot examples of inverted_x
            :param rmse_vec: rmse values to select examples based on its content when looking for min, median
                and max values. Could be values of rmse_x, rmse_y or rmse_resims.
            :param inverted_x: source of examples to plot
            :param ref_x: reference example to plot alongside the inverted_x examples.
            :param file_name: plot file name to save the examples to.
            :param random_samples: whether to select random samples from the inverted_x
                instead of min, median and max of rmse_vec values.
            :param return_rmse_x: whether to add the RMSE X values to the labels of the examples.
            :param rmse_x_vec: rmse_x values to use for the labels of the examples if return_rmse_x is True, and to use
                when selecting random samples (not min, median and max values).
            :param ssim_vec: ssim values to use for the labels of the examples if provided.
            :return:
            """
            k = 3

            if return_rmse_x:
                if rmse_x_vec is None:
                    raise ValueError(
                        "Please provide the rmse_x_vec to return the RMSE X values of the examples."
                    )

            if not random_samples:
                # if not random samples, then we take the min, median and max RMSE values
                rmse_samples_to_plot = self.find_min_median_max_values(rmse_vec)
                k_min_idx = rmse_samples_to_plot["min_ids"]
                k_min = (
                    rmse_samples_to_plot["min"]
                    if not return_rmse_x
                    else rmse_x_vec[k_min_idx]
                )

                k_median_idx = rmse_samples_to_plot["median_ids"]
                k_median = (
                    rmse_samples_to_plot["median"]
                    if not return_rmse_x
                    else rmse_x_vec[k_median_idx]
                )

                k_max_idx = rmse_samples_to_plot["max_ids"]
                k_max = (
                    rmse_samples_to_plot["max"]
                    if not return_rmse_x
                    else rmse_x_vec[k_max_idx]
                )

            else:
                random_indices = np.random.choice(
                    np.arange(inverted_x.shape[0]), size=k * 3, replace=False
                )
                k_min_idx = random_indices[:k]
                k_min = (
                    rmse_vec[k_min_idx] if not return_rmse_x else rmse_x_vec[k_min_idx]
                )

                k_median_idx = random_indices[k : k * 2]
                k_median = (
                    rmse_vec[k_median_idx]
                    if not return_rmse_x
                    else rmse_x_vec[k_median_idx]
                )

                k_max_idx = random_indices[k * 2 :]
                k_max = (
                    rmse_vec[k_max_idx] if not return_rmse_x else rmse_x_vec[k_max_idx]
                )

            # prepare the examples to plot
            examples_min = np.concatenate(
                (
                    ref_x.reshape(1, ny * nx),
                    inverted_x[k_min_idx, :],
                ),
                axis=0,
            ).reshape(1, k + 1, -1)
            examples_median = np.concatenate(
                (
                    ref_x.reshape(1, ny * nx),
                    inverted_x[k_median_idx, :],
                ),
                axis=0,
            ).reshape(1, k + 1, -1)
            examples_max = np.concatenate(
                (
                    ref_x.reshape(1, ny * nx),
                    inverted_x[k_max_idx, :],
                ),
                axis=0,
            ).reshape(1, k + 1, -1)

            examples = np.concatenate(
                (examples_min, examples_median, examples_max), axis=0
            )
            examples = examples.reshape(-1, k + 1, ny * nx)

            if ssim_vec is not None:
                k_min_ssim = ssim_vec[k_min_idx]
                k_median_ssim = ssim_vec[k_median_idx]
                k_max_ssim = ssim_vec[k_max_idx]
                ssim_labels = [k_min_ssim, k_median_ssim, k_max_ssim]
            else:
                ssim_labels = None

            plot.plot_samples(
                examples,
                rmse_labels=[k_min, k_median, k_max],
                ssim_labels=ssim_labels,
                width=nx,
                height=ny,
                grd_truth=True,
                save_location=file_name,
                dpi=dpi,
                show=show,
                figsize=(15, 6),
            )

        for obs_idx in self.inverted_obs:
            ref_x = self.test_x[obs_idx, :].reshape(
                1, dim_x
            )  # Note: test data was never normalized
            # ref_y = self.test_y[obs_idx, :].reshape(1, dim_y)

            for noise_label in self.noise_list:
                y_obs = self.inverted_obs_vectors[obs_idx][noise_label]["obs"].reshape(
                    1, dim_y
                )

                N = self.inference_params[noise_label]["N"]

                for epsilon in self.epsilon_values[noise_label]:
                    self.all_obs_rmse_threshold[(obs_idx, noise_label, epsilon)] = []
                    self.all_obs_rmse_y_threshold[(obs_idx, noise_label, epsilon)] = []
                    self.all_obs_rmse_resims_threshold[
                        (obs_idx, noise_label, epsilon)
                    ] = []

                    min_mean_rmse_x = float("inf")
                    min_mean_rmse_x_threshold = 0
                    min_mean_rmse_y = float("inf")
                    min_mean_rmse_y_threshold = 0
                    min_mean_rmse_resims = float("inf")
                    min_mean_rmse_resims_threshold = 0

                    for i in range(
                        len(
                            self.all_thresholds_inverted_x[
                                (obs_idx, noise_label, epsilon)
                            ]
                        )
                    ):
                        # get threshold i
                        current_threshold = self.all_thresholds[
                            (obs_idx, noise_label, epsilon)
                        ][i]

                        inverted_x = self.all_thresholds_inverted_x[
                            (obs_idx, noise_label, epsilon)
                        ][i].reshape(-1, dim_x)
                        inverted_y = self.all_thresholds_inverted_y[
                            (obs_idx, noise_label, epsilon)
                        ][i].reshape(-1, dim_y)
                        resimulated_inverted_x = (
                            self.all_thresholds_resimulated_inverted_x[
                                (obs_idx, noise_label, epsilon)
                            ][i].reshape(-1, dim_y)
                        )

                        rmse_x = torch_rmse(ref_x, inverted_x, on_gpu=True)
                        mean_rmse_x = np.mean(rmse_x)
                        rmse_y = torch_rmse(y_obs, inverted_y, on_gpu=True)
                        mean_rmse_y = np.mean(rmse_y)
                        rmse_resims = torch_rmse(
                            y_obs, resimulated_inverted_x, on_gpu=True
                        )
                        mean_rmse_resims = np.mean(rmse_resims)

                        if compute_ssim:
                            ssim_vec = np.array(
                                self.compute_ssim(inverted_x, ref_x, ny, nx)
                            )
                        else:
                            ssim_vec = None

                        if mean_rmse_x < min_mean_rmse_x:
                            min_mean_rmse_x = mean_rmse_x
                            min_mean_rmse_x_threshold = current_threshold

                        if mean_rmse_y < min_mean_rmse_y:
                            min_mean_rmse_y = mean_rmse_y
                            min_mean_rmse_y_threshold = current_threshold

                        if mean_rmse_resims < min_mean_rmse_resims:
                            min_mean_rmse_resims = mean_rmse_resims
                            min_mean_rmse_resims_threshold = current_threshold

                        self.all_obs_rmse_threshold[
                            (obs_idx, noise_label, epsilon)
                        ].append(rmse_x)
                        self.all_obs_rmse_y_threshold[
                            (obs_idx, noise_label, epsilon)
                        ].append(rmse_y)
                        self.all_obs_rmse_resims_threshold[
                            (obs_idx, noise_label, epsilon)
                        ].append(rmse_resims)

                        if plot_samples:
                            # makes samples dir :
                            os.makedirs(
                                f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/samples_plots",
                                exist_ok=True,
                            )
                            # rmse_x_samples_to_plot = self.find_min_median_max_values(rmse_x)
                            # rmse_y_samples_to_plot = self.find_min_median_max_values(rmse_y)
                            # rmse_resims_samples_to_plot = self.find_min_median_max_values(rmse_resims)

                            # examples based on rmse_x
                            save_location_file = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/samples_plots/posteriorExamples_MinMedianMaxRMSE_x_ex_{obs_idx}_{current_threshold}.pdf"
                            plot_examples(
                                rmse_x,
                                inverted_x,
                                ref_x,
                                save_location_file,
                                ssim_vec=ssim_vec,
                                random_samples=random_samples,
                            )

                            # examples based on rmse_y
                            save_location_file = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/samples_plots/posteriorExamples_MinMedianMaxRMSE_y_ex_{obs_idx}_{current_threshold}.pdf"
                            plot_examples(
                                rmse_y,
                                inverted_x,
                                ref_x,
                                save_location_file,
                                return_rmse_x=True,
                                rmse_x_vec=rmse_x,
                                ssim_vec=ssim_vec,
                                random_samples=random_samples,
                            )

                            # examples based on rmse_resims
                            save_location_file = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/samples_plots/posteriorExamples_MinMedianMaxRMSE_resims_ex_{obs_idx}_{current_threshold}.pdf"
                            plot_examples(
                                rmse_resims,
                                inverted_x,
                                ref_x,
                                save_location_file,
                                return_rmse_x=True,
                                rmse_x_vec=rmse_x,
                                ssim_vec=ssim_vec,
                                random_samples=random_samples,
                            )

                    with open(
                        f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/min_rmse_thresholds_{epsilon}.txt",
                        "w",
                    ) as f:
                        f.write(
                            f"min mean rmse x : {min_mean_rmse_x} @ threshold : {min_mean_rmse_x_threshold}"
                        )
                        f.write(
                            f"min mean rmse y : {min_mean_rmse_y} @ threshold : {min_mean_rmse_y_threshold}"
                        )
                        f.write(
                            f"min mean rmse resims : {min_mean_rmse_resims} @ threshold : {min_mean_rmse_resims_threshold}"
                        )

                    if make_boxplots:
                        # make boxplots of the RMSE values for different thresholds for a given noise type and for a given obs
                        # (assuming (explicit) epsilon is the same for all obs)
                        num_thresholds = len(
                            self.all_thresholds[(obs_idx, noise_label, epsilon)]
                        )

                        sorted_args = np.argsort(
                            self.all_thresholds[(obs_idx, noise_label, epsilon)]
                        )
                        sorted_args = np.array(sorted_args, dtype=int)
                        # make boxplot labels for the thresholds in log scale and round to 2 decimal places and avoid duplicates
                        # labels = [f"{round(np.log10(self.all_thresholds[(obs_idx, noise_label, epsilon)][i]), 2)}" for i in sorted_args]
                        # labels = [labels[i] if labels.count(labels[i]) == 1 else f"{labels[i]}_{i}" for i in range(num_thresholds)]
                        labels = [f"{i}" for i in sorted_args]

                        # plot.plot_boxplot
                        data_rmse_x_vs_thresh = np.array(
                            self.all_obs_rmse_threshold[(obs_idx, noise_label, epsilon)]
                        )[sorted_args]
                        data_rmse_x_vs_thresh = data_rmse_x_vs_thresh.reshape(
                            1, num_thresholds, N
                        )
                        titles = [
                            [
                                r"$RMSE(x^*, x_{post})$ vs SuS thresholds; min mean = %.3f @ %.3f"
                                % (min_mean_rmse_x, min_mean_rmse_x_threshold)
                            ],
                            "SuS threshold index",
                            r"$RMSE(x^*, x_{post})$",
                        ]
                        file_name = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/Boxplots_RMSE_x_vs_threshold_{epsilon}.pdf"
                        plot.plot_boxplots(
                            data_rmse_x_vs_thresh,
                            labels,
                            axes_plot_titles=titles,
                            lower_lim=0,
                            save_location=file_name,
                            dpi=dpi,
                            show=show,
                        )

                        data_rmse_y_vs_thresh = np.array(
                            self.all_obs_rmse_y_threshold[
                                (obs_idx, noise_label, epsilon)
                            ]
                        )[sorted_args]
                        data_rmse_y_vs_thresh = data_rmse_y_vs_thresh.reshape(
                            1, num_thresholds, N
                        )
                        titles = [
                            [
                                r"$RMSE(y_{obs}, y_{post})$ vs SuS thresholds; min mean = %.3f @ %.3f"
                                % (min_mean_rmse_y, min_mean_rmse_y_threshold)
                            ],
                            "SuS threshold index",
                            r"$RMSE(y_{obs}, y_{post})$",
                        ]
                        file_name = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/Boxplots_RMSE_y_vs_threshold_{epsilon}.pdf"
                        plot.plot_boxplots(
                            data_rmse_y_vs_thresh,
                            labels,
                            axes_plot_titles=titles,
                            lower_lim=0,
                            save_location=file_name,
                            dpi=dpi,
                            show=show,
                        )

                        data_rmse_resims_vs_thresh = np.array(
                            self.all_obs_rmse_resims_threshold[
                                (obs_idx, noise_label, epsilon)
                            ]
                        )[sorted_args]
                        data_rmse_resims_vs_thresh = data_rmse_resims_vs_thresh.reshape(
                            1, num_thresholds, N
                        )
                        titles = [
                            [
                                r"$RMSE(y_{obs}, y_{resim})$ vs SuS thresholds; min mean = %.3f @ %.3f"
                                % (min_mean_rmse_resims, min_mean_rmse_resims_threshold)
                            ],
                            "SuS threshold index",
                            r"$RMSE(y_{obs}, y_{resim})$",
                        ]
                        file_name = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/Boxplots_RMSE_resims_vs_threshold_{epsilon}.pdf"
                        plot.plot_boxplots(
                            data_rmse_resims_vs_thresh,
                            labels,
                            axes_plot_titles=titles,
                            lower_lim=0,
                            save_location=file_name,
                            dpi=dpi,
                            show=show,
                        )

        save_to_disk(
            self.all_obs_rmse_threshold,
            f"{self.inference_root_dir}/all_obs_rmse_threshold.pkl",
        )
        save_to_disk(
            self.all_obs_rmse_y_threshold,
            f"{self.inference_root_dir}/all_obs_rmse_y_threshold.pkl",
        )
        save_to_disk(
            self.all_obs_rmse_resims_threshold,
            f"{self.inference_root_dir}/all_obs_rmse_resims_threshold.pkl",
        )

    def make_sinkhorn_stats(
        self,
        sinkhorn_params,
        bootstraps=1,
        bootstrap_replace=False,
        on_gpu=True,
        make_plots=True,
        dpi=600,
        show=False,
    ):
        """
        Make Sinkhorn divergence for all observations and noise types and epsilon values, per threshold.
        :param sinkhorn_params: dict, parameters for the Sinkhorn divergence algorithm.
               Should contain the following keys:
               - epsilon: float, regularization parameter
               - niter: int, number of iterations
               - p: int, power for the cost function
        :param on_gpu: bool, whether to use GPU or not
        :param make_plots: bool, whether to make plots of ES and VS
        :param dpi: int, dpi for the plots
        :param show: bool, whether to show the plots or save them to disk
        """
        import fastabc_inversion.geo_problems.utils.sinkhorn.sinkhorn_pointcloud as spc

        print("Making Sinkhorn divergence for all observations...")

        dim_x = self.experiment_obj.dim_x
        dim_y = self.experiment_obj.dim_y

        # get sinkhorn parameters from the input dict
        if not isinstance(sinkhorn_params, dict):
            raise ValueError("sinkhorn_params should be a dictionary.")

        if on_gpu:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # sample size for sinkhorn divergence
        spc_params = {
            "epsilon": sinkhorn_params.get("epsilon", 100),
            "n": sinkhorn_params.get("n", 500),
            "niter": sinkhorn_params.get("niter", 100),
            "p": sinkhorn_params.get("p", 2),
            "device": device,
        }

        self.sinkhorn_stats = {}
        self.sinkhorn_stats_y = {}
        self.sinkhorn_stats_resims = {}

        self.all_obs_min_es

        for obs_idx in self.inverted_obs:
            ref_x = self.test_x[obs_idx, :].reshape(
                1, dim_x
            )  # ground truth (unnormalized by default)
            # make torch tensor if not already
            if not isinstance(ref_x, torch.Tensor):
                ref_x = torch.tensor(ref_x, dtype=torch.float32, device=device)

            for noise_label in self.noise_list:
                N = self.inference_params[noise_label]["N"]

                m = (
                    N // 3
                )  # number of samples to use for sinkhorn divergence (to reduce computational cost)
                spc_params["n"] = m

                # repeat ref_x m times to match the sample size, using tensors operations
                ref_x = ref_x.repeat(m, 1)  # ref_x is (1, dim_x), now (m, dim_x)

                y_obs = self.inverted_obs_vectors[obs_idx][noise_label]["obs"].reshape(
                    1, dim_y
                )  # observation
                # make torch tensor if not already
                if not isinstance(y_obs, torch.Tensor):
                    y_obs = torch.tensor(y_obs, dtype=torch.float32, device=device)

                # repeat y_obs m times to match the sample size, using tensors operations
                y_obs = y_obs.repeat(m, 1)  # y_obs is (1, dim_y), now (m, dim_y)

                for epsilon in self.epsilon_values[noise_label]:
                    self.sinkhorn_stats[(obs_idx, noise_label, epsilon)] = []
                    self.sinkhorn_stats_y[(obs_idx, noise_label, epsilon)] = []
                    self.sinkhorn_stats_resims[(obs_idx, noise_label, epsilon)] = []

                    min_mean_sink = float("inf")
                    min_mean_sink_threshold = 0
                    min_mean_sink_y = float("inf")
                    min_mean_sink_y_threshold = 0
                    min_mean_sink_resims = float("inf")
                    min_mean_sink_resims_threshold = 0

                    for i in range(
                        len(
                            self.all_thresholds_inverted_x[
                                (obs_idx, noise_label, epsilon)
                            ]
                        )
                    ):
                        # for each intermediate threshold

                        # get inverted x values
                        inverted_x = self.all_thresholds_inverted_x[
                            (obs_idx, noise_label, epsilon)
                        ][i].reshape(N, dim_x)
                        # make torch tensor if not already
                        if not isinstance(inverted_x, torch.Tensor):
                            inverted_x = torch.tensor(
                                inverted_x, dtype=torch.float32, device=device
                            )

                        # get inverted y values
                        inverted_y = self.all_thresholds_inverted_y[
                            (obs_idx, noise_label, epsilon)
                        ][i].reshape(N, dim_y)
                        # make torch tensor if not already
                        if not isinstance(inverted_y, torch.Tensor):
                            inverted_y = torch.tensor(
                                inverted_y, dtype=torch.float32, device=device
                            )

                        # get resimulated inverted x values
                        resimulated_x = self.all_thresholds_resimulated_inverted_x[
                            (obs_idx, noise_label, epsilon)
                        ][i].reshape(N, dim_y)
                        # make torch tensor if not already
                        if not isinstance(resimulated_x, torch.Tensor):
                            resimulated_x = torch.tensor(
                                resimulated_x, dtype=torch.float32, device=device
                            )

                        boot_sink = []
                        boot_sink_y = []
                        boot_sink_resims = []

                        for _ in range(bootstraps):
                            # sample N//bootstraps indices
                            indices = np.random.choice(N, m, replace=bootstrap_replace)

                            # get the bootstrapped inverted x, y and resimulated x
                            boot_inverted_x = inverted_x[indices, :]
                            boot_inverted_y = inverted_y[indices, :]
                            boot_resimulated_x = resimulated_x[indices, :]

                            boot_sink.append(
                                spc.sinkhorn_normalized(
                                    ref_x, boot_inverted_x, **spc_params
                                )
                            )
                            boot_sink_y.append(
                                spc.sinkhorn_normalized(
                                    y_obs, boot_inverted_y, **spc_params
                                )
                            )
                            boot_sink_resims.append(
                                spc.sinkhorn_normalized(
                                    y_obs, boot_resimulated_x, **spc_params
                                )
                            )

                        # compute mean of the bootstrapped sinkhorn divergences
                        mean_sink = np.mean(boot_sink)
                        if mean_sink < min_mean_sink:
                            min_mean_sink = mean_sink
                            min_mean_sink_threshold = self.all_thresholds[
                                (obs_idx, noise_label, epsilon)
                            ][i]

                        mean_sink_y = np.mean(boot_sink_y)
                        if mean_sink_y < min_mean_sink_y:
                            min_mean_sink_y = mean_sink_y
                            min_mean_sink_y_threshold = self.all_thresholds[
                                (obs_idx, noise_label, epsilon)
                            ][i]

                        mean_sink_resims = np.mean(boot_sink_resims)
                        if mean_sink_resims < min_mean_sink_resims:
                            min_mean_sink_resims = mean_sink_resims
                            min_mean_sink_resims_threshold = self.all_thresholds[
                                (obs_idx, noise_label, epsilon)
                            ][i]

                        self.sinkhorn_stats[(obs_idx, noise_label, epsilon)].append(
                            np.array(boot_inverted_x).flatten()
                        )
                        self.sinkhorn_stats_y[(obs_idx, noise_label, epsilon)].append(
                            np.array(boot_inverted_y).flatten()
                        )
                        self.sinkhorn_stats_resims[
                            (obs_idx, noise_label, epsilon)
                        ].append(np.array(boot_resimulated_x).flatten())

    def make_scores_stats(
        self,
        es_p=2,
        vs_p=0.5,
        bootstraps=1,
        bootstrap_replace=False,
        make_plots=True,
        dpi=600,
        show=False,
    ):
        """
        Make Energy score (ES) and Variogram score (VS) for all observations and noise types and epsilon values, per
        threshold.
        :param es_p: int, power for ES
        :param vs_p: int, power for VS
        :param bootstraps: int, number of bootstraps for computing multiple estimations of ES and VS
        :param bootstrap_replace: bool, whether to sample with replacement
        :param make_plots: bool, whether to make plots of ES and VS
        :param dpi: int, dpi for the plots
        :param show: bool, whether to show the plots or save them to disk
        :return:
        """
        from fastabc_inversion.geo_problems.utils.evaluation.scorers import (
            torch_es, torch_vs)

        print("Making ES and VS for all observations...")

        dim_x = self.experiment_obj.dim_x
        dim_y = self.experiment_obj.dim_y

        # choose kernels to use :
        # if es_p == 2:
        #    kernel_es = pairwise_distances  # kernel version is slightly faster than scorers.es
        #    kwargs_es = {"metric": "euclidean", "squared": True}
        # if es_p == 1:
        #    kernel_es = pairwise_distances
        #    kwargs_es = {"metric": 'l1'}

        # check if es_p is a list, and make it a list if single value
        if isinstance(es_p, int) or isinstance(es_p, float):
            es_p = [es_p]
        if not isinstance(es_p, list):
            raise ValueError("es_p should be a list of integers or a single integer.")

        if isinstance(vs_p, int) or isinstance(vs_p, float):
            vs_p = [vs_p]
        if not isinstance(vs_p, list):
            raise ValueError("vs_p should be a list of integers or a single integer.")

        self.es_stats = {}  # \tilde{x} vs x*
        self.vs_stats = {}
        self.es_stats_y = {}  # \tilde{y} vs y_obs
        self.vs_stats_y = {}
        self.es_stats_resims = {}  # Resim(\tilde{x}) vs y_obs
        self.vs_stats_resims = {}

        self.all_obs_min_es = {}
        self.all_obs_min_vs = {}
        self.all_obs_min_es_y = {}
        self.all_obs_min_vs_y = {}
        self.all_obs_min_es_resims = {}
        self.all_obs_min_vs_resims = {}

        for obs_idx in self.inverted_obs:
            ref_x = self.test_x[obs_idx, :].reshape(
                1, dim_x
            )  # ground truth (unnormalized by default)

            for noise_label in self.noise_list:
                y_obs = self.inverted_obs_vectors[obs_idx][noise_label]["obs"].reshape(
                    1, dim_y
                )  # observation

                for epsilon in self.epsilon_values[noise_label]:
                    self.es_stats[
                        (obs_idx, noise_label, epsilon)
                    ] = {}  # one key per es_p value
                    self.vs_stats[
                        (obs_idx, noise_label, epsilon)
                    ] = {}  # one key per vs_p value

                    self.es_stats_y[(obs_idx, noise_label, epsilon)] = {}
                    self.vs_stats_y[(obs_idx, noise_label, epsilon)] = {}

                    self.es_stats_resims[(obs_idx, noise_label, epsilon)] = {}
                    self.vs_stats_resims[(obs_idx, noise_label, epsilon)] = {}

                    self.all_obs_min_es[(obs_idx, noise_label, epsilon)] = {}
                    self.all_obs_min_es_y[(obs_idx, noise_label, epsilon)] = {}
                    self.all_obs_min_es_resims[(obs_idx, noise_label, epsilon)] = {}

                    self.all_obs_min_vs[(obs_idx, noise_label, epsilon)] = {}
                    self.all_obs_min_vs_y[(obs_idx, noise_label, epsilon)] = {}
                    self.all_obs_min_vs_resims[(obs_idx, noise_label, epsilon)] = {}

                    min_es = {}
                    min_es_threshold = {}
                    min_es_y = {}
                    min_es_y_threshold = {}
                    min_es_resims = {}
                    min_es_resims_threshold = {}

                    for p in es_p:
                        self.es_stats[(obs_idx, noise_label, epsilon)][p] = []
                        self.es_stats_y[(obs_idx, noise_label, epsilon)][p] = []
                        self.es_stats_resims[(obs_idx, noise_label, epsilon)][p] = []

                        self.all_obs_min_es[(obs_idx, noise_label, epsilon)][p] = None
                        self.all_obs_min_es_y[(obs_idx, noise_label, epsilon)][p] = None
                        self.all_obs_min_es_resims[(obs_idx, noise_label, epsilon)][
                            p
                        ] = None

                        min_es[p] = float("inf")
                        min_es_threshold[p] = 0
                        min_es_y[p] = float("inf")
                        min_es_y_threshold[p] = 0
                        min_es_resims[p] = float("inf")
                        min_es_resims_threshold[p] = 0

                    min_vs = {}
                    min_vs_threshold = {}
                    min_vs_y = {}
                    min_vs_y_threshold = {}
                    min_vs_resims = {}
                    min_vs_resims_threshold = {}
                    for p in vs_p:
                        self.vs_stats[(obs_idx, noise_label, epsilon)][p] = []
                        self.vs_stats_y[(obs_idx, noise_label, epsilon)][p] = []
                        self.vs_stats_resims[(obs_idx, noise_label, epsilon)][p] = []

                        self.all_obs_min_vs[(obs_idx, noise_label, epsilon)][p] = None
                        self.all_obs_min_vs_y[(obs_idx, noise_label, epsilon)][p] = None
                        self.all_obs_min_vs_resims[(obs_idx, noise_label, epsilon)][
                            p
                        ] = None

                        min_vs[p] = float("inf")
                        min_vs_threshold[p] = 0
                        min_vs_y[p] = float("inf")
                        min_vs_y_threshold[p] = 0
                        min_vs_resims[p] = float("inf")
                        min_vs_resims_threshold[p] = 0

                    for i in range(
                        len(
                            self.all_thresholds_inverted_x[
                                (obs_idx, noise_label, epsilon)
                            ]
                        )
                    ):
                        # for each threshold

                        N = self.inference_params[noise_label]["N"]

                        # get inverted x values
                        inverted_x = self.all_thresholds_inverted_x[
                            (obs_idx, noise_label, epsilon)
                        ][i].reshape(N, dim_x)
                        # get inverted y values
                        inverted_y = self.all_thresholds_inverted_y[
                            (obs_idx, noise_label, epsilon)
                        ][i].reshape(N, dim_y)
                        # get resimulated inverted x values
                        resimulated_x = self.all_thresholds_resimulated_inverted_x[
                            (obs_idx, noise_label, epsilon)
                        ][i].reshape(N, dim_y)

                        boot_es = {}
                        boot_es_y = {}
                        boot_es_resims = {}
                        for p in es_p:
                            boot_es[p] = []
                            boot_es_y[p] = []
                            boot_es_resims[p] = []

                        boot_vs = {}
                        boot_vs_y = {}
                        boot_vs_resims = {}
                        for p in vs_p:
                            boot_vs[p] = []
                            boot_vs_y[p] = []
                            boot_vs_resims[p] = []

                        # bootstrap ES and VS by sampling a number of n//bootstraps of the inverted x, y and resimulated x
                        for _ in range(bootstraps):
                            # sample N//bootstraps indices
                            indices = np.random.choice(
                                N, N // 3, replace=bootstrap_replace
                            )

                            # get the bootstrapped inverted x, y and resimulated x
                            boot_inverted_x = inverted_x[indices, :]
                            boot_inverted_y = inverted_y[indices, :]
                            boot_resimulated_x = resimulated_x[indices, :]

                            for p in es_p:
                                boot_es[p].append(
                                    torch_es(
                                        ref_x, boot_inverted_x, power=p, on_gpu=True
                                    )
                                )
                                boot_es_y[p].append(
                                    torch_es(
                                        y_obs, boot_inverted_y, power=p, on_gpu=True
                                    )
                                )
                                boot_es_resims[p].append(
                                    torch_es(
                                        y_obs, boot_resimulated_x, power=p, on_gpu=True
                                    )
                                )

                            for p in vs_p:
                                boot_vs[p].append(
                                    torch_vs(
                                        ref_x, boot_inverted_x, power=p, on_gpu=True
                                    )
                                )
                                boot_vs_y[p].append(
                                    torch_vs(
                                        y_obs, boot_inverted_y, power=p, on_gpu=True
                                    )
                                )
                                boot_vs_resims[p].append(
                                    torch_vs(
                                        y_obs, boot_resimulated_x, power=p, on_gpu=True
                                    )
                                )

                        for p in es_p:
                            # list of bootstraps lists of ES values
                            self.es_stats[(obs_idx, noise_label, epsilon)][p].append(
                                np.array(boot_es[p]).flatten()
                            )
                            self.es_stats_y[(obs_idx, noise_label, epsilon)][p].append(
                                np.array(boot_es_y[p]).flatten()
                            )
                            self.es_stats_resims[(obs_idx, noise_label, epsilon)][
                                p
                            ].append(np.array(boot_es_resims[p]).flatten())

                            if bootstraps == 1:
                                if boot_es[p][0] < min_es[p]:
                                    min_es[p] = boot_es[p][0]
                                    min_es_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                                if boot_es_y[p][0] < min_es_y[p]:
                                    min_es_y[p] = boot_es_y[p][0]
                                    min_es_y_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                                if boot_es_resims[p][0] < min_es_resims[p]:
                                    min_es_resims[p] = boot_es_resims[p][0]
                                    min_es_resims_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]
                            else:
                                mean_es = np.mean(np.array(boot_es[p]).flatten())
                                if mean_es < min_es[p]:
                                    min_es[p] = mean_es
                                    min_es_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                                mean_es_y = np.mean(np.array(boot_es_y[p]).flatten())
                                if mean_es_y < min_es_y[p]:
                                    min_es_y[p] = mean_es_y
                                    min_es_y_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                                mean_es_resims = np.mean(
                                    np.array(boot_es_resims[p]).flatten()
                                )
                                if mean_es_resims < min_es_resims[p]:
                                    min_es_resims[p] = mean_es_resims
                                    min_es_resims_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                        for p in vs_p:
                            # list of bootstraps lists of VS values
                            self.vs_stats[(obs_idx, noise_label, epsilon)][p].append(
                                np.array(boot_vs[p]).flatten()
                            )
                            self.vs_stats_y[(obs_idx, noise_label, epsilon)][p].append(
                                np.array(boot_vs_y[p]).flatten()
                            )
                            self.vs_stats_resims[(obs_idx, noise_label, epsilon)][
                                p
                            ].append(np.array(boot_vs_resims[p]).flatten())

                            if bootstraps == 1:
                                if boot_vs[p][0] < min_vs[p]:
                                    min_vs[p] = boot_vs[p][0]
                                    min_vs_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                                if boot_vs_y[p][0] < min_vs_y[p]:
                                    min_vs_y[p] = boot_vs_y[p][0]
                                    min_vs_y_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                                if boot_vs_resims[p][0] < min_vs_resims[p]:
                                    min_vs_resims[p] = boot_vs_resims[p][0]
                                    min_vs_resims_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]
                            else:
                                mean_vs = np.mean(np.array(boot_vs[p]).flatten())
                                if mean_vs < min_vs[p]:
                                    min_vs[p] = mean_vs
                                    min_vs_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                                mean_vs_y = np.mean(np.array(boot_vs_y[p]).flatten())
                                if mean_vs_y < min_vs_y[p]:
                                    min_vs_y[p] = mean_vs_y
                                    min_vs_y_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                                mean_vs_resims = np.mean(
                                    np.array(boot_vs_resims[p]).flatten()
                                )
                                if mean_vs_resims < min_vs_resims[p]:
                                    min_vs_resims[p] = mean_vs_resims
                                    min_vs_resims_threshold[p] = self.all_thresholds[
                                        (obs_idx, noise_label, epsilon)
                                    ][i]

                    for p in es_p:
                        self.all_obs_min_es[(obs_idx, noise_label, epsilon)][p] = (
                            min_es[p],
                            min_es_threshold[p],
                        )
                        self.all_obs_min_es_y[(obs_idx, noise_label, epsilon)][p] = (
                            min_es_y[p],
                            min_es_y_threshold[p],
                        )
                        self.all_obs_min_es_resims[(obs_idx, noise_label, epsilon)][
                            p
                        ] = (min_es_resims[p], min_es_resims_threshold[p])
                    for p in vs_p:
                        self.all_obs_min_vs[(obs_idx, noise_label, epsilon)][p] = (
                            min_vs[p],
                            min_vs_threshold[p],
                        )
                        self.all_obs_min_vs_y[(obs_idx, noise_label, epsilon)][p] = (
                            min_vs_y[p],
                            min_vs_y_threshold[p],
                        )
                        self.all_obs_min_vs_resims[(obs_idx, noise_label, epsilon)][
                            p
                        ] = (min_vs_resims[p], min_vs_resims_threshold[p])

                    if make_plots:
                        b_vec = self.all_thresholds[(obs_idx, noise_label, epsilon)]

                        for p in es_p:
                            es_stats_vec = self.es_stats[
                                (obs_idx, noise_label, epsilon)
                            ][p]
                            save_path = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/ES_{p}_x_{epsilon}.pdf"
                            plot_title = rf"$ES_{p}(x^*, D_{{post}})$ vs SuS thresholds; min = {min_es[p]:.3f} @ {min_es_threshold[p]:.3f}, bootstraps = {bootstraps}"
                            plot_title = plot_title if bootstraps == 1 else [plot_title]
                            titles = [
                                plot_title,
                                f"SuS threshold index",
                                rf"$ES_{{{p}}}(x^*, D_{{post}})$",
                            ]
                            plot_stats(
                                es_stats_vec,
                                b_vec,
                                titles=titles,
                                file_name=save_path,
                                bootstraps=bootstraps,
                                dpi=dpi,
                                show=show,
                            )

                            es_stats_y = self.es_stats_y[
                                (obs_idx, noise_label, epsilon)
                            ][p]
                            save_path = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/ES_{p}_y_{epsilon}.pdf"
                            plot_title = rf"$ES_{p}(y_{{obs}}, D_{{Y_{{post}}}})$ vs SuS thresholds; min = {min_es_y[p]:.3f} @ {min_es_y_threshold[p]:.3f}, bootstraps = {bootstraps}"
                            plot_title = plot_title if bootstraps == 1 else [plot_title]
                            titles = [
                                plot_title,
                                f"SuS threshold index",
                                rf"$ES_{{{p}}}(y_{{obs}}, D_{{Y_{{post}}}})$",
                            ]
                            plot_stats(
                                es_stats_y,
                                b_vec,
                                titles=titles,
                                file_name=save_path,
                                bootstraps=bootstraps,
                                dpi=dpi,
                                show=show,
                            )

                            es_stats_resims = self.es_stats_resims[
                                (obs_idx, noise_label, epsilon)
                            ][p]
                            save_path = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/ES_{p}_resims_{epsilon}.pdf"
                            plot_title = rf"$ES_{p}(y_{{obs}}, D_{{Y_{{resim}}}})$ vs SuS thresholds; min = {min_es_resims[p]:.3f} @ {min_es_resims_threshold[p]:3f}, bootstraps = {bootstraps}"
                            plot_title = plot_title if bootstraps == 1 else [plot_title]
                            titles = [
                                plot_title,
                                f"SuS threshold index",
                                rf"$ES_{{{p}}}(y_{{obs}}, D_{{Y_{{resim}}}})$",
                            ]
                            plot_stats(
                                es_stats_resims,
                                b_vec,
                                titles=titles,
                                file_name=save_path,
                                bootstraps=bootstraps,
                                dpi=dpi,
                                show=show,
                            )

                        for p in vs_p:
                            vs_stats = self.vs_stats[(obs_idx, noise_label, epsilon)][p]
                            save_path = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/VS_{p}_x_{epsilon}.pdf"
                            plot_title = rf"$VS_{p}(x^*, D_{{post}})$ vs SuS thresholds; min = {min_vs[p]:.3f} @ {min_vs_threshold[p]:.3f}, bootstraps = {bootstraps}"
                            plot_title = plot_title if bootstraps == 1 else [plot_title]
                            titles = [
                                plot_title,
                                f"SuS threshold index",
                                rf"$VS_{{{p}}}(x^*, D_{{post}})$",
                            ]
                            plot_stats(
                                vs_stats,
                                b_vec,
                                titles=titles,
                                file_name=save_path,
                                bootstraps=bootstraps,
                                dpi=dpi,
                                show=show,
                            )

                            vs_stats_y = self.vs_stats_y[
                                (obs_idx, noise_label, epsilon)
                            ][p]
                            save_path = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/VS_{p}_y_{epsilon}.pdf"
                            plot_title = rf"$VS_{p}(y_{{obs}}, D_{{Y_{{post}}}})$ vs SuS thresholds; min = {min_vs_y[p]:.3f} @ {min_vs_y_threshold[p]:.3f}, bootstraps = {bootstraps}"
                            plot_title = plot_title if bootstraps == 1 else [plot_title]
                            titles = [
                                plot_title,
                                f"SuS threshold index",
                                rf"$VS_{{{p}}}(y_{{obs}}, D_{{Y_{{post}}}})$",
                            ]
                            plot_stats(
                                vs_stats_y,
                                b_vec,
                                titles=titles,
                                file_name=save_path,
                                bootstraps=bootstraps,
                                dpi=dpi,
                                show=show,
                            )

                            vs_stats_resims = self.vs_stats_resims[
                                (obs_idx, noise_label, epsilon)
                            ][p]
                            save_path = f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/VS_{p}_resims_{epsilon}.pdf"
                            plot_title = rf"$VS_{p}(y_{{obs}}, D_{{Y_{{resim}}}})$ vs SuS thresholds; min = {min_vs_resims[p]:.3f} @ {min_vs_resims_threshold[p]:.3f}, bootstraps = {bootstraps}"
                            plot_title = plot_title if bootstraps == 1 else [plot_title]
                            titles = [
                                plot_title,
                                f"SuS threshold index",
                                rf"$VS_{{{p}}}(y_{{obs}}, D_{{Y_{{resim}}}})$",
                            ]
                            plot_stats(
                                vs_stats_resims,
                                b_vec,
                                titles=titles,
                                file_name=save_path,
                                bootstraps=bootstraps,
                                dpi=dpi,
                                show=show,
                            )

                    save_to_disk(
                        self.es_stats[(obs_idx, noise_label, epsilon)],
                        f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/ES_{epsilon}.pkl",
                    )
                    save_to_disk(
                        self.vs_stats[(obs_idx, noise_label, epsilon)],
                        f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/VS_{epsilon}.pkl",
                    )

                    save_to_disk(
                        self.es_stats_y[(obs_idx, noise_label, epsilon)],
                        f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/ES_y_{epsilon}.pkl",
                    )
                    save_to_disk(
                        self.vs_stats_y[(obs_idx, noise_label, epsilon)],
                        f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/VS_y_{epsilon}.pkl",
                    )

                    save_to_disk(
                        self.es_stats_resims[(obs_idx, noise_label, epsilon)],
                        f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/ES_resims_{epsilon}.pkl",
                    )
                    save_to_disk(
                        self.vs_stats_resims[(obs_idx, noise_label, epsilon)],
                        f"{self.inverted_obs_vectors[obs_idx][noise_label]['obs_inference_dir']}/VS_resims_{epsilon}.pkl",
                    )

        save_to_disk(self.es_stats, f"{self.inference_root_dir}/es_stats.pkl")
        save_to_disk(self.vs_stats, f"{self.inference_root_dir}/vs_stats.pkl")
        save_to_disk(self.es_stats_y, f"{self.inference_root_dir}/es_stats_y.pkl")
        save_to_disk(self.vs_stats_y, f"{self.inference_root_dir}/vs_stats_y.pkl")
        save_to_disk(
            self.es_stats_resims, f"{self.inference_root_dir}/es_stats_resims.pkl"
        )
        save_to_disk(
            self.vs_stats_resims, f"{self.inference_root_dir}/vs_stats_resims.pkl"
        )

        save_to_disk(
            self.all_obs_min_es,
            f"{self.inference_root_dir}/all_obs_min_ES_x.txt",
            _pickle=False,
            _text=True,
        )
        save_to_disk(
            self.all_obs_min_vs,
            f"{self.inference_root_dir}/all_obs_min_VS_x.txt",
            _pickle=False,
            _text=True,
        )
        save_to_disk(
            self.all_obs_min_es_y,
            f"{self.inference_root_dir}/all_obs_min_ES_y.txt",
            _pickle=False,
            _text=True,
        )
        save_to_disk(
            self.all_obs_min_vs_y,
            f"{self.inference_root_dir}/all_obs_min_VS_y.txt",
            _pickle=False,
            _text=True,
        )
        save_to_disk(
            self.all_obs_min_es_resims,
            f"{self.inference_root_dir}/all_obs_min_ES_resims.txt",
            _pickle=False,
            _text=True,
        )
        save_to_disk(
            self.all_obs_min_vs_resims,
            f"{self.inference_root_dir}/all_obs_min_VS_resims.txt",
            _pickle=False,
            _text=True,
        )

        save_to_disk(
            self.all_obs_min_es, f"{self.inference_root_dir}/all_obs_min_ES_x.pkl"
        )
        save_to_disk(
            self.all_obs_min_vs, f"{self.inference_root_dir}/all_obs_min_VS_x.pkl"
        )
        save_to_disk(
            self.all_obs_min_es_y, f"{self.inference_root_dir}/all_obs_min_ES_y.pkl"
        )
        save_to_disk(
            self.all_obs_min_vs_y, f"{self.inference_root_dir}/all_obs_min_VS_y.pkl"
        )
        save_to_disk(
            self.all_obs_min_es_resims,
            f"{self.inference_root_dir}/all_obs_min_ES_resims.pkl",
        )
        save_to_disk(
            self.all_obs_min_vs_resims,
            f"{self.inference_root_dir}/all_obs_min_VS_resims.pkl",
        )

    def make_vs_training_refs(
        self,
        obs_idx_vec,
        metrics,
        bootstraps=10,
        m=500,
        make_summary_stats=True,
        load_existing=False,
    ):
        """
        Make metrics references between obs id test ground truth and samples from training (repeat bootstraps times for each id).
        :param obs_idx_vec: vector of ids of test ground truth to consider
        :param bootstraps: number of repetitions for each id
        :param metrics: dictionary of metrics to consider, with key the metric name and value the parameters to use
        (also as dictionary of relevant functions argument names)
        e.g. {'rmse': {}, 'vs': {'power': 0.5}, 'es_2': {'power': 2}, 'es_1': {'power': 1},
        'sinkhorn': {'epsilon': 100, 'n': 500, 'niter': 100, 'p': 2}}
        :param m: sample size to consider from training set
        :param make_summary_stats: bool, whether to compute summary statistics of the training references
        :param load_existing: bool, whether to load/reused existing training references if they exist on disk. If not, recompute them.
        :return: dict, keys are metrics
        """
        from fastabc_inversion.geo_problems.utils.evaluation.scorers import (
            torch_es, torch_rmse, torch_vs)

        _map_metrics_to_functions = {
            "rmse": torch_rmse,
            "vs": torch_vs,
            "es_1": torch_es,
            "es_2": torch_es,
            "sinkhorn": call_sinkhorn,
        }

        training_refs_summary_stats = {} if make_summary_stats else None

        if load_existing:
            try:
                training_refs = load_from_disk(
                    f"{self.inference_root_dir}/training_refs.pkl"
                )
                print("Loaded existing training references from disk.")
            except FileNotFoundError:
                print(
                    "No existing training references found on disk. Computing new ones."
                )
                training_refs = {}
        else:
            training_refs = {}

        for metric, metric_param in metrics.items():
            if metric not in _map_metrics_to_functions:
                raise ValueError(
                    f"Metric {metric} not recognized. Available metrics: {list(_map_metrics_to_functions.keys())}"
                )
            if metric_param is None:
                metric_param = {}

            metric_func = _map_metrics_to_functions[metric]

            if metric not in training_refs:
                training_refs[metric] = []

                # add on_gpu as keyword to metric_param
                metric_param["on_gpu"] = True

                for obs_idx in obs_idx_vec:
                    ref_x = self.test_x[obs_idx, :].reshape(
                        1, self.experiment_obj.dim_x
                    )
                    for _ in range(bootstraps):
                        indices = np.random.choice(
                            self.train_x.shape[0], m, replace=False
                        )
                        sample_x = self.train_x[indices, :]

                        score = metric_func(ref_x, sample_x, **metric_param)
                        training_refs[metric].append(score)

            if make_summary_stats:
                training_refs_summary_stats[metric] = compute_array_stats(
                    training_refs[metric]
                )

        save_to_disk(training_refs, f"{self.inference_root_dir}/training_refs.pkl")
        save_to_disk(
            training_refs_summary_stats,
            f"{self.inference_root_dir}/training_refs_summary_stats.txt",
            _pickle=False,
            _text=True,
        )

        return training_refs, training_refs_summary_stats

    def load_diags(self, list):
        """
        load previously saved diagnostics
        :param list: list containing one of 'rmse', 'scores' (for ES and VS)
        :return:
        """
        if "rmse" in list:
            self.all_obs_rmse_threshold = load_from_disk(
                f"{self.inference_root_dir}/all_obs_rmse_threshold.pkl"
            )
            self.all_obs_rmse_y_threshold = load_from_disk(
                f"{self.inference_root_dir}/all_obs_rmse_y_threshold.pkl"
            )
            self.all_obs_rmse_resims_threshold = load_from_disk(
                f"{self.inference_root_dir}/all_obs_rmse_resims_threshold.pkl"
            )

        if "scores" in list:
            # ES
            self.es_stats = load_from_disk(f"{self.inference_root_dir}/es_stats.pkl")
            self.es_stats_y = load_from_disk(
                f"{self.inference_root_dir}/es_stats_y.pkl"
            )
            self.es_stats_resims = load_from_disk(
                f"{self.inference_root_dir}/es_stats_resims.pkl"
            )
            # self.all_obs_min_es = load_from_disk(f"{self.inference_root_dir}/all_obs_min_ES_x.pkl")
            # self.all_obs_min_es_y = load_from_disk(f"{self.inference_root_dir}/all_obs_min_ES_y.pkl")
            # self.all_obs_min_es_resims = load_from_disk(f"{self.inference_root_dir}/all_obs_min_ES_resims.pkl")

            # VS
            self.vs_stats = load_from_disk(f"{self.inference_root_dir}/vs_stats.pkl")
            self.vs_stats_y = load_from_disk(
                f"{self.inference_root_dir}/vs_stats_y.pkl"
            )
            self.vs_stats_resims = load_from_disk(
                f"{self.inference_root_dir}/vs_stats_resims.pkl"
            )
            # self.all_obs_min_vs = load_from_disk(f"{self.inference_root_dir}/all_obs_min_VS_x.pkl")
            # self.all_obs_min_vs_y = load_from_disk(f"{self.inference_root_dir}/all_obs_min_VS_y.pkl")
            # self.all_obs_min_vs_resims = load_from_disk(f"{self.inference_root_dir}/all_obs_min_VS_resims.pkl")

    def get_threshold_exact_posterior_percentile(
        self, y_obs, threshold, noise_dict, distance_type="l1", m=1000, on_gpu=True
    ):
        """
        Finds the percentile corresponding to a given distance (with y_obs) threshold in the exact posterior
        distribution for a given observation.
        :param y_obs: the observation vector to compare against and to use to make the exact posterior samples
        :param threshold: the threshold value to find the corresponding percentile for
        :param noise_dict: dictionnary containing the noise distribution configuration
                e.g., {'distribution': 'Gaussian', 'location': 0, 'scale': 0.5}
        :param distance_type: the distance type 'l1' or 'l2' to use to compute between y_obs and the simulated y for the posterior samples
        :param m: number of samples from the posterior to use
        :param on_gpu: bool, whether to use GPU for distance computations
        :return: the percentile corresponding to the threshold value
        """
        # check if linear problem
        if self.experiment_obj.config.solver_type != "linear":
            print(
                "Only linear problems are implemented for exact posterior percentile computation."
            )
            return None

        # check in noise distribution is Gaussian
        if noise_dict["distribution"] != "Gaussian":
            print(
                "Only Gaussian noise distribution is implemented for exact posterior percentile computation."
            )
            return None

        noise_scale = noise_dict["scale"]

        jitter = 1e-6 if noise_scale == 0 else 0

        from fastabc_inversion.geo_problems.linear.analytical_inversion import (
            generate_Gauss_samples, setup_posterior)

        nx = self.experiment_obj.config.nx
        ny = self.experiment_obj.config.ny
        nc = self.experiment_obj.config.nc
        dim_x = nx * ny
        dim_y = self.experiment_obj.config.ndata
        forward = self.experiment_obj.config.solver_matrix
        prior_cov = self.experiment_obj.config.CM
        prior_mean = self.experiment_obj.config.m_prior

        # make y_obs an numpy array if not already
        if not isinstance(y_obs, np.ndarray):
            y_obs = np.array(y_obs)

        # make analytical posterior
        post_dict = setup_posterior(
            noise_scale,
            y_obs,
            forward,
            dim_y,
            prior_cov,
            prior_mean,
            dim_x,
            jitter=jitter,
        )

        # Generate samples from posterior distribution
        post_samples = generate_Gauss_samples(
            m,
            mean=post_dict["posterior_mean"],
            cov=post_dict["posterior_covariance"],
            dims=[nc, nx, ny],
        ).reshape(m, dim_x)

        # resimulate the posterior samples
        post_resimulations = np.dot(forward, post_samples.T).T

        # compute distances between y_obs and post_resimulations
        # make y_obs tensor and repeat m times
        y_obs_tensor = (
            torch.tensor(y_obs, dtype=torch.float32).reshape(1, dim_y).repeat(m, 1)
        )
        post_resimulations_tensor = torch.tensor(
            post_resimulations, dtype=torch.float32
        )  # shape : (m, dim_y)

        if on_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            y_obs_tensor = y_obs_tensor.to(device)
            post_resimulations_tensor = post_resimulations_tensor.to(device)

        if distance_type == "l1":
            distances = torch_dist.lpp_torch(
                y_obs_tensor, post_resimulations_tensor, p=1
            )
        elif distance_type == "l2":
            distances = torch_dist.lpp_torch(
                y_obs_tensor, post_resimulations_tensor, p=2
            )

        if on_gpu:
            distances = distances.cpu()

        distances = distances.numpy()

        # compute percentile corresponding to threshold
        percentile = find_percentile_of_value_in_array(distances, threshold)

        return percentile

    def assess_threshold_posterior_percentiles(
        self, probability_thresholds=None, sample_size=1000
    ):
        """
        Assess the percentile of a given threshold id in the exact posterior distribution for all inverted observations.
        :param probability_thresholds: the SuS failure probability thresholds to assess.
                    Expected to refer to the log_10 probability threshold.
                    e.g., -3 refers to probability threshold P_f = 1e-3 after the filtering of the samples.
                    In the list of all_thresholds it will refer to SuS threshold id = 2 (0-based indexing).
                    Note that 0 is not a valid input as it refers to the initial sample before any filtering.
        :param sample_size: number of samples to use from the exact posterior to estimate the percentiles
        :return: percentiles statistics
        """
        print("Assessing thresholds percentiles in the exact posterior distribution...")

        if probability_thresholds is not None:
            if not isinstance(probability_thresholds, list):
                probability_thresholds = [probability_thresholds]

        percentiles = {}
        summary_stats = {}

        epsilon = 0.01  # TODO: make this dynamic if multiple epsilons are used

        for noise_label, noise_dict in self.experiment_obj.noise_dicts.items():
            if probability_thresholds is None:
                # enumerate all thresholds up to max_it and add 1 to each element to avoid the 0 (initial sample)
                probability_thresholds = list(
                    np.arange(1, self.inference_params[noise_label]["max_it"] + 1)
                )

            percentiles[noise_label] = {}
            summary_stats[noise_label] = {}

            distance_type = self.inference_params[noise_label]["norm_fct"]

            for obs_idx in self.all_obs_results.keys():
                y_obs = self.inverted_obs_vectors[obs_idx][noise_label]["obs"]

                for threshold_id in probability_thresholds:
                    if threshold_id == 0:
                        threshold_id = 1  # avoid 0, assuming that this is what is meant

                    threshold_id = abs(threshold_id)  # make sure it's positive

                    if threshold_id not in percentiles[noise_label]:
                        percentiles[noise_label][threshold_id] = []

                    threshold = self.all_obs_results[obs_idx][noise_label][epsilon][
                        "all_thresholds"
                    ][0][threshold_id - 1]
                    percentiles[noise_label][threshold_id].append(
                        self.get_threshold_exact_posterior_percentile(
                            y_obs,
                            threshold,
                            noise_dict,
                            distance_type=distance_type,
                            m=sample_size,
                            on_gpu=True,
                        )
                    )
            for id in percentiles[noise_label].keys():
                summary_stats[noise_label][id] = compute_array_stats(
                    np.array(percentiles[noise_label][id])
                )

        save_to_disk(
            percentiles, f"{self.inference_root_dir}/threshold_percentiles.pkl"
        )
        save_to_disk(
            summary_stats,
            f"{self.inference_root_dir}/threshold_percentiles_summary.txt",
            _pickle=False,
            _text=True,
        )

        return percentiles, summary_stats

    def agg_obs_metrics(
        self,
        min_prob_thresh=None,
        make_plots=True,
        refs_dict_x=None,
        dpi=600,
        show=False,
    ):
        """
        Aggregate observation metrics (RMSE, ES, VS) for a given observation vector.
        :param obs_vec:
        :param min_prob_thresh: float, minimum probability threshold (in log10) to consider for aggregation.
               If None, defaults to -13, equivalent to 1e-13.
        :return:
        """
        if min_prob_thresh is None:
            min_prob_thresh = abs(-13)  # default to 1e-13
        else:
            min_prob_thresh = abs(min_prob_thresh)

        thresholds = np.arange(
            1, min_prob_thresh + 1, 1
        )  # skip the 0 (the initial sample)

        data = {}
        data_y = {}  # tilde{y} vs y_obs
        data_resims = {}  # y_resim vs y_obs

        for noise_label in self.noise_list:
            for epsilon in self.epsilon_values[noise_label]:
                data[(noise_label, epsilon)] = {}
                data[(noise_label, epsilon)]["rmse"] = {}
                data[(noise_label, epsilon)]["es"] = {}
                data[(noise_label, epsilon)]["vs"] = {}
                # TODO: make ES , VS dict dynamic for different powers
                data[(noise_label, epsilon)]["es"][1] = {}
                data[(noise_label, epsilon)]["es"][2] = {}
                data[(noise_label, epsilon)]["vs"][0.5] = {}

                data_y[(noise_label, epsilon)] = {}
                data_y[(noise_label, epsilon)]["rmse"] = {}
                data_y[(noise_label, epsilon)]["es"] = {}
                data_y[(noise_label, epsilon)]["vs"] = {}
                data_y[(noise_label, epsilon)]["es"][1] = {}
                data_y[(noise_label, epsilon)]["es"][2] = {}
                data_y[(noise_label, epsilon)]["vs"][0.5] = {}

                data_resims[(noise_label, epsilon)] = {}
                data_resims[(noise_label, epsilon)]["rmse"] = {}
                data_resims[(noise_label, epsilon)]["es"] = {}
                data_resims[(noise_label, epsilon)]["vs"] = {}
                data_resims[(noise_label, epsilon)]["es"][1] = {}
                data_resims[(noise_label, epsilon)]["es"][2] = {}
                data_resims[(noise_label, epsilon)]["vs"][0.5] = {}

                for threshold in thresholds:
                    data[(noise_label, epsilon)]["rmse"][threshold] = []
                    data[(noise_label, epsilon)]["es"][1][threshold] = []
                    data[(noise_label, epsilon)]["es"][2][threshold] = []
                    data[(noise_label, epsilon)]["vs"][0.5][threshold] = []

                    data_y[(noise_label, epsilon)]["rmse"][threshold] = []
                    data_y[(noise_label, epsilon)]["es"][1][threshold] = []
                    data_y[(noise_label, epsilon)]["es"][2][threshold] = []
                    data_y[(noise_label, epsilon)]["vs"][0.5][threshold] = []

                    data_resims[(noise_label, epsilon)]["rmse"][threshold] = []
                    data_resims[(noise_label, epsilon)]["es"][1][threshold] = []
                    data_resims[(noise_label, epsilon)]["es"][2][threshold] = []
                    data_resims[(noise_label, epsilon)]["vs"][0.5][threshold] = []

                    for obs_idx in self.inverted_obs:
                        data[(noise_label, epsilon)]["rmse"][threshold].extend(
                            self.all_obs_rmse_threshold[
                                (obs_idx, noise_label, epsilon)
                            ][threshold]
                        )
                        data_y[(noise_label, epsilon)]["rmse"][threshold].extend(
                            self.all_obs_rmse_y_threshold[
                                (obs_idx, noise_label, epsilon)
                            ][threshold]
                        )
                        data_resims[(noise_label, epsilon)]["rmse"][threshold].extend(
                            self.all_obs_rmse_resims_threshold[
                                (obs_idx, noise_label, epsilon)
                            ][threshold]
                        )

                        data[(noise_label, epsilon)]["es"][1][threshold].extend(
                            self.es_stats[(obs_idx, noise_label, epsilon)][1][threshold]
                        )  # takes all bootstraps
                        data_y[(noise_label, epsilon)]["es"][1][threshold].extend(
                            self.es_stats_y[(obs_idx, noise_label, epsilon)][1][
                                threshold
                            ]
                        )
                        data_resims[(noise_label, epsilon)]["es"][1][threshold].extend(
                            self.es_stats_resims[(obs_idx, noise_label, epsilon)][1][
                                threshold
                            ]
                        )

                        data[(noise_label, epsilon)]["es"][2][threshold].extend(
                            self.es_stats[(obs_idx, noise_label, epsilon)][2][threshold]
                        )
                        data_y[(noise_label, epsilon)]["es"][2][threshold].extend(
                            self.es_stats_y[(obs_idx, noise_label, epsilon)][2][
                                threshold
                            ]
                        )
                        data_resims[(noise_label, epsilon)]["es"][2][threshold].extend(
                            self.es_stats_resims[(obs_idx, noise_label, epsilon)][2][
                                threshold
                            ]
                        )

                        data[(noise_label, epsilon)]["vs"][0.5][threshold].extend(
                            self.vs_stats[(obs_idx, noise_label, epsilon)][0.5][
                                threshold
                            ]
                        )
                        data_y[(noise_label, epsilon)]["vs"][0.5][threshold].extend(
                            self.vs_stats_y[(obs_idx, noise_label, epsilon)][0.5][
                                threshold
                            ]
                        )
                        data_resims[(noise_label, epsilon)]["vs"][0.5][
                            threshold
                        ].extend(
                            self.vs_stats_resims[(obs_idx, noise_label, epsilon)][0.5][
                                threshold
                            ]
                        )

                # make boxplots of the aggregated data
                if make_plots:
                    # invert the order of thresholds to have the lowest threshold first
                    inverted_order = thresholds.argsort()[::-1]
                    labels = thresholds[inverted_order] * -1

                    if refs_dict_x is not None:
                        # use the references from the refs_dict_x if provided
                        rmse_refs = refs_dict_x["rmse"]
                        es_1_refs = refs_dict_x["es"][1]
                        es_2_refs = refs_dict_x["es"][2]
                        vs_05_refs = refs_dict_x["vs"][0.5]
                    else:
                        rmse_refs = es_1_refs = es_2_refs = vs_05_refs = None

                    # references from CCA files : reference_val_metrics_es2_vs05_rmse_summaries.json,
                    # reference_val_metrics_es1_summaries.json - not recomputed
                    # rmse_refs = {'train':{'lower':1.517, 'center':1.817, 'upper':2.186},}
                    # es_1_refs = {'train':{'lower':1792.621, 'center':2016.382, 'upper':2278.220},}
                    # es_2_refs = {'train':{'lower':2310.725, 'center':3301.291, 'upper':4535.569},}
                    # vs_05_refs = {'train':{'lower':635041.558, 'center':707509.737, 'upper':796022.504},}
                    # rmse
                    rmse_data = [
                        data[(noise_label, epsilon)]["rmse"][thresh]
                        for thresh in thresholds[inverted_order]
                    ]  # expect to be as list of lists (13, 30'000)
                    rmse_data = np.array(rmse_data).reshape(1, len(thresholds), -1)
                    axes_plot_titles = [
                        [rf"$RMSE(x^*, x_{{post}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$RMSE(x^*, x_{{post}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_RMSE_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        rmse_data,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=rmse_refs,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )
                    # rmse y
                    rmse_data_y = [
                        data_y[(noise_label, epsilon)]["rmse"][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    rmse_data_y = np.array(rmse_data_y).reshape(1, len(thresholds), -1)
                    axes_plot_titles = [
                        [rf"$RMSE(y_{{obs}}, y_{{post}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$RMSE(y_{{obs}}, y_{{post}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_RMSE_y_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        rmse_data_y,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=None,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # rmse resim
                    rmse_data_resims = [
                        data_resims[(noise_label, epsilon)]["rmse"][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    rmse_data_resims = np.array(rmse_data_resims).reshape(
                        1, len(thresholds), -1
                    )
                    axes_plot_titles = [
                        [rf"$RMSE(y_{{obs}}, y_{{resim}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$RMSE(y_{{obs}}, y_{{resim}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_RMSE_resims_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        rmse_data_resims,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=None,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # ES 1
                    es_1_data = [
                        data[(noise_label, epsilon)]["es"][1][thresh]
                        for thresh in thresholds[inverted_order]
                    ]  # expect to be as list of lists (13, 300)
                    es_1_data = np.array(es_1_data).reshape(1, len(thresholds), -1)
                    axes_plot_titles = [
                        [rf"$ES_1(x^*, D_{{post}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$ES_1(x^*, D_{{post}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_ES_1_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        es_1_data,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=es_1_refs,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # ES 1 y
                    es_1_data_y = [
                        data_y[(noise_label, epsilon)]["es"][1][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    es_1_data_y = np.array(es_1_data_y).reshape(1, len(thresholds), -1)
                    axes_plot_titles = [
                        [rf"$ES_1(y_{{obs}}, D_{{Y_{{post}}}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$ES_1(y_{{obs}}, D_{{Y_{{post}}}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_ES_1_y_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        es_1_data_y,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=None,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # ES 1 resim
                    es_1_data_resims = [
                        data_resims[(noise_label, epsilon)]["es"][1][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    es_1_data_resims = np.array(es_1_data_resims).reshape(
                        1, len(thresholds), -1
                    )
                    axes_plot_titles = [
                        [rf"$ES_1(y_{{obs}}, D_{{Y_{{resim}}}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$ES_1(y_{{obs}}, D_{{Y_{{resim}}}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_ES_1_resims_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        es_1_data_resims,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=None,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # ES 2
                    es_2_data = [
                        data[(noise_label, epsilon)]["es"][2][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    es_2_data = np.array(es_2_data).reshape(1, len(thresholds), -1)
                    axes_plot_titles = [
                        [rf"$ES_2(x^*, D_{{post}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$ES_2(x^*, D_{{post}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_ES_2_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        es_2_data,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=es_2_refs,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # ES 2 y
                    es_2_data_y = [
                        data_y[(noise_label, epsilon)]["es"][2][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    es_2_data_y = np.array(es_2_data_y).reshape(1, len(thresholds), -1)
                    axes_plot_titles = [
                        [rf"$ES_2(y_{{obs}}, D_{{Y_{{post}}}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$ES_2(y_{{obs}}, D_{{Y_{{post}}}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_ES_2_y_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        es_2_data_y,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=None,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # ES 2 resim
                    es_2_data_resims = [
                        data_resims[(noise_label, epsilon)]["es"][2][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    es_2_data_resims = np.array(es_2_data_resims).reshape(
                        1, len(thresholds), -1
                    )
                    axes_plot_titles = [
                        [rf"$ES_2(y_{{obs}}, D_{{Y_{{resim}}}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$ES_2(y_{{obs}}, D_{{Y_{{resim}}}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_ES_2_resims_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        es_2_data_resims,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=None,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # VS 0.5
                    vs_05_data = [
                        data[(noise_label, epsilon)]["vs"][0.5][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    vs_05_data = np.array(vs_05_data).reshape(1, len(thresholds), -1)
                    axes_plot_titles = [
                        [rf"$VS_{0.5}(x^*, D_{{post}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$VS_{0.5}(x^*, D_{{post}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_VS_05_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        vs_05_data,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=vs_05_refs,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # VS 0.5 y
                    vs_05_data_y = [
                        data_y[(noise_label, epsilon)]["vs"][0.5][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    vs_05_data_y = np.array(vs_05_data_y).reshape(
                        1, len(thresholds), -1
                    )
                    axes_plot_titles = [
                        [rf"$VS_{0.5}(y_{{obs}}, D_{{Y_{{post}}}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$VS_{0.5}(y_{{obs}}, D_{{Y_{{post}}}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_VS_05_y_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        vs_05_data_y,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=None,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

                    # VS 0.5 resim
                    vs_05_data_resims = [
                        data_resims[(noise_label, epsilon)]["vs"][0.5][thresh]
                        for thresh in thresholds[inverted_order]
                    ]
                    vs_05_data_resims = np.array(vs_05_data_resims).reshape(
                        1, len(thresholds), -1
                    )
                    axes_plot_titles = [
                        [rf"$VS_{0.5}(y_{{obs}}, D_{{Y_{{resim}}}})$"],
                        rf"$log_{{10}}(P_f)$",
                        rf"$VS_{0.5}(y_{{obs}}, D_{{Y_{{resim}}}})$",
                    ]
                    file_name = f"{self.inference_root_dir}/All_obs_boxplots_VS_05_resims_{noise_label}_{epsilon}.pdf"
                    plot.plot_boxplots(
                        vs_05_data_resims,
                        labels=labels,
                        axes_plot_titles=axes_plot_titles,
                        references_dict=None,
                        save_location=file_name,
                        dpi=dpi,
                        show=show,
                    )

    def compute_composite_sample_metrics(
        self, composite_sample, noise_list, es_p=[1, 2], vs_p=0.5, m=500
    ):
        """
        Compute the metrics for the composite sample. Computes RMSE, ES and VS for the composite sample against the ground
        truth or the observation vector (x, y, resimulated x).
        :param composite_sample: dict, composite sample containing 'x', 'y', 'resim_x' keys
        :param noise_list: list, list of noise labels to compute the metrics for
        :param es_p: list, powers for ES to compute
        :param vs_p: float or list, power for VS to compute
        :param m: int, number of samples to use for the ES and VS computation
        :return: dict, containing the computed metrics
        """
        from fastabc_inversion.geo_problems.utils.evaluation.scorers import (
            torch_es, torch_rmse, torch_vs)

        metrics = {}
        summary_stats = {}

        dim_x = self.experiment_obj.dim_x
        dim_y = self.experiment_obj.dim_y

        nx = self.experiment_obj.nx
        ny = self.experiment_obj.ny

        # compute SSIM
        metrics["ssim"] = {}  # only concerns x
        summary_stats["ssim"] = {}

        # compute RMSE
        metrics["rmse"] = {}
        metrics["rmse"]["x"] = {}
        metrics["rmse"]["y"] = {}
        metrics["rmse"]["resim_x"] = {}

        summary_stats["rmse"] = {}
        summary_stats["rmse"]["x"] = {}
        summary_stats["rmse"]["y"] = {}
        summary_stats["rmse"]["resim_x"] = {}

        for noise in noise_list:
            metrics["ssim"][noise] = []
            summary_stats["ssim"][noise] = {}

            metrics["rmse"]["x"][noise] = []
            summary_stats["rmse"]["x"][noise] = {}

            metrics["rmse"]["y"][noise] = []
            summary_stats["rmse"]["y"][noise] = {}

            metrics["rmse"]["resim_x"][noise] = []
            summary_stats["rmse"]["resim_x"][noise] = {}

        # compute ES
        if not isinstance(es_p, list):
            es_p = [es_p]
        metrics["es"] = {}
        summary_stats["es"] = {}

        for p in es_p:
            metrics["es"][p] = {}
            summary_stats["es"][p] = {}

            metrics["es"][p]["x"] = {}
            summary_stats["es"][p]["x"] = {}

            metrics["es"][p]["y"] = {}
            summary_stats["es"][p]["y"] = {}

            metrics["es"][p]["resim_x"] = {}
            summary_stats["es"][p]["resim_x"] = {}

            for noise in noise_list:
                metrics["es"][p]["x"][noise] = []
                summary_stats["es"][p]["x"][noise] = {}

                metrics["es"][p]["y"][noise] = []
                summary_stats["es"][p]["y"][noise] = {}

                metrics["es"][p]["resim_x"][noise] = []
                summary_stats["es"][p]["resim_x"][noise] = {}

        # compute VS
        if not isinstance(vs_p, list):
            vs_p = [vs_p]
        metrics["vs"] = {}
        summary_stats["vs"] = {}

        for p in vs_p:
            metrics["vs"][p] = {}
            summary_stats["vs"][p] = {}

            metrics["vs"][p]["x"] = {}
            summary_stats["vs"][p]["x"] = {}

            metrics["vs"][p]["y"] = {}
            summary_stats["vs"][p]["y"] = {}

            metrics["vs"][p]["resim_x"] = {}
            summary_stats["vs"][p]["resim_x"] = {}

            for noise in noise_list:
                metrics["vs"][p]["x"][noise] = []
                summary_stats["vs"][p]["x"][noise] = {}

                metrics["vs"][p]["y"][noise] = []
                summary_stats["vs"][p]["y"][noise] = {}

                metrics["vs"][p]["resim_x"][noise] = []
                summary_stats["vs"][p]["resim_x"][noise] = {}

        for obs_key in composite_sample.keys():
            N = composite_sample[obs_key]["x"].shape[0]
            obs_idx = obs_key[0]
            noise = obs_key[1]

            print(
                f"Computing composite sample metrics for observation {obs_idx} with noise {noise}"
            )

            ref_x = self.test_x[obs_idx, :].numpy().reshape(1, dim_x)  # ground truth

            y_obs = self.inverted_obs_vectors[obs_idx][noise]["obs"]
            # check if y_obs is numpy array :
            if not isinstance(y_obs, np.ndarray):
                y_obs = y_obs.numpy()
            y_obs = y_obs.reshape(1, dim_y)

            indices = np.random.choice(N, m, replace=False)
            sample_x = composite_sample[obs_key]["x"].reshape(N, dim_x)
            sample_y = composite_sample[obs_key]["y"].reshape(N, dim_y)
            sample_resim_x = composite_sample[obs_key]["resim_x"].reshape(N, dim_y)

            # compute SSIM
            metrics["ssim"][noise].append(
                np.array(self.compute_ssim(sample_x, ref_x, ny, nx))
            )

            # compute RMSE
            metrics["rmse"]["x"][noise].append(torch_rmse(ref_x, sample_x, on_gpu=True))
            metrics["rmse"]["y"][noise].append(torch_rmse(y_obs, sample_y, on_gpu=True))
            metrics["rmse"]["resim_x"][noise].append(
                torch_rmse(y_obs, sample_resim_x, on_gpu=True)
            )

            # compute ES
            for p in es_p:
                metrics["es"][p]["x"][noise].append(
                    torch_es(ref_x, sample_x[indices, :], power=p, on_gpu=True)
                )
                metrics["es"][p]["y"][noise].append(
                    torch_es(y_obs, sample_y[indices, :], power=p, on_gpu=True)
                )
                metrics["es"][p]["resim_x"][noise].append(
                    torch_es(y_obs, sample_resim_x[indices, :], power=p, on_gpu=True)
                )

            # compute VS
            for p in vs_p:
                metrics["vs"][p]["x"][noise].append(
                    torch_vs(ref_x, sample_x[indices, :], power=p, on_gpu=True)
                )
                metrics["vs"][p]["y"][noise].append(
                    torch_vs(y_obs, sample_y[indices, :], power=p, on_gpu=True)
                )
                metrics["vs"][p]["resim_x"][noise].append(
                    torch_vs(y_obs, sample_resim_x[indices, :], power=p, on_gpu=True)
                )

        save_to_disk(metrics, f"{self.inference_root_dir}/composite_sample_metrics.pkl")

        # make summary statistics for each metric
        for noise in noise_list:
            summary_stats["ssim"][noise] = compute_array_stats(metrics["ssim"][noise])

            summary_stats["rmse"]["x"][noise] = compute_array_stats(
                metrics["rmse"]["x"][noise]
            )
            summary_stats["rmse"]["y"][noise] = compute_array_stats(
                metrics["rmse"]["y"][noise]
            )
            summary_stats["rmse"]["resim_x"][noise] = compute_array_stats(
                metrics["rmse"]["resim_x"][noise]
            )

            for p in es_p:
                summary_stats["es"][p]["x"][noise] = compute_array_stats(
                    metrics["es"][p]["x"][noise]
                )
                summary_stats["es"][p]["y"][noise] = compute_array_stats(
                    metrics["es"][p]["y"][noise]
                )
                summary_stats["es"][p]["resim_x"][noise] = compute_array_stats(
                    metrics["es"][p]["resim_x"][noise]
                )

            for p in vs_p:
                summary_stats["vs"][p]["x"][noise] = compute_array_stats(
                    metrics["vs"][p]["x"][noise]
                )
                summary_stats["vs"][p]["y"][noise] = compute_array_stats(
                    metrics["vs"][p]["y"][noise]
                )
                summary_stats["vs"][p]["resim_x"][noise] = compute_array_stats(
                    metrics["vs"][p]["resim_x"][noise]
                )

        save_to_disk(
            summary_stats,
            f"{self.inference_root_dir}/composite_sample_metrics_summary_stats.txt",
            _pickle=False,
            _text=True,
        )

        self.composite_sample_metrics = metrics

        return metrics, summary_stats

    def plot_posterior_samples(
        self,
        composite_samples,
        composite_sample_metrics,
        obs_vec,
        k=3,
        pick_from_min_rmse_x=True,
        pick_from_min_rmse_y=False,
        min_rmse_number=10,
        dpi=600,
        show=False,
    ):
        """
        Randomly select k samples from the composite samples and plot them along the ground truth showing RMSE and SSIM.
        :param composite_samples: dict, composite sample containing 'x', 'y', 'resim_x' keys
        :param composite_sample_metrics: dict, metrics for the composite sample
        :param obs_vec: vector of observations ids to plot the samples for
        :param k: int, number of samples to plot
        :param pick_from_min_rmse_x: bool, whether to randomly pick the samples from the lowest (min_rmse_number*k) RMSE(x^*, x_post) range
                or randomly select k samples from the whole sample.
        :param min_rmse_number: int, number of minimum RMSE samples to consider for picking
        :param dpi: int, dpi for the plots
        :param show: bool, whether to show the plots or save them to disk
        """

        noise_list = self.noise_list
        dim_x = self.experiment_obj.dim_x

        nx = self.experiment_obj.nx
        ny = self.experiment_obj.ny

        min_rmse_to_pick_from = (
            min_rmse_number * k
        )  # number of minimum RMSE samples to consider for picking

        for noise in noise_list:
            epsilon = self.inference_params[noise]["epsilon_vec"][0]
            N = self.inference_params[noise]["N"]

            for obs_idx in obs_vec:
                print(
                    f"Plotting posterior samples for observation {obs_idx} with noise {noise} and epsilon {epsilon}"
                )

                obs_idx_in_vec = obs_vec.index(obs_idx)

                obs_key = (obs_idx, noise, epsilon)
                ref_x = self.test_x[obs_idx, :].numpy().reshape(1, dim_x)

                # folder where to save the plots
                file_name = (
                    f"{self.inverted_obs_vectors[obs_idx][noise]['obs_inference_dir']}/"
                    f"posterior_samples{'_random' if not pick_from_min_rmse_x else min_rmse_number}.pdf"
                )

                # randomly select k samples from the composite samples
                # pick the samples based on a combination of lowest rmse for x and y, if both flags are True
                if pick_from_min_rmse_x and pick_from_min_rmse_y:
                    # get the indices of the samples with minimum RMSE for x
                    rmse_values_x = composite_sample_metrics["rmse"]["x"][noise][
                        obs_idx_in_vec
                    ]
                    sorted_indices_x = np.argsort(rmse_values_x)[
                        :(min_rmse_to_pick_from)
                    ]
                    # get the indices of the samples with minimum RMSE for y
                    rmse_values_y = composite_sample_metrics["rmse"]["y"][noise][
                        obs_idx_in_vec
                    ]
                    sorted_indices_y = np.argsort(rmse_values_y)[
                        :(min_rmse_to_pick_from)
                    ]
                    # get the intersection of the two sets of indices
                    intersected_indices = np.intersect1d(
                        sorted_indices_x, sorted_indices_y
                    )
                    if len(intersected_indices) >= k:
                        indices = np.random.choice(
                            intersected_indices, k, replace=False
                        )
                    else:
                        print(
                            f"Warning: Not enough samples in the intersection of min RMSE x and y. Picking randomly from min RMSE x."
                        )
                        indices = np.random.choice(sorted_indices_x, k, replace=False)

                if pick_from_min_rmse_y and not pick_from_min_rmse_x:
                    # get the indices of the samples with minimum RMSE for y
                    rmse_values_y = composite_sample_metrics["rmse"]["y"][noise][
                        obs_idx_in_vec
                    ]
                    # select indices from the first 300 smallest RMSE values
                    sorted_indices = np.argsort(rmse_values_y)[:(min_rmse_to_pick_from)]
                    indices = np.random.choice(sorted_indices, k, replace=False)

                if pick_from_min_rmse_x and not pick_from_min_rmse_y:
                    # get the indices of the samples with minimum RMSE for y
                    rmse_values_x = composite_sample_metrics["rmse"]["x"][noise][
                        obs_idx_in_vec
                    ]
                    # select indices from the first 300 smallest RMSE values
                    sorted_indices = np.argsort(rmse_values_x)[:(min_rmse_to_pick_from)]
                    indices = np.random.choice(sorted_indices, k, replace=False)

                if not pick_from_min_rmse_x and not pick_from_min_rmse_y:
                    # pick randomly from the whole sample
                    indices = np.random.choice(N, k, replace=False)

                # concatenate ref_x and the selected samples
                selected_samples_x = composite_samples[obs_key]["x"][
                    indices, :
                ].reshape(k, dim_x)

                all_examples_to_plot = np.concatenate(
                    (ref_x, selected_samples_x), axis=0
                ).reshape(1, k + 1, -1)

                # get RMSE and SSIM for the selected samples
                # ATTENTION: ASSUMES THE ORDER IN OBS_VEC IS THE SAME AS IN THE COMPOSITE SAMPLES

                # get index of obs_idx in obs_vec
                rmse_values = composite_sample_metrics["rmse"]["x"][noise][
                    obs_idx_in_vec
                ][[indices]][0]
                # ssim_values = composite_sample_metrics['ssim'][noise][obs_idx_in_vec][[indices]][0]
                ssim_values = composite_sample_metrics["rmse"]["y"][noise][
                    obs_idx_in_vec
                ][[indices]][
                    0
                ]  # overrides SSIM with RMSE(y_{post}, y_obs) values

                # plot the samples
                plot.plot_samples(
                    all_examples_to_plot,
                    rmse_labels=[list(rmse_values)],
                    ssim_labels=[list(ssim_values)],
                    width=nx,
                    height=ny,
                    grd_truth=True,
                    save_location=file_name,
                    dpi=dpi,
                    show=show,
                    figsize=(15, 6),
                )

    def bench_sus_metrics(
        self,
        methods_metrics,
        n_obs=50,
        refs_dict=None,
        make_plots=True,
        dpi=600,
        show=False,
    ):
        """
        Benchmarks the composite SuS samples metrics (RMSE, ES, VS) against other methods metrics.
        :param methods_metrics: dictionary of metrics dictionaries, one dictionary per method.
               The first level of the keys should be the method name e.g., 'CCA' , 'cVAE' (as it should appear on the plots).
               Each dictionary should contain keys for 'RMSE', 'ES'(1 and or 2) and VS (0.5), with subkeys for 'x', 'y' and 'resim_x' and the noise labels.
               The code assumes that all metrics are provided for all noise labels.
               e.g., {'CCA':
                        'RMSE': {'small_gauss': [...],
                                {'large_gauss': [...]},
                        'ES': {
                            1: {'small_gauss': [...],
                               'large_gauss': [...]},
                            2: {'small_gauss': [...],
                               'large_gauss': [...]},
                            },
                        'VS': {0.5: {'small_gauss': [...],
                                    {'large_gauss': { [...]}
                    }
        :param make_plots: makes comparison boxplots of the metrics
        :param dpi: dpi for the plots
        :param show: whether to show the plots or save them to disk
        :return: summary statistics for each method's metrics
        """
        composite_sample_metrics = self.composite_sample_metrics
        m = 500
        total_rmse = int(m * n_obs)

        if composite_sample_metrics is None:
            raise ValueError(
                "No composite sample metrics found. Please compute them first using compute_composite_sample_metrics()."
            )

        methods = list(methods_metrics.keys())
        if len(methods) == 0:
            print("No methods metrics provided for benchmarking.")
            return
        # check if all methods have the same keys, if not make plots only for the common keys or the common powers (e.g., ES 1 and 2, VS 0.5)
        # TODO : get the the expected keys from self.composite_sample_metrics
        exp_metric_keys = ["rmse", "es", "vs"]  # expected keys for the metrics
        exp_es_powers = set([1, 2])
        exp_vs_powers = set([0.5])

        bench_metrics = exp_metric_keys
        for key in exp_metric_keys:
            if key not in methods_metrics[methods[0]].keys():
                print(
                    f"Key {key} not found in all methods metrics. Skipping benchmarking for this key."
                )
                # remove the key from the list of metrics to benchmark
                bench_metrics.remove(key)
        # get noise labels from the first method's metrics
        noise_labels = []
        for key in bench_metrics:
            noise_labels.extend(methods_metrics[methods[0]][key].keys())
            break

        # check if all methods have the same powers for ES and VS, if not make plots only for the common powers
        if "es" in bench_metrics:
            es_powers = exp_es_powers
        if "vs" in bench_metrics:
            vs_powers = exp_vs_powers
        for method in methods:
            if "es" in methods_metrics[method].keys():
                if methods_metrics[method]["es"].keys() != exp_es_powers:
                    # remove from es_powers what is not in the keys
                    es_powers = es_powers.intersection(
                        methods_metrics[method]["es"].keys()
                    )

            if "vs" in methods_metrics[method].keys():
                if methods_metrics[method]["vs"].keys() != exp_vs_powers:
                    # remove from vs_powers what is not in the keys
                    vs_powers = vs_powers.intersection(
                        methods_metrics[method]["vs"].keys()
                    )

        if len(es_powers) == 0:
            print(
                "No ES powers found in the methods metrics. Skipping benchmarking for ES."
            )
            # remove 'ES' from bench_metrics
            bench_metrics.remove("ES")

        if len(vs_powers) == 0:
            print(
                "No VS powers found in the methods metrics. Skipping benchmarking for VS."
            )
            # remove 'VS' from bench_metrics
            bench_metrics.remove("VS")

        if len(bench_metrics) == 0:
            print("No metrics to benchmark. Exiting.")
            return

        # prepare the sus_sample_metrics:
        sus_sample_metrics = {}
        for metric in bench_metrics:
            sus_sample_metrics[metric] = {}
            if metric == "es":
                for p in list(es_powers):
                    sus_sample_metrics[metric][p] = composite_sample_metrics[metric][p][
                        "x"
                    ]

            elif metric == "vs":
                for p in list(vs_powers):
                    sus_sample_metrics[metric][p] = composite_sample_metrics[metric][p][
                        "x"
                    ]

            elif metric == "rmse":
                sus_sample_metrics[metric] = {}
                all_rmses = composite_sample_metrics[metric]["x"]
                all_rmses_first_key = list(all_rmses.keys())[0]
                n_obs_sus = len(all_rmses[all_rmses_first_key])
                len_rmse_per_obs = all_rmses[all_rmses_first_key][0].shape[0]
                if int(n_obs_sus * len_rmse_per_obs) > total_rmse:
                    for noise in noise_labels:
                        sus_sample_metrics[metric][noise] = []
                        # pick m rmse values at random for each observation index
                        for idx in range(len(all_rmses[noise])):
                            # pick_idx = np.argsort(all_rmses[noise][idx])[:m] # pick smallest rmse values
                            pick_idx = np.random.choice(
                                len_rmse_per_obs, m, replace=False
                            )  # pick m random indices
                            sus_sample_metrics[metric][noise].extend(
                                all_rmses[noise][idx][pick_idx]
                            )
        del all_rmses

        # prepare boxplots data & hypothesis testing
        from scipy import stats

        methods.append("ours")
        paired_ttest = {}
        mean_ci = {}
        data_array_to_plot = {}
        for metric in bench_metrics:
            paired_ttest[metric] = {}
            mean_ci[metric] = {}
            if metric == "rmse":
                data_array_to_plot[metric] = {}
                for noise in noise_labels:
                    paired_ttest[metric][noise] = {}
                    data_array_to_plot[metric][noise] = {}
                    mean_ci[metric][noise] = {}
                    samples_for_ttest = {}

                    for method in methods:
                        data_array_to_plot[metric][noise][method] = []

                        if method == "ours":
                            data_array_to_plot[metric][noise][method] = np.array(
                                sus_sample_metrics[metric][noise]
                            )
                        else:
                            data_array_to_plot[metric][noise][method] = np.array(
                                methods_metrics[method][metric][noise]
                            )
                        samples_for_ttest[method] = data_array_to_plot[metric][noise][
                            method
                        ]

                        mean_ci[metric][noise][method] = (
                            np.mean(samples_for_ttest[method]),
                            stats.t.interval(
                                0.95,
                                len(samples_for_ttest[method]) - 1,
                                loc=np.mean(samples_for_ttest[method]),
                                scale=stats.sem(samples_for_ttest[method]),
                            ),
                        )
                    for method in methods:
                        if method != "ours":
                            paired_ttest[metric][noise][method] = stats.ttest_rel(
                                samples_for_ttest["ours"], samples_for_ttest[method]
                            )
                        else:
                            continue

                if make_plots:
                    plot.plot_bench_boxplots(
                        data_array_to_plot,
                        metric=metric,
                        references_dict=refs_dict[metric],
                        save_location=f"{self.inference_root_dir}/bench_{metric}_boxplots.pdf",
                    )

            if metric == "es":
                for p in es_powers:
                    paired_ttest[metric][p] = {}
                    mean_ci[metric][p] = {}
                    data_array_to_plot[metric] = {}
                    for noise in noise_labels:
                        paired_ttest[metric][p][noise] = {}
                        data_array_to_plot[metric][noise] = {}
                        mean_ci[metric][p][noise] = {}
                        samples_for_ttest = {}
                        for method in methods:
                            data_array_to_plot[metric][noise][method] = []

                            if method == "ours":
                                data_array_to_plot[metric][noise][method] = np.array(
                                    sus_sample_metrics[metric][p][noise]
                                )
                            else:
                                data_array_to_plot[metric][noise][method] = np.array(
                                    methods_metrics[method][metric][p][noise]
                                )
                            samples_for_ttest[method] = data_array_to_plot[metric][
                                noise
                            ][method]

                            mean_ci[metric][p][noise][method] = (
                                np.mean(samples_for_ttest[method]),
                                stats.t.interval(
                                    0.95,
                                    len(samples_for_ttest[method]) - 1,
                                    loc=np.mean(samples_for_ttest[method]),
                                    scale=stats.sem(samples_for_ttest[method]),
                                ),
                            )
                        for method in methods:
                            if method != "ours":
                                paired_ttest[metric][p][noise][
                                    method
                                ] = stats.ttest_rel(
                                    samples_for_ttest["ours"], samples_for_ttest[method]
                                )
                            else:
                                continue

                    if make_plots:
                        plot.plot_bench_boxplots(
                            data_array_to_plot,
                            metric=metric,
                            references_dict=refs_dict[metric][p],
                            save_location=f"{self.inference_root_dir}/bench_{metric}_{p}_boxplots.pdf",
                        )

            if metric == "vs":
                for p in vs_powers:
                    paired_ttest[metric][p] = {}
                    mean_ci[metric][p] = {}
                    data_array_to_plot[metric] = {}
                    for noise in noise_labels:
                        paired_ttest[metric][p][noise] = {}
                        data_array_to_plot[metric][noise] = {}
                        mean_ci[metric][p][noise] = {}
                        samples_for_ttest = {}
                        for method in methods:
                            data_array_to_plot[metric][noise][method] = []

                            if method == "ours":
                                data_array_to_plot[metric][noise][method] = np.array(
                                    sus_sample_metrics[metric][p][noise]
                                )
                            else:
                                data_array_to_plot[metric][noise][method] = np.array(
                                    methods_metrics[method][metric][p][noise]
                                )
                            samples_for_ttest[method] = data_array_to_plot[metric][
                                noise
                            ][method]

                            mean_ci[metric][p][noise][method] = (
                                np.mean(samples_for_ttest[method]),
                                stats.t.interval(
                                    0.95,
                                    len(samples_for_ttest[method]) - 1,
                                    loc=np.mean(samples_for_ttest[method]),
                                    scale=stats.sem(samples_for_ttest[method]),
                                ),
                            )

                        for method in methods:
                            if method != "ours":
                                paired_ttest[metric][p][noise][
                                    method
                                ] = stats.ttest_rel(
                                    samples_for_ttest["ours"], samples_for_ttest[method]
                                )
                            else:
                                continue

                    if make_plots:
                        plot.plot_bench_boxplots(
                            data_array_to_plot,
                            metric=metric,
                            references_dict=refs_dict[metric][p],
                            save_location=f"{self.inference_root_dir}/bench_{metric}_{p}_boxplots.pdf",
                            dpi=dpi,
                            show=show,
                        )

        save_to_disk(
            paired_ttest,
            f"{self.inference_root_dir}/bench_sus_metrics_paired_ttest.txt",
            _text=True,
            _pickle=False,
        )
        save_to_disk(
            mean_ci,
            f"{self.inference_root_dir}/bench_sus_metrics_mean_ci.txt",
            _text=True,
            _pickle=False,
        )
