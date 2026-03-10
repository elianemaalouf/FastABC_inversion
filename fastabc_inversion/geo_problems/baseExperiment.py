"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Base class for all joint Generative Neural Network with ABC by SubSet Simulation experiments
"""
import os
import pickle
import random
import string
import time
from abc import ABC, abstractmethod

import fastabc_inversion.geo_problems.utils.evaluation.Bernstein_curveSmoothing as bcs
import fastabc_inversion.geo_problems.utils.torch_data_prep as tdp
import numpy as np
import scipy as sp
import torch
import torch.distributions as dists
import torchinfo
from fastabc_inversion.geo_problems.utils import torch_distances as torch_dist
from fastabc_inversion.geo_problems.utils.subset_simulation.aCS import aCS
from fastabc_inversion.geo_problems.utils.subset_simulation.corr_factor import \
    corr_factor
from fastabc_inversion.geo_problems.utils.subset_simulation.ERADist import \
    ERADist
from fastabc_inversion.geo_problems.utils.visualization import \
    plotting_tools as plot
from scipy.interpolate import interp1d


class BaseExperiment(ABC):
    map_ERADist_to_torch_dist = {
        "standardnormal": dists.Normal,
        "normal": dists.Normal,
        "uniform": dists.Uniform,
    }

    def __init__(
        self,
        name,
        data_rootdir,
        latent_dist_name="standardnormal",
        latent_dist_params_list=[],
        model=None,
        run_id=None,
        seed=None,
        abc_add_noise=False,
        load_pretrained_model=False,
        pretrained_model_path_netD=None,
        pretrained_model_path_netG=None,
        sinkhorn_params=None,
        model_training_params=None,
        inference_params=None,
        noise_dicts=None,
    ):
        """
        Experiment base class
        :param name: name of the experiment
        :param data_rootdir: directory where the data is stored and results will be saved
        :param latent_dist_name: name of the latent distribution as expected by ERADist
        :param latent_dist_params_list: parameters list of the latent distribution. Order of parameters
        should be the same as the order of parameters expected by ERADist
        :param model: model to be used for the experiment
        :param run_id: run id of the experiment. Used to create a folder for the experiment in the data_rootdir.
        :param seed: seed for reproducibility
        :param abc_add_noise: whether to add noise to the generated vectors during ABC inference
        :param load_pretrained_model: whether to load a pretrained model
        :param pretrained_model_path_netD: path to the pretrained netD model
        :param pretrained_model_path_netG: path to the pretrained netG model
        :param sinkhorn_params: parameters for the sinkhorn algorithm
        :param model_training_params: parameters for training the model
        :param inference_params: parameters for inference
        :param noise_dict: dictionary containing the noise configuration
        """
        self.g_fun = None
        self.u2x = None
        self.latent_dim = None
        self.input_dim = None
        self.output_dim = None
        self.inversion_dir = None
        self.abc_add_noise = abc_add_noise
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
        self.latent_dist_name = latent_dist_name
        self.latent_dist_params_list = latent_dist_params_list
        self.model_training_stats = {
            "overall_loss_vs_epochs": {},
            "reconstruction_loss_vs_epochs": {},
            "latent_Distdistance_loss_vs_epochs": {},
        }
        self.nn_params = None
        self.norms_params = None
        self.sinkhorn_params = sinkhorn_params
        self.sinkhorn_lambda_scheduling_params = None
        self.model_training_params = model_training_params
        self.best_training_epoch = None

        self.inference_params = inference_params
        self.noise_dicts = noise_dicts
        self.obs_inference_dir_prefix = None
        self.all_epsilon_results = None
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

        self.name = (
            f"{name}_{self.run_id}"
            # if self.noise_dict is None
            # f"{name}_Noise_{self.noise_dict['distribution']}_scale_{self.noise_dict['scale']}_{self.run_id}"
        )

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

        # map latent distribution name to torch distribution
        self.latent_dist = self.map_ERADist_to_torch_dist[self.latent_dist_name]

        # prepare for SuS obs inference directories creation
        self.obs_inference_dir_prefix = "test_SuS_obs_"

    @abstractmethod
    def load_data(self):
        """
        Load the data
        """
        pass

    @abstractmethod
    def get_observation(self):
        """
        Get observation from the test set
        """
        pass

    @abstractmethod
    def prep_model(self):
        """
        Load the model
        """
        pass

    @abstractmethod
    def train_model(self):
        """
        Train the model
        """
        pass

    @abstractmethod
    def save_checkpoint(self):
        """
        Save model checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self):
        """
        Load model checkpoint
        """
        pass

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
            f.write(f"Experiment latent distribution name: {self.latent_dist_name}\n")
            f.write(
                f"Experiment latent distribution parameters: {self.latent_dist_params_list}\n"
            )
            f.write(f"Experiment latent space dimension: {self.latent_dim}\n")
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
            f.write(f"Experiment sinkhorn parameters: {self.sinkhorn_params}\n")
            f.write(
                f"Experiment sinkhorn lambda scheduling parameters: {self.sinkhorn_lambda_scheduling_params}\n"
            )
            f.write(
                f"Experiment model training parameters: {self.model_training_params}\n"
            )
            f.write(f"Experiment metric spaces norms parameters: {self.norms_params}\n")
            f.write(f"Experiment dims: x, y: {self.dim_x}, {self.dim_y}\n")
            f.write(f"Experiment training best epoch: {self.best_training_epoch}\n")

    def save_model_summary(self, input_dim):
        """
        Save the model summary to a file in the model training directory
        """
        result = torchinfo.summary(self.model, input_size=input_dim, device=self.device)

        with open(f"{self.model_training_dir}/model_summary.txt", "a+") as f:
            f.write(str(result))

    def update_inference_params(self, inference_params, noise_label):
        """
        Update the inference parameters.
        :param inference_params: dictionary of inference parameters.
        :param noise_label: the noise label to update the inference parameters for.
        """
        self.inference_params[noise_label] = inference_params.copy()

    def prep_SuS(self):
        """
        Prepare for SuS

        :param latent_dim: dimension of the latent space
        :param latent_distribution_name: name of the latent distribution as expected by ERADist
        """

        # we assume all latent dims have the same distribution and are independent
        pi_pdf = list()

        for i in range(self.latent_dim):
            pi_pdf.append(
                ERADist(self.latent_dist_name, "PAR", self.latent_dist_params_list)
            )  # independent rv

        self.u2x = lambda u: pi_pdf[0].icdf(
            sp.stats.norm.cdf(u)
        )  # from standard to latent distribution

        def g_fun(
            u_vect,
            latent_dim,
            generator_net,
            device,
            observation,
            noise_dict,
            dim_x,
            dim_y,
            u2x,
            norm_fct=None,
            abc_add_noise=False,
            unormalization_dict=None,
        ):
            """
            Limit State Function
            :param u_vect: random vector in the standard space
            :param latent_dim: generative model latent space dimension
            :param generator_net: generative model
            :param device: device where the generative model is stored
            :param observation: observation vector
            :param noise_dict: measurement noise configuration
            :param dim_x: dimension of variable X
            :param dim_y: dimension of variable Y
            :param u2x: function to transform from standard space to latent space
            :param norm_fct: function to compute the difference between the generated vectors and the observation
            :param abc_add_noise: whether to add noise to the generated vector during ABC inference
            :param unormalization_dict: dictionary containing the unnormalization parameters for the generated vectors
            :return: computed differences between the generated vectors and the observation
            """
            # TODO: check that no code calling g_fun is using gen_features parameter + verify all signature changes

            # back to original latent dist
            u_vect = u2x(
                u_vect.reshape(latent_dim, -1)
            )  # TODO: check if reshaping is needed

            u_vect = (
                torch.FloatTensor(u_vect).view(-1, latent_dim).to(device)
            )  # to device

            gen_i = (generator_net(u_vect)[:, dim_x:]).detach().cpu()

            if unormalization_dict is not None:
                gen_i = tdp.un_normalize(gen_i, unormalization_dict)
            else:
                pass

            if abc_add_noise and noise_dict is not None:
                noise_distribution = noise_dict["distribution"]
                noise_location = noise_dict["location"]
                noise_scale = noise_dict["scale"]

                if noise_distribution == "Gaussian".lower():
                    gen_i = gen_i + torch.FloatTensor(
                        noise_scale * np.random.randn(dim_y)
                    ).view(1, -1)
                elif noise_distribution == "Gumbel".lower():
                    gen_i = gen_i + torch.FloatTensor(
                        np.random.gumbel(noise_location, noise_scale, dim_y)
                    ).view(1, -1)
                else:
                    raise ValueError("Noise distribution not supported")

            elif abc_add_noise and noise_dict is None:
                raise ValueError(
                    "Noise dictionary is None and adding noise during ABC is True"
                )

            else:
                pass

            if norm_fct is None:
                # Sum of squared differences
                gen_i_norm_diff = torch_dist.lpp_torch(gen_i, observation, p=2)
            else:
                if norm_fct == "l2":
                    gen_i_norm_diff = torch_dist.lpp_torch(gen_i, observation, p=2)
                if norm_fct == "l1":
                    gen_i_norm_diff = torch_dist.lpp_torch(gen_i, observation, p=1)

            return gen_i_norm_diff.numpy()

        self.g_fun = g_fun
        return None

    def SuS_run(
        self,
        N,
        p0,
        epsilon,
        observation_vec,
        noise_dict,
        norm_fct=None,
        max_it=50,
        sus_run_id=0,
        return_full_results=False,
    ):
        """
        Run SuS algorithm

        :param N: number of samples
        :param p0: probability of failure
        :param epsilon: target threshold
        :param observation_vec: observation vector
        :param noise_dict: dictionary containing the noise configuration
        :param max_it: maximum number of iterations for SuS
        :param sus_run_id: run id of the SuS algorithm, if running SuS multiple times
        :param return_full_results: whether to return the full results of the SuS run including the samples
                at each intermediate level and Pf evolution data
        :return: dictionary containing the results of the SuS run

        Note :
        This function is based on the SuS algorithm from
        https://www.cee.ed.tum.de/era/software/reliability/subset-simulation/
        Code version : Version 2021-03
        It was slightly modified to return the inverted latent samples and to numpy 1.26.0 library update.
        The original code is in a separate file in SubsetSimulation/SuS_original.py
        """

        #  Initialization of variables and storage
        j = 0  # initial conditional level
        Nc = int(N * p0)  # number of markov chains
        Ns = int(1 / p0)  # number of samples simulated from each Markov chain
        lam = 0.6  # recommended initial value for lambda

        max_it = max_it

        samplesU = {
            "seeds": list(),
            "total": list(),
            "original": list(),
        }  # store samples

        #
        geval = np.zeros(N)  # space for the LSF evaluations
        gsort = np.zeros([max_it, N])  # space for the sorted LSF evaluations
        delta = np.zeros(max_it)  # space for the coefficient of variation
        nF = np.zeros(max_it)  # space for the number of failure point per level
        prob = np.zeros(max_it)  # space for the failure probability at each level
        b = np.zeros(max_it)  # space for the intermediate levels

        #  SuS procedure
        # initial MCS stage
        print(f"epsilon:{epsilon}")
        print(f"Run #{sus_run_id} - Evaluating performance function:\t", end="")
        u_j = sp.stats.norm.rvs(
            size=(self.latent_dim, N)
        )  # samples in the standard space

        g_fun_params = {
            "latent_dim": self.latent_dim,
            "generator_net": self.netG,
            "device": self.device,
            "observation": observation_vec,
            "noise_dict": noise_dict,
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "u2x": self.u2x,
            "norm_fct": norm_fct,
            "abc_add_noise": self.abc_add_noise,
            "unormalization_dict": self.normalization_dict_y,
        }

        start_time = time.time()

        for i in range(N):
            geval[i] = self.g_fun(
                u_j[:, i],
                **g_fun_params,
            )
            # print(geval[i])

        print("OK!")

        # adaptive Conditional sampling loop
        while True:
            # sort values in ascending total
            idx = np.argsort(geval)
            gsort[j, :] = geval[idx]

            # total the samples according to idx
            u_j_sort = u_j[:, idx]
            samplesU["total"].append(u_j_sort)  # store the ordered samples

            if return_full_results:
                inverted_latent = np.zeros([N, self.latent_dim])

                for iii in range(N):
                    u_j_vect = self.u2x(u_j_sort[:, iii].reshape(self.latent_dim, -1))
                    inverted_latent[iii, :] = torch.FloatTensor(u_j_vect).view(
                        -1, self.latent_dim
                    )  # on cpu
                samplesU["original"].append(inverted_latent)

            # intermediate level
            b[j] = np.percentile(geval, p0 * 100)

            # number of failure points in the next level
            nF[j] = sum(geval <= max(b[j], epsilon))

            # assign conditional probability to the level
            if b[j] <= epsilon:
                b[j] = epsilon
                prob[j] = nF[j] / N
            else:
                prob[j] = p0
            print(f"\nRun #{sus_run_id} -Threshold intermediate level {j} = {b[j]}")

            # compute coefficient of variation
            if j == 0:
                delta[j] = np.sqrt(
                    ((1 - p0) / (N * p0))
                )  # cov for p(1): MCS (Ref. 2 Eq. 8)
            else:
                I_Fj = np.reshape(
                    geval <= b[j], (Ns, Nc)
                )  # indicator function for the failure samples
                p_j = (1 / N) * np.sum(I_Fj[:])  # ~=p0, sample conditional probability
                gamma = corr_factor(I_Fj, p_j, Ns, Nc)  # corr factor (Ref. 2 Eq. 10)
                delta[j] = np.sqrt(
                    ((1 - p_j) / (N * p_j)) * (1 + gamma)
                )  # coeff of variation(Ref. 2 Eq. 9)

            # select seeds
            samplesU["seeds"].append(
                u_j_sort[:, : int(nF[j])]
            )  # store ordered level seeds

            # randomize the totaling of the samples (to avoid bias)
            idx_rnd = np.random.permutation(int(nF[j]))
            rnd_seeds = samplesU["seeds"][j][:, idx_rnd]  # non-ordered seeds

            # sampling process using adaptive conditional sampling
            [u_j, geval, lam, sigma, accrate] = aCS(
                N,
                lam,
                b[j],
                rnd_seeds,
                self.g_fun,
                **g_fun_params,
            )
            print(
                "\t*aCS lambda =",
                lam,
                "\t*aCS sigma =",
                sigma[0],
                "\t*aCS accrate =",
                accrate,
            )

            # next level
            j = j + 1

            if b[j - 1] <= epsilon or j == max_it:
                break

        m = j
        samplesU["total"].append(u_j)  # store final failure samples (non-totaled)

        end_time = time.time()

        run_time = end_time - start_time

        # delete unnecesary data
        if m < max_it:
            gsort = gsort[:m, :]
            prob = prob[:m]
            b = b[:m]
            delta = delta[:m]

        # probability of failure
        # failure probability estimate
        Pf_SuS = np.prod(prob)  # or p0^(m-1)*(Nf(m)/N)
        print(f"Run #{sus_run_id} - Failure probability estimate = {Pf_SuS}")

        # coefficient of variation estimate
        delta_SuS = np.sqrt(np.sum(delta**2))  # (Ref. 2 Eq. 12)
        print(f"Coeff. variation = {delta_SuS}")

        # Pf evolution
        Pf = np.zeros(m)
        Pf_line = np.zeros((m, Nc))
        b_line = np.zeros((m, Nc))
        Pf[0] = p0
        Pf_line[0, :] = np.linspace(p0, 1, Nc)
        b_line[0, :] = np.percentile(gsort[0, :], Pf_line[0, :] * 100)
        for i in range(1, m):
            Pf[i] = Pf[i - 1] * p0
            Pf_line[i, :] = Pf_line[i - 1, :] * p0
            b_line[i, :] = np.percentile(gsort[i, :], Pf_line[0, :] * 100)

        # Pf_line = np.sort(Pf_line.reshape(-1)) # original SuS code
        # b_line = np.sort(b_line.reshape(-1)) # original SuS code
        Pf_line = Pf_line.reshape(-1)
        b_line = b_line.reshape(-1)

        #  transform the final sample to the original latent space
        u_j_final_run = u_j

        inverted_latent = np.zeros([N, self.latent_dim])

        for i in range(N):
            u_vect = self.u2x(u_j_final_run[:, i].reshape(self.latent_dim, -1))
            inverted_latent[i, :] = torch.FloatTensor(u_vect).view(
                -1, self.latent_dim
            )  # on cpu

        if return_full_results:
            samplesU["original"].append(inverted_latent)

        results_dict = {
            "p_f": Pf_SuS,  # final P_f
            "delta": delta_SuS,  # final delta
            "final_epsilon": b[-1],
            "all_prob": prob,
            "all_thresholds": b,
            "all_delta": delta,
            "original_epsilon": epsilon,
            "SuS_run_time": run_time,
            "final_inverted_latent": torch.FloatTensor(inverted_latent),
            "Pf_line": Pf_line,
            "b_line": b_line,
            "samples_per_thresh": samplesU["original"] if return_full_results else None,
        }

        return results_dict

    def run_sus_inference(
        self,
        sus_runs=20,
        observation_idx=1,
        noise_label=None,
        return_full_results=False,
    ):
        """
        Run inference using SuS
        :param sus_runs: number of times to repeat the SuS algorithm
        :param observation_idx: identifier of the observation to read from file
        :param inference_params: parameters for inference
        :return:
        """

        all_epsilon_results = {}

        N = self.inference_params[noise_label]["N"]
        p0 = self.inference_params[noise_label]["p0"]
        epsilon_vec = self.inference_params[noise_label]["epsilon_vec"]
        norm_fct = self.inference_params[noise_label]["norm_fct"]
        max_it = self.inference_params[noise_label]["max_it"]
        noise_dict = self.noise_dicts[noise_label]

        self.model.eval()  # set to evaluation mode
        self.prep_SuS()  # prepare SuS

        # add sus_runs inference_params
        self.inference_params[noise_label]["sus_runs"] = sus_runs

        # load observation
        observation = self.get_observation(observation_idx, noise_label)

        for epsilon in epsilon_vec:
            sus_runs_results_dict = {
                "p_f": [],
                "delta": [],
                "all_prob": [],
                "all_thresholds": [],
                "all_delta": [],
                "final_epsilon": [],
                "original_epsilon": [],
                "SuS_run_time": [],
                "Pf_line": [],
                "b_line": [],
                "final_inverted_latent": torch.FloatTensor(),
                "samples_per_thresh": [] if return_full_results else None,
            }

            # run SuS
            for run in range(sus_runs):
                sus_run_result = self.SuS_run(
                    N=N,
                    p0=p0,
                    epsilon=epsilon,
                    observation_vec=observation,
                    noise_dict=noise_dict,
                    norm_fct=norm_fct,
                    max_it=max_it,
                    sus_run_id=run,
                    return_full_results=return_full_results,
                )

                # store results
                sus_runs_results_dict["p_f"].append(sus_run_result["p_f"])
                sus_runs_results_dict["delta"].append(sus_run_result["delta"])
                sus_runs_results_dict["all_prob"].append(sus_run_result["all_prob"])
                sus_runs_results_dict["all_thresholds"].append(
                    sus_run_result["all_thresholds"]
                )
                sus_runs_results_dict["all_delta"].append(sus_run_result["all_delta"])

                sus_runs_results_dict["final_epsilon"].append(
                    sus_run_result["final_epsilon"]
                )
                sus_runs_results_dict["original_epsilon"].append(
                    sus_run_result["original_epsilon"]
                )
                sus_runs_results_dict["SuS_run_time"].append(
                    sus_run_result["SuS_run_time"]
                )
                # concatenate latent samples in a single torch Floattensor (for all runs)
                sus_runs_results_dict["final_inverted_latent"] = torch.cat(
                    (
                        sus_runs_results_dict["final_inverted_latent"],
                        sus_run_result["final_inverted_latent"],
                    ),
                    dim=0,
                )
                sus_runs_results_dict["Pf_line"].append(sus_run_result["Pf_line"])
                sus_runs_results_dict["b_line"].append(sus_run_result["b_line"])
                if return_full_results:
                    sus_runs_results_dict["samples_per_thresh"].append(
                        sus_run_result["samples_per_thresh"]
                    )

            all_epsilon_results[epsilon] = sus_runs_results_dict

        # self.all_epsilon_results = all_epsilon_results

        # dump inference results with latent samples
        obs_inference_dir = f"{self.inference_dir}/{noise_label}/{self.obs_inference_dir_prefix}{observation_idx}"
        os.makedirs(obs_inference_dir, exist_ok=True)
        # if not os.path.exists(obs_inference_dir):
        #    try:
        #        os.mkdir(obs_inference_dir)
        #    except OSError:
        #        print(f"Creation of the directory {obs_inference_dir} failed")

        inference_results_filename = f"{obs_inference_dir}/inference_results_dict.pkl"
        with open(inference_results_filename, "wb") as f:
            pickle.dump(all_epsilon_results, f)

        return all_epsilon_results

    def run_sus_inference_all_observations(
        self,
        observation_vec,
        sus_runs=20,
        noise_label=None,
        return_full_results=False,
    ):
        """
        Run inference using SuS for all observations in observation_vec
        :param observation_vec: list of observations indices to read from file and run inference on
        :param sus_runs: number of times to repeat the SuS algorithm
        :return: dictionary containing the results of the SuS runs for all observations
        """
        print("Running SuS inference ...\n")
        self.inverted_obs_idx = observation_vec
        if self.all_obs_inference_results is None:
            self.all_obs_inference_results = {}

        for observation_idx in self.inverted_obs_idx:
            print(f"Running inference for observation {observation_idx} ...")
            if observation_idx not in self.all_obs_inference_results:
                self.all_obs_inference_results[observation_idx] = {}
            self.all_obs_inference_results[observation_idx][
                noise_label
            ] = self.run_sus_inference(
                observation_idx=observation_idx,
                sus_runs=sus_runs,
                noise_label=noise_label,
                return_full_results=return_full_results,
            )
        if len(self.inverted_obs_idx) > 1:
            self.all_epsilon_results = None  # reset all_epsilon_results if running inference for multiple observations

        with open(f"{self.experiment_dir}/experiment_configurations.txt", "a+") as f:
            # add date and time
            f.write(f"\n Experiment date and time: {time.ctime()}\n")
            f.write(
                f"Experiment ABC noise dictionary: {self.noise_dicts[noise_label]}\n"
            )
            f.write(
                f"Experiment inference parameters: {self.inference_params[noise_label]}\n"
            )
            f.write(f"Experiment abc add noise: {self.abc_add_noise}\n")
            f.write(
                f"Experiment inverted observation indices: {self.inverted_obs_idx}\n"
            )

        return self.all_obs_inference_results

    def load_inference_results(self, observation_idx_vec, noise_label, all_obs_results):
        """
        Load inference results from file
        :param observation_idx_vec: list of observation indices to read from file
        :param noise_label: the noise label to load the inference results for
        :param all_obs_results: dictionary containing the inference results for all observations to be updated with the
        inference results for the observations and noise type
        :return: dictionary containing the inference results for all observations
        """
        for observation_idx in observation_idx_vec:
            if observation_idx not in all_obs_results.keys():
                all_obs_results[observation_idx] = {}
            obs_inference_dir = f"{self.inference_dir}/{noise_label}/{self.obs_inference_dir_prefix}{observation_idx}"
            inference_results_filename = (
                f"{obs_inference_dir}/inference_results_dict.pkl"
            )
            with open(inference_results_filename, "rb") as f:
                all_obs_results[observation_idx][noise_label] = pickle.load(f)
        self.all_obs_inference_results = all_obs_results

        return self.all_obs_inference_results

    def get_inverted_x_y(self, inverted_latent):
        """
        Get inverted x from the inverted latent z, for one epsilon. All samples from all SuS runs are combined.
        :param inverted_latent: inverted latent z samples in Torch tensor format
        """

        inverted_latent_gpu = inverted_latent.to(self.device)
        inverted_xy = self.netG(inverted_latent_gpu).detach().cpu()  # through generator

        inverted_x = inverted_xy[:, 0 : self.dim_x].squeeze(0)  # generated x
        inverted_y = inverted_xy[:, self.dim_x :].squeeze(0)  # generated y

        return inverted_x, inverted_y

    def get_all_epsilon_inverted_x_y(self):
        """
        Get inverted x (and their jointly generated y) from the inverted latent z, for all epsilons
        :return all_epsilon_inverted_x_dict: dictionary containing inverted x for all epsilons
                all_epsilon_inverted_y_dict: dictionary containing inverted y for all epsilons
        """
        all_epsilon_inverted_x_dict = {}
        all_epsilon_inverted_y_dict = {}

        for epsilon in self.inference_params["epsilon_vec"]:
            inverted_latent = self.all_epsilon_results[epsilon]["inverted_latent"]
            inverted_x, inverted_y = self.get_inverted_x_y(inverted_latent)
            all_epsilon_inverted_x_dict[epsilon] = inverted_x
            all_epsilon_inverted_y_dict[epsilon] = inverted_y

        return all_epsilon_inverted_x_dict, all_epsilon_inverted_y_dict

    # Evaluation functions
    def plot_training_loss(
        self, labels, loss_type="overall_loss_vs_epochs", save=True, show=True
    ):
        """
        Plot evolution of training loss vs epochs
        :param labels: dictionary containing the labels for the plot with the following keys:
        "title", "x_label", "y_label"
        :param loss_type: selector of loss to plot from model_training_stats
        :param save: whether to save the plot or not
        :param show: whether to show the plot or not
        """

        # setup plotting tools
        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config(mpl)

        fig = plt.figure()
        ax = fig.subplots(1, 1)

        ax.plot(self.model_training_stats[loss_type])
        ax.set_xlabel(labels["x_label"])
        ax.set_ylabel(labels["y_label"])
        ax.set_title(labels["title"])

        # put xticks only at integers
        if self.model_training_params["nb_epochs"] > 10:
            if self.model_training_params["nb_epochs"] % 2 == 0:
                multiple = 2
            elif self.model_training_params["nb_epochs"] % 3 == 0:
                multiple = 3
            elif self.model_training_params["nb_epochs"] % 5 == 0:
                multiple = 5
        else:
            multiple = 1

        ax.xaxis.set_major_locator(tick.MultipleLocator(multiple))

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()

        if save:
            plt.savefig(
                f"{self.model_training_dir}/{loss_type}.pdf",
                bbox_inches="tight",
                dpi=300,
            )

        if show:
            plt.show()
        plt.close()

    def estimate_p_f_curvature(
        self,
        p_f_stats,
        estimate_time=False,
        pf_to_ignore=0.2,
        step=0.25,
        pol_d=700,
        acc=1e-20,
        log_scale=True,
        observation_idx=None,
    ):
        """
        Estimate the curvature of the P(Z \in \Gamma_Z) vs ABC-epsilon curve
        :param P_f_stats: dictionary containing the statistics of P(Z \in \Gamma_Z) vs ABC-epsilon
        :param estimate_time: whether to estimate the time needed to compute the curvature or not
        :param pf_to_ignore: probability of failure values above which not to plot
        :param step: step size for the x-axis
        :param pol_d: degree of the polynomial to use for the approximation
        :param acc: minimum value of the probability to consider as different from 0
        :param log_scale: whether to use log scale for the y-axis or not
        :param observation_idx: index of the observation for which inference is being performed
        :return:
        """
        final_epsilon_vec = p_f_stats["final_epsilon_vec"]
        mean_pf_vec = p_f_stats["mean_pf_vec"]

        if estimate_time:
            start_time = time.time()

        min_x = np.min(final_epsilon_vec)
        max_x = np.max(final_epsilon_vec)
        final_eps_t = bcs.transform_x(final_epsilon_vec, min_x, max_x)
        idx_to_hide = np.argwhere(np.array(mean_pf_vec) > pf_to_ignore)

        if log_scale:
            f_1 = interp1d(final_eps_t, np.log(mean_pf_vec))
        else:
            f_1 = interp1d(final_eps_t, mean_pf_vec)

        min_eps_plot = np.min(np.delete(final_epsilon_vec, idx_to_hide))
        max_eps_plot = np.max(np.delete(final_epsilon_vec, idx_to_hide))

        step = step
        pol_d = pol_d
        x_new = np.linspace(
            min_eps_plot,
            max_eps_plot,
            ((max_eps_plot - min_eps_plot) // step).astype(int),
        )
        t = bcs.transform_x(x_new, min_x, max_x)

        bpoly = bcs.bpoly

        apx_y = []
        curv_y = []
        d1_y = []
        d2_y = []
        for t_i in t:
            apx_y.append(bcs.approx_fn(f_1, pol_d, bpoly, t_i, min_x, max_x))
            d1_y.append(bcs.deriv1_approx_fn(f_1, pol_d, bpoly, t_i, min_x, max_x))
            d2_y.append(bcs.deriv2_approx_fn(f_1, pol_d, bpoly, t_i, min_x, max_x))
            curv_y.append(
                bcs.curvature(
                    t_i,
                    bcs.deriv1_approx_fn,
                    bcs.deriv2_approx_fn,
                    f_1,
                    pol_d,
                    bpoly,
                    min_x,
                    max_x,
                )
            )
        apx_y = np.array(apx_y)
        curv_y = np.array(curv_y)
        d1_y = np.array(d1_y)
        d2_y = np.array(d2_y)

        id_sorted = np.argsort(-curv_y)  # - for descending order
        acc = np.log(acc) if log_scale else acc

        for id_max in id_sorted:
            if apx_y[id_max] < acc:
                continue
            else:
                break
        id_max_eps = np.argmin(
            np.abs(final_epsilon_vec - x_new[id_max])
        )  # id of original epsilon closest to maximum curvature id issued from approximation

        if estimate_time:
            end_time = time.time()
            print("Curvature estimation time : ", end_time - start_time)

        results = {
            "apx_y": apx_y,
            "curv_y": curv_y,
            "id_max_eps": id_max_eps,
            "idx_to_hide": idx_to_hide,
            "x_new": x_new,
            "id_max": id_max,
            "min_eps_plot": min_eps_plot,
            "max_eps_plot": max_eps_plot,
        }

        # pickle results
        with open(
            f"{self.inference_dir}/"
            f"{self.obs_inference_dir_prefix}{observation_idx}"
            f"/curvatureEst_results_dict.pkl",
            "wb",
        ) as f:
            pickle.dump(results, f)

        return results

    def plot_p_f_curvature_vs_epsilon(
        self,
        curvature_data,
        p_f_stats,
        observation_idx,
        log_scale=True,
        save=True,
        show=True,
    ):
        """
        Plot the curvature of the P(Z \in \Gamma_Z) vs ABC-epsilon curve
        :param curvature_data: dictionary containing the curvature data as generated by estimate_p_f_curvature()
        :param p_f_stats: dictionary containing the statistics of P(Z \in \Gamma_Z) vs ABC-epsilon
        as generated by p_f_stats()
        :param observation_idx: index of the observation for which inference is being performed
        :param log_scale: whether to use log scale for the y-axis or not
        :param save: whether to save the plot or not
        :param show: whether to show the plot or not
        """

        # setup plotting tools
        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config()

        mean_pf_vec = p_f_stats["mean_pf_vec"]
        final_epsilon_vec = p_f_stats["final_epsilon_vec"]

        idx_to_hide = curvature_data["idx_to_hide"]
        x_new = curvature_data["x_new"]
        apx_y = curvature_data["apx_y"]
        curv_y = curvature_data["curv_y"]
        id_max = curvature_data["id_max"]
        id_max_eps = curvature_data["id_max_eps"]
        min_eps_plot = curvature_data["min_eps_plot"]
        max_eps_plot = curvature_data["max_eps_plot"]

        mean_pf_vec_to_plot = np.log(mean_pf_vec) if log_scale else mean_pf_vec

        min_y = np.min(np.log(mean_pf_vec))

        fig, ax1 = plt.subplots(1, 1)

        ax1.plot(
            np.delete(final_epsilon_vec, idx_to_hide),
            np.delete(mean_pf_vec_to_plot, idx_to_hide),
            "--",
            linewidth="3",
            label="true",
        )
        ax1.plot(x_new, apx_y, linewidth="3", label="berns")
        ax1.scatter(x_new[id_max], apx_y[id_max], s=60, c="green")
        ax1.scatter(
            final_epsilon_vec[id_max_eps],
            np.log(mean_pf_vec)[id_max_eps],
            s=60,
            c="gray",
        )
        ax1.vlines(
            final_epsilon_vec[id_max_eps],
            ymin=min_y,
            ymax=np.log(mean_pf_vec)[id_max_eps],
            linestyles="--",
            colors="gray",
        )
        ax1.set_ylim(bottom=min_y, top=0.05)
        ax1.set_xlim(left=min_eps_plot - 5, right=max_eps_plot)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_ylabel(r"$log(\widehat{p}_{\epsilon})$")

        xticks_main = np.arange(min_eps_plot, max_eps_plot, 10)
        xticks_minor = xticks_main

        xlabels_main = np.rint(xticks_main).astype(
            int
        )  # np.around(np.delete(final_epsilon_vec, idx_to_hide),2)
        xlabels_minor = np.around(np.sqrt(xlabels_main / self.dim_y), 2)

        xlabels_main = [str(elem) for elem in xlabels_main]
        xlabels_minor = [str(elem) for elem in xlabels_minor]
        xlabels = [a + "\n" + b for a, b in zip(xlabels_main, xlabels_minor)]

        ax1.set_xticks(xticks_main)
        ax1.set_xticks(xticks_minor, minor=True)
        ax1.set_xticklabels(xlabels)
        ax1.set_xlabel(r"Reached $\epsilon$ (ns$^2$) \n Reached $\epsilon_n$ (ns)")

        ax2 = ax1.twinx()
        ax2.plot(x_new, curv_y, ":", c="green", linewidth="3")
        ax2.set_ylim(bottom=0, top=np.max(curv_y) + 0.01001)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_ylabel("Curvature")

        plt.tight_layout()
        if save:
            plt.savefig(
                f"{self.inference_dir}/"
                f"{self.obs_inference_dir_prefix}{observation_idx}"
                f"/logYpf_curvature_vs_finalEpsilon.pdf",
                dpi=300,
            )
        if show:
            plt.show()
        plt.close()

    def p_f_stats(self, epsilon_vec, all_epsilon_results):
        """
        Compute mean (and its t-dist 95% CI) of probability that the latent vector is in the Gamma_z set over the SuS runs.
        Will be used to draw intervals in the plot of P(Z \in \Gamma_Z) vs ABC-epsilon.
        :param epsilon_vec: vector of ABC-epsilon values tested
        :param all_epsilon_results: dictionary containing the results of the SuS runs for all epsilons
        :return: dictionary containing the statistics
        """
        from scipy.stats import t

        n = len(all_epsilon_results[epsilon_vec[0]]["p_f"])
        t_value = t.ppf(0.975, n - 1) if n > 1 else 0  # t-dist at 95%

        lower_pf_vec = []
        upper_pf_vec = []
        mean_pf_vec = []
        final_epsilon_vec = []

        for epsilon in epsilon_vec:
            pf_vec = np.array(all_epsilon_results[epsilon]["p_f"])
            final_epsilon_vec.append(
                np.mean(np.array(all_epsilon_results[epsilon]["final_epsilon"]))
            )
            pf_sd = np.std(pf_vec)

            pf_mean = np.mean(pf_vec)
            mean_pf_vec.append(pf_mean)

            lower_pf_vec.append(pf_mean - t_value * pf_sd / np.sqrt(n))
            upper_pf_vec.append(pf_mean + t_value * pf_sd / np.sqrt(n))

        return {
            "lower_pf_vec": lower_pf_vec,
            "upper_pf_vec": upper_pf_vec,
            "mean_pf_vec": mean_pf_vec,
            "final_epsilon_vec": final_epsilon_vec,
        }

    def plot_pf_vs_epsilon(
        self,
        p_f_stats,
        noise_level,
        observation_idx,
        labels,
        ticks_to_delete,
        inference_params,
        x_scale="log",
        y_scale="log",
        min_x=None,
        max_x=None,
        min_y=None,
        max_y=None,
        save=True,
        show=True,
    ):
        """
        Plot the P(Z \in \Gamma_Z) vs ABC-epsilon curve
        :param p_f_stats: dictionary containing the statistics of P(Z \in \Gamma_Z) vs ABC-epsilon
        as generated by p_f_stats()
        :param noise_level: noise level added to observation
        :param observation_idx: index of the observation for which inference is being performed
        :param labels: dictionary containing the labels for the plot with the following keys:
        "title", "x_label", "y_label"
        :param ticks_to_delete: list of ticks to delete from the x-axis for visibility
        :param x_scale: scale of the x-axis, either "linear" or "log"
        :param min_x: minimum value of the x-axis
        :param save: whether to save the plot or not
        :param show: whether to show the plot or not
        """
        # setup plotting tools
        mpl, plt, make_axes_locatable, tick = plot.plots_imports()
        plot.base_config()

        fig, ax = plt.subplots()

        final_epsilon_vec = p_f_stats["final_epsilon_vec"]
        mean_pf_vec = p_f_stats[
            "mean_pf_vec"
        ]  # if y_scale == 'linear' else np.log(p_f_stats["mean_pf_vec"])
        lower_pf_vec = p_f_stats[
            "lower_pf_vec"
        ]  # if y_scale == 'linear' else np.log(p_f_stats["lower_pf_vec"])
        upper_pf_vec = p_f_stats[
            "upper_pf_vec"
        ]  # if y_scale == 'linear' else np.log(p_f_stats["upper_pf_vec"])

        epsilon_vec = inference_params["epsilon_vec"]

        noise_level_text = np.round(
            np.sqrt(noise_level / self.dim_y), 2
        )  # to add to the plot

        ax.plot(final_epsilon_vec, mean_pf_vec, c="blue", linewidth=2)
        ax.fill_between(
            final_epsilon_vec, lower_pf_vec, upper_pf_vec, color="blue", alpha=0.2
        )

        # epsilon_vec_id_dot = range(len(epsilon_vec))

        # for ii in epsilon_vec_id_dot:
        #    plt.plot(
        #        (final_epsilon_vec[ii], final_epsilon_vec[ii]),
        #        (lower_pf_vec[ii], upper_pf_vec[ii]),
        #        color="blue",
        #        linewidth=2,
        #    )
        #    plt.scatter(final_epsilon_vec[ii], mean_pf_vec[ii], s=15, color="blue")
        #    plt.scatter(
        #        final_epsilon_vec[ii], lower_pf_vec[ii], marker="_", color="blue"
        #    )
        #    plt.scatter(
        #        final_epsilon_vec[ii], upper_pf_vec[ii], marker="_", color="blue"
        #    )

        #    epsilon_vec_reduced.append(final_epsilon_vec[ii])
        epsilon_vec_reduced = final_epsilon_vec
        epsilon_vec_reduced = np.array(epsilon_vec_reduced)

        if min_x is None:
            min_x = np.min(np.delete(final_epsilon_vec, ticks_to_delete))
            if x_scale == "log":
                min_x = min(min_x, noise_level)

        if max_x is None:
            max_x = np.max(np.delete(final_epsilon_vec, ticks_to_delete))

        if min_y is None:
            min_y = -0.05 if y_scale == "linear" else np.min(mean_pf_vec)

        if max_y is None:
            max_y = 1.01 if y_scale == "linear" else 0.05

        # plt.plot(final_epsilon_vec, mean_pf_vec, linewidth=3, color="blue")

        # plt.scatter(noise_level, -0.01, marker="2", color="red", s=100)

        # noise_x_loc = noise_level if x_scale == 'linear' else np.log(noise_level)
        noise_text_y = -0.04 if y_scale == "linear" else min_y - 0.04

        ax.text(
            noise_level, noise_text_y, f"noise = {noise_level_text:.2} ns", color="red"
        )

        plt.xticks([])

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        xticks_main = np.arange(min_x, max_x, 10)
        xticks_minor = xticks_main

        xlabels_main = np.rint(xticks_main).astype(
            int
        )  # np.around(np.delete(final_epsilon_vec, idx_to_hide),2)
        xlabels_minor = np.around(np.sqrt(xlabels_main / self.dim_y), 2)

        xlabels_main = [str(elem) for elem in xlabels_main]
        xlabels_minor = [str(elem) for elem in xlabels_minor]
        xlabels = [a + "\n" + b for a, b in zip(xlabels_main, xlabels_minor)]

        ax.set_xticks(xticks_main)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xlabels)

        # plt.xticks(
        #    np.insert(np.delete(epsilon_vec_reduced, ticks_to_delete), 0, min_x),
        #    np.insert(
        #        np.around(
        #            np.sqrt(
        #                np.delete(epsilon_vec_reduced, ticks_to_delete) / self.dim_y
        #            ),
        #            2,
        #        ),
        #        0,
        #        np.around(np.sqrt(min_x / self.dim_y), 2),
        #    ),
        # )

        ax.set_xlabel(labels["x_label"])
        ax.set_ylabel(labels["y_label"])

        ax.set_ylim(bottom=min_y, top=max_y)
        ax.set_xlim(left=min_x - 5, right=max_x)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.title(labels["title"])
        plt.tight_layout()

        if save:
            plt.savefig(
                f"{self.inference_dir}/"
                f"{self.obs_inference_dir_prefix}{observation_idx}"
                f"/pf_vs_epsilon_{self.latent_dim}z_{observation_idx}.pdf",
                dpi=300,
            )
        if show:
            plt.show()

        plt.close()

        print(final_epsilon_vec)

        print(np.around(np.sqrt(epsilon_vec_reduced / self.dim_y), 2))
