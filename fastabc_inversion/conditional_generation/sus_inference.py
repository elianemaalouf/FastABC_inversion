import numpy as np
import scipy as sp
import torch
import time
import os
import pickle
from fastabc_inversion.utils.SubsetSimulation.ERADist import ERADist
from fastabc_inversion.utils.SubsetSimulation.aCS import aCS
from fastabc_inversion.utils.SubsetSimulation.corr_factor import corr_factor
import fastabc_inversion.utils.torch_distances as torch_dist
from fastabc_inversion.conditional_generation.mnist.morphology.measure import measure_slant, measure_thickness, measure_length

def prep_SuS(experiment):
    """
    Prepare for SuS.
    Define the limit state function g_fun and the u2x transformation function.
    :param experiment: experiment object containing the generative model and latent distribution info
    """
    experiment.model.eval()

    # we assume all latent dims have the same distribution and are independent
    pi_pdf = list()

    for i in range(experiment.latent_dim):
        pi_pdf.append(
            ERADist(experiment.latent_dist_name, "PAR", experiment.latent_dist_params_list)
        )  # independent rv

    experiment.u2x = lambda u: pi_pdf[0].icdf(
        sp.stats.norm.cdf(u)
    )  # from standard to latent distribution

    from scipy.stats import wasserstein_distance

    def g_fun(
            u_vect,
            latent_dim,
            generator_net,
            device,
            observation,
            u2x,
            norm_fct=None,
    ):
        """
        Limit State Function
        :param u_vect: random vector in the standard space
        :param latent_dim: generative model latent space dimension
        :param generator_net: generative model
        :param device: device where the generative model is stored
        :param observation: observation vector
        :param dim_x: dimension of variable X
        :param dim_y: dimension of variable Y
        :param u2x: function to transform from standard space to latent space
        :param norm_fct: function to compute the difference between the generated vectors and the observation
        :return: computed differences between the generated vectors and the observation
        """

        implemented_distances = ['l1', 'l2', 'cross_entropy', 'kl_divergence', 'slant', 'thickness', 'length', 'emd']
        if norm_fct is not None:
            if norm_fct not in implemented_distances:
                raise ValueError(f"Distance {norm_fct} not implemented. Choose one of {implemented_distances}.")

        # back to original latent dist
        u_vect = u2x(
            u_vect.reshape(latent_dim, -1)
        )  # TODO: check if reshaping is needed

        u_vect = (
            torch.FloatTensor(u_vect).view(-1, latent_dim).to(device)
        )  # to device

        if norm_fct is None:
            norm_fct = 'l2'  # default

        with torch.no_grad():
            gen_x, gen_y = generator_net(u_vect)
            gen_x = gen_x.cpu().squeeze().numpy()
            gen_x = gen_x * 0.5 + 0.5 # de-normalize to [0,1] ! important for morphology measures

            if norm_fct == 'slant':
                # compute slant for gen_x
                gen_i = measure_slant(gen_x)
                #gen_i = np.rad2deg(gen_i)
                # make observation a tensor with two dimensions
                observation = np.array(observation).reshape(1, -1)
            elif norm_fct == 'thickness':
                gen_i = measure_thickness(gen_x)
                observation = np.array(observation).reshape(1, -1)
            elif norm_fct == 'length':
                gen_i = measure_length(gen_x)
                observation = np.array(observation).reshape(1, -1)
            else:
                gen_i = gen_y  # use the generated labels for the g_fun
                gen_i = gen_i.cpu()

            if norm_fct in ['l1', 'l2']:
                # transform to clr space
                observation = experiment.torch_clr_transformer(observation.unsqueeze(0))
                gen_i = experiment.torch_clr_transformer(gen_i)

        if norm_fct == 'l2':
            gen_i_norm_diff = torch_dist.lpp_torch(gen_i, observation, p=2).numpy()
        if norm_fct == 'l1' or norm_fct == 'slant' or norm_fct == 'thickness' or norm_fct == 'length':
            gen_i_norm_diff = torch_dist.lpp_torch(gen_i, observation, p=1).numpy()
        if norm_fct == 'cross_entropy':
            gen_i_norm_diff = torch_dist.cross_entropy_torch(gen_i, observation).numpy()
        if norm_fct == 'kl_divergence':
            gen_i_norm_diff = torch_dist.D_KL_simplex(gen_i, observation).numpy()
        if norm_fct== 'emd':
            domain = np.arange(gen_i.shape[1]) # assuming gen_i is of shape (1, num_classes)
            gen_i_norm_diff = wasserstein_distance(u_values=domain, v_values=domain,
                                                   u_weights=gen_i.numpy().flatten(),
                                                   v_weights=observation.flatten())

        #print(f"Generated value: {gen_i}, Observation: {observation}, Difference: {gen_i_norm_diff}")

        return gen_i_norm_diff

    experiment.g_fun = g_fun
    return None

def SuS_run(
        experiment,
        N,
        p0,
        epsilon,
        observation_vec,
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
    The original code is Originals/SuS_aCS_python_2021.zip
    """

    #  Initialization of variables and storage
    j = 0  # initial conditional level
    Nc = int(N * p0)  # number of markov chains
    Ns = int(1 / p0)  # number of samples simulated from each Markov chain
    lam = 0.6  # recommended initial value for lambda

    max_it = max_it

    samplesU = {"seeds": list(), "total": list(), "original": list()}  # store samples

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
        size=(experiment.latent_dim, N)
    )  # samples in the standard space

    g_fun_params = {
        "latent_dim": experiment.latent_dim,
        "generator_net": experiment.netG,
        "device": experiment.device,
        "observation": observation_vec,
        "u2x": experiment.u2x,
        "norm_fct": norm_fct,
    }

    start_time = time.time()

    for i in range(N):
        geval[i] = experiment.g_fun(
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
            inverted_latent = np.zeros([N, experiment.latent_dim])

            for iii in range(N):
                u_j_vect = experiment.u2x(u_j_sort[:, iii].reshape(experiment.latent_dim, -1))
                inverted_latent[iii, :] = torch.FloatTensor(u_j_vect).view(
                    -1, experiment.latent_dim
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
            experiment.g_fun,
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
    delta_SuS = np.sqrt(np.sum(delta ** 2))  # (Ref. 2 Eq. 12)
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

    inverted_latent = np.zeros([N, experiment.latent_dim])

    for i in range(N):
        u_vect = experiment.u2x(u_j_final_run[:, i].reshape(experiment.latent_dim, -1))
        inverted_latent[i, :] = torch.FloatTensor(u_vect).view(
            -1, experiment.latent_dim
        )  # on cpu

    if return_full_results:
        samplesU["original"].append(inverted_latent)

    results_dict = {
        "p_f": Pf_SuS,  # final P_f
        "delta": delta_SuS,  # final delta
        "final_epsilon": b[- 1],
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

def run_sus_inference(experiment, label_obs, inference_params):
    """
    Run inference using SuS
    :param experiment: experiment object containing the generative model and latent distribution info
    :param sus_runs: number of times to repeat the SuS algorithm
    :param label_obs: observed label to condition on
    :param inference_params: parameters for inference
    :return:
    """

    all_epsilon_results = {}

    N = inference_params["N"]
    p0 = inference_params["p0"]
    epsilon_vec = inference_params["epsilon_vec"]
    norm_fct = inference_params["norm_fct"]
    max_it = inference_params["max_it"]
    return_full_results = inference_params["return_full_results"]
    sus_runs = inference_params["sus_runs"]

    prep_SuS(experiment)  # prepare SuS
    experiment.inference_params = inference_params

    # check if label_obs is an int value or a one-hot vector
    # if one value transform into one-hot vector
    # if already a vector, leave as is
    # does not apply to slant
    if norm_fct not in ['slant', 'thickness', 'length'] :
        if isinstance(label_obs, int):
            # one-hot encode
            observation = experiment.label_transform(label_obs) # leave clr transformation to prep_SuS and g_fun
        else:
            # make sure it is a torch tensor
            observation = torch.tensor(label_obs, dtype=torch.float)
    else:
        observation = label_obs  # for slant and thickness, observation is a scalar value

    for epsilon in epsilon_vec:
        sus_runs_results_dict = {
            "p_f": [],
            "delta": [],
            "all_prob":[],
            "all_thresholds":[],
            "all_delta":[],
            "final_epsilon": [],
            "original_epsilon": [],
            "SuS_run_time": [],
            "Pf_line":[],
            "b_line":[],
            "final_inverted_latent": torch.FloatTensor(),
            "samples_per_thresh": [] if return_full_results else None,
        }

        # run SuS
        for run in range(sus_runs):
            sus_run_result = SuS_run(
                experiment = experiment,
                N=N,
                p0=p0,
                epsilon=epsilon,
                observation_vec=observation,
                norm_fct=norm_fct,
                max_it=max_it,
                sus_run_id=run,
                return_full_results=return_full_results,
            )

            # store results
            sus_runs_results_dict["p_f"].append(sus_run_result["p_f"])
            sus_runs_results_dict["delta"].append(sus_run_result["delta"])
            sus_runs_results_dict["all_prob"].append(sus_run_result["all_prob"])
            sus_runs_results_dict["all_thresholds"].append(sus_run_result["all_thresholds"])
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
                sus_runs_results_dict["samples_per_thresh"].append(sus_run_result["samples_per_thresh"])

        all_epsilon_results[epsilon] = sus_runs_results_dict

    # dump inference results with latent samples
    obs_inference_dir = (
        f"{experiment.inference_dir}/label_{label_obs}"
    )
    os.makedirs(obs_inference_dir, exist_ok=True)

    inference_results_filename = f"{obs_inference_dir}/inference_results_dict.pkl"
    with open(inference_results_filename, "wb") as f:
        pickle.dump(all_epsilon_results, f)

    return all_epsilon_results

def run_sus_inference_all_observations(
        experiment, observation_vec, inference_params,
):
    """
    Run inference using SuS for all observations in observation_vec
    :param experiment: experiment object containing the generative model and latent distribution info
    :param observation_vec: list of observations to run inference for. In conditional generation, these are the labels to condition on.
    :param inference_params: parameters for inference
    :return: dictionary containing the results of the SuS runs for all observations
    """
    print("Running SuS inference ...\n")
    experiment.inverted_obs_idx = observation_vec

    experiment.all_obs_inference_results = {}

    for observation_idx in experiment.inverted_obs_idx:
        print(f"Running inference for condition {observation_idx} ...")
        # Convert list to tuple for dictionary key
        dict_key = tuple(observation_idx) if isinstance(observation_idx, list) else observation_idx

        if dict_key not in experiment.all_obs_inference_results:
            experiment.all_obs_inference_results[dict_key] = {}

        experiment.all_obs_inference_results[dict_key] = run_sus_inference(
            experiment=experiment,
            label_obs=observation_idx,  # Pass original format to function
            inference_params=inference_params
        )

    with open(f"{experiment.inference_dir}/inference_config.txt", "a+") as f:
        # add date and time
        f.write(f"\n Experiment date and time: {time.ctime()}\n")
        f.write(f"Experiment inference parameters: {inference_params}\n")
        f.write(f"Experiment inverted observation indices: {experiment.inverted_obs_idx}\n")

    return experiment.all_obs_inference_results

def get_inverted_x_y(experiment, inverted_latent, return_labels=True):
    """
    Get inverted x from the inverted latent z, for one epsilon. All samples from all SuS runs are combined.
    :param experiment: experiment object containing the generative model
    :param inverted_latent: inverted latent z samples in Torch tensor format
    :param return_labels: whether to return labels (True) or one-hot encoded y (False)
    """

    inverted_latent_gpu = inverted_latent.to(experiment.device)
    with torch.no_grad():
        inverted_x, inverted_y = experiment.netG(inverted_latent_gpu)  # through generator

    inverted_x = inverted_x.cpu()
    inverted_y = inverted_y.cpu()

    if return_labels:
        inverted_y = experiment.label_transform.simplex_vec_to_label(inverted_y)

    return inverted_x, inverted_y

def get_all_epsilon_inverted_x_y(experiment):
    """
    Get inverted x (and their jointly generated y) from the inverted latent z, for all epsilons
    :param experiment: experiment object containing the generative model and all epsilon results
    :return all_epsilon_inverted_x_dict: dictionary containing inverted x for all epsilons
            all_epsilon_inverted_y_dict: dictionary containing inverted y for all epsilons
    """
    all_epsilon_inverted_x_dict = {}
    all_epsilon_inverted_y_dict = {}

    for epsilon in experiment.inference_params["epsilon_vec"]:
        inverted_latent = experiment.all_epsilon_results[epsilon]["inverted_latent"]
        inverted_x, inverted_y = experiment.get_inverted_x_y(inverted_latent)
        all_epsilon_inverted_x_dict[epsilon] = inverted_x
        all_epsilon_inverted_y_dict[epsilon] = inverted_y

    return all_epsilon_inverted_x_dict, all_epsilon_inverted_y_dict

def read_sus_inference_results_from_files(experiment, observation_vec):
    """
    Read SuS inference results from files for all observations in observation_vec
    :param experiment: experiment object containing the generative model and latent distribution info
    :param observation_vec: list of observations to run inference for. In conditional generation, these are the labels to condition on.
    :return: dictionary containing the results of the SuS runs for all observations
    """
    print("Reading SuS inference results from files ...\n")
    experiment.inverted_obs_idx = observation_vec

    experiment.all_obs_inference_results = {}

    for observation_idx in experiment.inverted_obs_idx:
        print(f"Reading inference results for condition {observation_idx} ...")
        # Convert list to tuple for dictionary key
        dict_key = tuple(observation_idx) if isinstance(observation_idx, list) else observation_idx

        if dict_key not in experiment.all_obs_inference_results:
            experiment.all_obs_inference_results[dict_key] = {}

        obs_inference_dir = (
            f"{experiment.inference_dir}/label_{observation_idx}"
        )
        inference_results_filename = f"{obs_inference_dir}/inference_results_dict.pkl"
        with open(inference_results_filename, "rb") as f:
            experiment.all_obs_inference_results[dict_key] = pickle.load(f)

    return experiment.all_obs_inference_results


