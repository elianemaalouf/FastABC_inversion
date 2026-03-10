# correct CCA format and compute ES2
from fastabc_inversion.geo_problems.Inference_Diagnostics import load_from_disk, save_to_disk

cca_metrics_file = "FastABC/data/geo_nonlinear/other_methods_inference_metrics/CCA/inversion_metrics.pkl"
cca_data_file = "FastABC/data/geo_nonlinear/other_methods_inference_metrics/CCA/inversion_data.pkl"

cca_metrics = load_from_disk(cca_metrics_file)
cca_data = load_from_disk(cca_data_file)

observation_vec = [102, 106, 270, 435, 860, 154, 253, 309, 548, 966, 385, 498, 583, 608, 836, 900, 10, 18, 19, 20, 1, 3,
                   45, 96, 140, 157, 179, 191, 204, 223, 262, 269, 283, 304, 305, 347, 363, 379, 506, 517, 521, 546,
                   573, 607, 656,
                   664, 671, 680, 792, 801]

noise_list = list(cca_data.keys())

cca_data_formatted = {}
for noise_label in noise_list:
    cca_data_formatted[noise_label] = {}
    for i, obs_idx in enumerate(observation_vec):
        cca_data_formatted[noise_label][obs_idx] = {}
        cca_data_formatted[noise_label][obs_idx]['samples'] = cca_data[noise_label]['predicted'][i, :, :].reshape(2000,
                                                                                                                  500).transpose()
        cca_data_formatted[noise_label][obs_idx]['ground_truth'] = cca_data[noise_label]['ground_truth'][i, :].reshape(
            1, 2000)

save_to_disk(cca_data_formatted,
             "FastABC/data/geo_nonlinear/other_methods_inference_metrics/CCA/inversion_data_formatted.pkl")

from fastabc_inversion.geo_problems.utils.evaluation.scorers import torch_es

cca_metrics_formatted = {}
metrics = {'rmse': None, 'es': [1, 2], 'vs': [0.5]}
for metric in metrics.keys():
    cca_metrics_formatted[metric] = {}
    for noise_label in noise_list:
        if noise_label == 'small_noise':
            noise_label_target = 'small_gauss'
        elif noise_label == 'large_noise':
            noise_label_target = 'large_gauss'

        if metric == 'rmse':
            cca_metrics_formatted[metric][noise_label_target] = cca_metrics[noise_label][metric]

        elif metric == 'es':
            for p in metrics[metric]:
                if p not in cca_metrics_formatted[metric].keys():
                    cca_metrics_formatted[metric][p] = {}

                if p == 1:
                    cca_metrics_formatted[metric][p][noise_label_target] = cca_metrics[noise_label]['es']
                elif p == 2:
                    es_stats = []
                    for obs_idx in observation_vec:
                        ref_x = cca_data_formatted[noise_label][obs_idx]['ground_truth']
                        samples = cca_data_formatted[noise_label][obs_idx]['samples']
                        es_stats.append(torch_es(ref_x, samples, power=p, on_gpu=True))
                    cca_metrics_formatted[metric][p][noise_label_target] = es_stats
                else:
                    pass

        elif metric == 'vs':
            for p in metrics[metric]:
                if p not in cca_metrics_formatted[metric].keys():
                    cca_metrics_formatted[metric][p] = {}
                cca_metrics_formatted[metric][p][noise_label_target] = cca_metrics[noise_label]['vs']
        else:
            pass

save_to_disk(cca_metrics_formatted,
             "FastABC/data/geo_nonlinear/other_methods_inference_metrics/CCA/inversion_metrics_formatted.pkl")







