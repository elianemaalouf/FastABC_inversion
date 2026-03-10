# correct analytical metrics format
from fastabc_inversion.geo_problems.Inference_Diagnostics import load_from_disk, save_to_disk

analytical_mertrics_file = "FastABC/data/geo_linear/other_methods_inference_metrics/analytical/inversion_metrics.pkl"
analytical_metrics = load_from_disk(analytical_mertrics_file)
noise_list = list(analytical_metrics.keys())
analytical_metrics_formatted = {}
metrics = {'rmse': None, 'es':[1 , 2], 'vs':[0.5]}
for metric in metrics.keys():
    analytical_metrics_formatted[metric] = {}
    for noise_label in noise_list:

        if metric == 'rmse':
            analytical_metrics_formatted[metric][noise_label] = analytical_metrics[noise_label][metric]

        else:
            for p in metrics[metric]:
                if p not in analytical_metrics_formatted[metric].keys():
                    analytical_metrics_formatted[metric][p] = {}
                analytical_metrics_formatted[metric][p][noise_label] = analytical_metrics[noise_label][metric][p]

save_to_disk(analytical_metrics_formatted,
             "FastABC/data/geo_linear/other_methods_inference_metrics/analytical/inversion_metrics_formatted.pkl")


