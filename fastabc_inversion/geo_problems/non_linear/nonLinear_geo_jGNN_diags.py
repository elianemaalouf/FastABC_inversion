"""
Written by Eliane Maalouf (eliane.maalouf@unine.ch)
Script to run jGNN diagnostics for the linear geophysics problem.
"""

import time

import fastabc_inversion.geo_problems.Diagnostics as diag
from fastabc_inversion.geo_problems.jGNN_SuS_exp import \
    load_experiment_from_file

# run diagnostics
epoch = 1999
experiment_id = 91
experiment_folder_name = "NonLin_Geo_jGNN_SuS_Matern_81_id_91_5OV4I"
dir_root = "../Geo_jGNN_SuS_experiments"
model_training_dir = f"{dir_root}/{experiment_folder_name}/model_training_data"

geo_jGNN_exp = load_experiment_from_file(
    f"{model_training_dir}/models/experiment_epoch_{epoch}.pkl"
)

diagnostics_to_run = []  # if empty: runs all

diags_params = {}  # if empty: takes defaults

diag_geo_jGNN_exp = diag.Diagnostics(geo_jGNN_exp, epoch=epoch)
diag_start_time = time.time()
diag_geo_jGNN_exp.run_diagnostics(
    diagnostics_to_run=diagnostics_to_run,
    diags_params=diags_params,
    logging_comment=f"{experiment_id}_epoch{epoch}",
)
diag_end_time = time.time()
print(f"Time to run diagnostics: {diag_end_time - diag_start_time:.3f} s")
