# FastABC methodology for solving inverse problems 

Data and Python code repository to reproduce the results in chapters 4 and 5 of the thesis 
"Contributions to data-driven Bayesian solutions to inverse problems: from classical multivariate statistics to modern
generative neural networks" by Eliane Maalouf (University of Neuchâtel, Switzerland).

# Contents
- `01_geophysics_inverse_problem_docs/`: Contains the plots from the thesis chapter 4 and detailed results.
- `02_geophysics_nonlinear_inverse_problem_docs/`: Contains the plots from the thesis chapter 5 (non linear geo problem) and detailed results.
- `03_conditional_generation_docs/`: Contains the plots from the thesis chapter 5 (conditional image generation problem) and detailed results.
- `data/`: Contains the data used in the experiments, including the geophysical data and the image data for the conditional generation problem.
The geophysical data generation code is available in the `fastabc_inversion/geo_problems/data_simulation/`subpackage.
- `fastabc_inversion/`: Python package containing the code to run the FastABC inversion experiments. 
We separate all codes pertaining to geophysical experiments in the `fastabc_inversion/geo_problems/` subpackage 
and all files pertaining to the conditional image generation problem in the `fastabc_inversion/conditional_generation/` subpackage.
Main files: 
  - `fastabc_inversion/conditional_generation/mnist/mnist.py`: main script to run the FastABC inversion experiments for 
  the conditional image generation problem with MNIST data. 
  - `fastabc_inversion/geo_problems/linear/analytical_inversion.py`: main script to run the analytical inversion for the linear geophysical problem.
  - `fastabc_inversion/geo_problems/linear/linear_geo_cVAE.py`: main script to train and run conditional generation experiments for the linear geophysical problem with a conditional VAE.
  - `fastabc_inversion/geo_problems/linear/linear_geo_jGNN.py`: main script to train the joint Sinkhorn Autoencoder (jSAE) for the linear geophysical problem.
  - `fastabc_inversion/geo_problems/linear/linear_geo_SuS.py`: main script to run the Subset Simulation (SuS) method for the linear geophysical problem.
  - `fastabc_inversion/geo_problems/non_linear/nonlinear_geo_jGNN.py`: main script to train the joint Sinkhorn Autoencoder (jSAE) for the non-linear geophysical problem.
  - `fastabc_inversion/geo_problems/non_linear/nonlinear_geo_SuS.py`: main script to run the Subset Simulation (SuS) method for the non-linear geophysical problem.

# Note on reproducibility
All experiments are run with fixed seeds (provided in the parameters files for the geophysical problems and in the 
`fastabc_inversion/conditional_generation/mnist/mnist.py` script for the conditional generation problem).
However, due to the use of GPU computations and the non-deterministic nature of some operations in PyTorch,
the results may not be exactly reproducible across different runs or different machines.
See https://pytorch.org/docs/stable/notes/randomness.html


## Usage
For convenience, we ran the scripts from the console of Pycharm IDE, but they can be easily adapted to run 
from the command line. Particular attention should be paid to the paths of the data and results folders. 

# Requirements
Python 3.9 or higher is required to run the code.
The required packages are listed in the `requirements.txt` file.

# Disclaimer 
This software is provided 'as is' without any warranty, express or implied. 
Please see the `LICENSE` file for full details.




