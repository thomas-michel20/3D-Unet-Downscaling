# 3D U-net super-resolution model

This repository is a based on a fork of the repository from Jiang, Chiyu Max, Soheil Esmaeilzadeh, Kamyar Azizzadenesheli, et al. 2020, corresponding to the paper $\textit{MeshfreeFlowNet: A Physics-Constrained Deep Continuous Space-Time Super-Resolution Framework}.$ [[Original paper](https://arxiv.org/abs/2005.01463)]
[[original GitHub](https://github.com/maxjiang93/space_time_pde?tab=readme-ov-file)]

Their super-resolution model is downscaling 2D simulations, which contains 2 spatial dimensions and 1 time dimension. In this project the model is adapted to ***downscale 3D snapshots of simulations***, which contains 3 spatial dimensions but no time dimension. Keeping the same number of dimensions allows relatively small changes in the code and the model's architecture remains the same. Please refer to their paper for more in-detail informations about the model. 

The report of this project, containing context, results and discussion is the file ```report.pdf```, that you can find it in the main root of this repository.

This ```READ.ME``` explains:
- How to install the environment
- How to download the same datasets as the report
- How to train a 3D super-resolution model
- How to evaluate a 3D super-resolution model

All these part are done in the ```/experiment``` folder. You shouldn't need to modify the ```/src``` folder. 

## 1. Setting up the environment
you have the environment.yml ...

## 2. Download Datasets
To train the model, you need a training and an evaluation dataset in the ```.npz``` format with physical variables (for example $p$, $u$, $v$ and $w$), each of them of shape $\left(n_t, n_x, n_y, n_z \right)$. This is the case for the **3D isotropic turbulence dataset** and the **MHD dataset**
used in the report.

The datasets can be downloaded using:


experiments/
└── Download_Data/
    ├── Download_JHTDB_Data.py
    └── Download_MHD_data.ipynb

- `experiments/Download_Data/Download_JHTDB_Data.py`
- `experiments/Download_Data/Download_MHD_data.ipynb`

⚠️ **Note:** Access to the JHTDB database requires prior authorization.

For the ABL dataset, you can only copy it from another folder and then merge it using consecutively:

experiments/
└── Download_Data/
    ├── Copy_ABL_Data.py
    └── Merge_ABL_Data.py

- `experiments/Download_Data/Copy_ABL_Data.py`
- `experiments/Download_Data/Merge_ABL_Data.py`

## 3. Train a model
To train the model, you should run the training script ```run_training.sh```. In this ```.sh``` you can tune the various hyper-parameters of the model. The training and evaluation are done using a HPC cluster and Slurm. The logs are stored in `logs` folder.

## 4. Evaluation of a model
To evaluate the model, you should run the evaluation script ```run_evaluation.sh```. 