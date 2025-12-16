# 3D U-Net Super-Resolution Model

This repository is based on a **fork** of the work by **Jiang, Chiyu Max, Soheil Esmaeilzadeh, Kamyar Azizzadenesheli, et al. (2020)**, corresponding to the paper:  

> *MeshfreeFlowNet: A Physics-Constrained Deep Continuous Space-Time Super-Resolution Framework*  
> [📄 Original Paper](https://arxiv.org/abs/2005.01463) | [💻 Original GitHub](https://github.com/maxjiang93/space_time_pde?tab=readme-ov-file)

Their model performs **super-resolution for 2D simulations** (2 spatial dimensions + 1 time dimension). In this project, the model is adapted to **downscale 3D snapshots of simulations** (3 spatial dimensions, no time dimension). This keeps the architecture largely unchanged while allowing 3D data processing.  

The **report** for this project, containing context, results, and discussion, is available as `report.pdf` in the root of this repository.

---

## 📚 Overview

This README explains:

- 1️⃣ How to install the environment  
- 2️⃣ How to download the datasets used in the report  
- 3️⃣ How to train a 3D super-resolution model  
- 4️⃣ How to evaluate the model  

## 📎 Notes
- All experiments are launched from the ```experiments/``` directory.
- The ```src/``` directory contains the core implementation and typically does not require modification.
- Refer to report.pdf for detailed explanations of datasets, metrics, and results.


---

## 1️⃣ Setting up the environment

Create a reproducible environment using **Conda**:

```bash
conda env create -f environment.yml
conda activate thomas
```


## 2️⃣ Download Datasets
To train the model, you need a training and an evaluation dataset in the ```.npz``` format with physical variables (for example $p$, $u$, $v$ and $w$), each of them of shape $\left(n_t, n_x, n_y, n_z \right)$. This is the case for the **3D isotropic turbulence dataset** and the **MHD dataset**
used in the report.

The datasets can be downloaded using:

```
experiments/
└── Download_Data/
    ├── Download_JHTDB_Data.py
    └── Download_MHD_data.ipynb
```
⚠️ **Note:** Access to the JHTDB database requires prior authorization.

For the ***ABL*** dataset, you can only copy it from another folder and then merge it using:
```
experiments/
└── Download_Data/
    ├── Copy_ABL_Data.py
    └── Merge_ABL_Data.py
```
⚠️ **Note:** This will create four ```.npy``` files (one for each variables), stored in a directory. The code can also take this format as input, but you need to specify the args ```--data_folder_training``` and 
  ```--data_folder_evaluation``` instead of ```--train_data``` and ```--eval_data``` in the training and evaluation script.
## 3️⃣ Train a model
To train the model, run:

```./run_training.sh```

This will create a new directory with the model weights and a plot of the losses curves. Hyperparameters can be configured directly inside the script.
Training and evaluation are designed to run on an ***HPC cluster using Slurm***.
All logs and printed outputs are saved in the ```logs/``` directory. 

## 4️⃣ Model Evaluation
To evaluate a trained model, run:

```./run_evaluation.sh```

This will create a new directory with the evaluation metrics and images of an example. All logs and printed outputs are saved in the ```logs/``` directory. 
