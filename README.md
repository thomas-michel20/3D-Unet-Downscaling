# 🚀 3D U-Net Super-Resolution Model

This repository is based on a **fork** of the work by **Jiang, Chiyu Max, Soheil Esmaeilzadeh, Kamyar Azizzadenesheli, et al. (2020)**, corresponding to the paper:  

> *MeshfreeFlowNet: A Physics-Constrained Deep Continuous Space-Time Super-Resolution Framework*  
> [📄 Original Paper](https://arxiv.org/abs/2005.01463) | [💻 Original GitHub](https://github.com/maxjiang93/space_time_pde?tab=readme-ov-file)

Their model performs **super-resolution for 2D simulations** (2 spatial dimensions + 1 time dimension). In this project, the model is adapted to **downscale 3D snapshots of simulations** (3 spatial dimensions, no time dimension). This keeps the architecture largely unchanged while allowing 3D data processing.  

The **report** for this project, containing context, results, and discussion, is available as `report.pdf` in the root of this repository.

---

## 📚 Overview

This README explains:

- ✅ How to install the environment  
- ✅ How to download the datasets used in the report  
- ✅ How to train a 3D super-resolution model  
- ✅ How to evaluate the model  

All tasks are performed in the `experiments/` folder. You **should not need to modify the `src/` folder**.

---

## 1️⃣ Setting up the environment

Create a reproducible environment using **Conda**:

```bash
conda env create -f environment.yml
conda activate thomas```


## 2️⃣ Download Datasets
To train the model, you need a training and an evaluation dataset in the ```.npz``` format with physical variables (for example $p$, $u$, $v$ and $w$), each of them of shape $\left(n_t, n_x, n_y, n_z \right)$. This is the case for the **3D isotropic turbulence dataset** and the **MHD dataset**
used in the report.

The datasets can be downloaded using:


experiments/
└── Download_Data/
    ├── Download_JHTDB_Data.py
    └── Download_MHD_data.ipynb

⚠️ **Note:** Access to the JHTDB database requires prior authorization.

For the ABL dataset, you can only copy it from another folder and then merge it using consecutively:

experiments/
└── Download_Data/
    ├── Copy_ABL_Data.py
    └── Merge_ABL_Data.py

## 3️⃣ Train a model
To train the model, run:
```./run_training.sh```

Hyperparameters can be configured directly inside the run_training.sh script.
Training and evaluation are designed to run on an ***HPC cluster using Slurm***.
All logs and outputs are saved in the logs/ directory.

## 4️⃣ Model Evaluation
To evaluate a trained model, run:
```./run_evaluation.sh```.

## 📎 Notes
- All experiments are launched from the ```experiments/``` directory.
- The ```src/``` directory contains the core implementation and typically does not require modification.
- Refer to report.pdf for detailed explanations of datasets, metrics, and results.
