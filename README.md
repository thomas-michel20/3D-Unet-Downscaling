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
conda activate rb2d


## 2. Download Datasets
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

## 3. Train a model
To train the model, you should run the training script ```run_training.sh```. In this ```.sh``` you can tune the various hyper-parameters of the model. The training and evaluation are done using a HPC cluster and Slurm. The logs are stored in `logs` folder.

## 4. Evaluation of a model
To evaluate the model, you should run the evaluation script ```run_evaluation.sh```. 
