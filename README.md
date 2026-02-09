# ⚛️ Data-Driven Functional-Group Atlas (FGA)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.placeholder)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Official implementation for "A data-driven functional-group atlas for programming interfacial wettability and heat transport for thermal energy storage"**

This repository houses the machine learning workflow used to decouple the intrinsic trade-off between interfacial wettability (*Eb*) and thermal conductivity (*NVOA*) in Carbon-Molten Salt composites. By integrating **rational feature deconstruction** with a **Stacking Ensemble framework**, this pipeline provides a transferable paradigm for designing heterogeneous interfaces in energy systems.

---

## 🚀 Methodological Framework

The workflow transitions interface design from empirical trial-and-error to rational programming through three core innovations:

*   **🧪 Rational Feature Deconstruction**: A physically interpretable encoding strategy that resolves functional groups into orthogonal "Elemental" and "Structural" dimensions.
*   **⚙️ Hierarchical Feature Selection**: A rigorous "Filter-Embedded-Wrapper" pipeline that combines Mutual Information, SHAP-based pruning, and Recursive Feature Elimination (RFE) to identify minimal yet potent descriptor sets.
*   **🧠 Robust Stacking Ensemble**: A two-tier architecture integrating 7 heterogeneous base learners (XGBoost, LightGBM, CatBoost, etc.) with an ElasticNet Meta-Learner, optimized via Bayesian search.
*   **📊 Two-Level Weighted SHAP**: A novel interpretability method that weights feature contributions by both *intra-fold* model fidelity and *inter-fold* generalization capability, ensuring robust mechanistic insights.

---

## 📂 Repository Structure

```plaintext
.
├── data/
│   ├── raw/
│   │   └── Original database-CR.xlsx          # Source data (DFT/AIMD labels + Descriptors)
│   ├── processed/
│   │   └── final_engineered_dataset-VDOS.xlsx # Output of feature engineering (Input for ML)
│   └── external/
│       └── yuceji-VDOS.xlsx                   # Candidate library for prediction (Unknown data)
│
├── scripts/
│   ├── Feature_Engineering.ipynb              # [Script 1] 3-Stage Feature Selection Pipeline
│   └── Tree_stacking.ipynb                    # [Script 2] Optimization, Stacking, Evaluation & Prediction
│
├── results/
│   ├── Feature_Engineering_Summary.xlsx       # Stepwise feature reduction logs
│   ├── SHAP_Analysis_Results.xlsx             # Global feature importance & mechanism discovery
│   └── Final_Predictions.xlsx                 # Predicted properties for candidate library
│
├── environment/
│   ├── spec-file.txt                          # Exact reproducibility (Windows x64)
│   └── requirements.txt                       # General dependencies
│
├── LICENSE
└── README.md
```

---

## 💻 Usage Guide

### 1. Environment Setup
To ensure exact numerical reproducibility of the Bayesian Optimization trajectories, we recommend using the locked Conda specification.

```bash
# Option A: Exact Reproducibility (Recommended)
conda create --name fga-env --file environment/spec-file.txt
conda activate fga-env

# Option B: Standard Installation
pip install -r environment/requirements.txt
```

### 2. Execution Workflow

The pipeline is divided into two sequential modules. Ensure `Original database-CR.xlsx` is present in the directory before starting.

#### Phase I: Feature Engineering
**Goal:** Distill high-dimensional descriptors into an optimal subset.
1.  Open `Feature_Engineering.ipynb`.
2.  Configure the target column index (default is set to extract descriptors).
3.  Run all cells.
    *   **Output:** Generates `final_engineered_dataset-VDOS.xlsx` and correlation matrices.

#### Phase II: Modeling & Prediction
**Goal:** Hyperparameter tuning, Stacking construction, and candidate screening.
1.  Open `Tree_stacking.ipynb`. This notebook consolidates three critical tasks:
    *   **Bayesian Optimization:** Automatically tunes 7 base learners using 10-fold CV.
    *   **Ensemble Evaluation:** Constructs the Stacking model and generates Two-Level SHAP plots.
    *   **Inference:** Loads the candidate library (`yuceji-VDOS.xlsx`), aligns features automatically, and outputs predictions.
2.  **Configuration:**
    *   Set `ENABLED_MODELS` to select base learners.
    *   Set `REUSE_PRETRAINED_STACKING_MODEL = False` for the first run to trigger full retraining.
3.  Run all cells.

---

## ⚙️ Key Configuration Parameters

Major algorithmic controls are located in the "CONFIGURATION AREA" at the top of each script.

| Parameter | Script | Description |
| :--- | :--- | :--- |
| `FILTER_METHOD_CRITERION` | Feature Eng. | Selection logic for correlated features (`mutual_info` or `pearson`). |
| `SHAP_COARSE_SELECTION_PERCENT` | Feature Eng. | Percent of features retained after Embedded selection (Stage 2). |
| `N_ITER_BAYESIAN` | Stacking | Number of iterations for Gaussian Process optimization. |
| `WEIGHTING_METHOD` | Stacking | Aggregation logic for SHAP values (e.g., `1/RMSE`). |
| `N_SEEDS_FOR_EVALUATION` | Stacking | Number of random seeds for nested cross-validation (Robustness check). |

---

## 🤝 Citation & Correspondence

If you use this code or data in your research, please cite our paper:

> **A data-driven functional-group atlas for programming interfacial wettability and heat transport for thermal energy storage**
> *Yifei Zhu, Tiansheng Wang, Guangmin Zhou*
> [Journal Name], 202X.

**Correspondence:**
For technical questions or collaboration, please contact Prof. Guangmin Zhou ([guangminzhou@sz.tsinghua.edu.cn](mailto:guangminzhou@sz.tsinghua.edu.cn)).

---

*Code developed by Yifei Zhu at Tsinghua Shenzhen International Graduate School.*
```
