# 🔬 NetVax Project: Code Introduction and Implementation Guide

---

> 📋 **Overview**: Implementation guide for feature selection and network traffic anomaly detection, covering system environment, data processing pipeline, feature selection module, and NetVax optimized model training/detection workflow with hyperparameter configurations.

---

## 📦 1. System Environment

| Item | Configuration |
|:---|:---|
| 🖥️ **Operating System** | Windows NT 10 / x64 |
| ⚙️ **CPU** | AMD Ryzen 9800X3D |
| 🎮 **GPU** | NVIDIA GeForce RTX 5070 Ti O16G |
| 🔧 **Driver Version** | Driver 610.62 / CUDA UMD 13.3 |
| 🐍 **Python Environment** | Python 3.11 |

### Core Dependencies

```
NumPy  •  Pandas  •  Scikit-learn  •  Scipy  •  Torch  •  Torchmetrics
mrmr-selection/mrmr  •  Python-docx (optional)
```

> 💡 The code automatically selects **CUDA** or **CPU** based on `torch.cuda.is_available()`.

---

## 🎯 2. Feature Selection Module

### 2.1 Input and Preprocessing

Data preprocessing pipeline:

- ✅ Data loading
- ✅ Irrelevant label removal
- ✅ Missing/invalid value handling
- ✅ Data normalization
- ✅ Train/test split

### 2.2 NetVax Feature Selection Optimization

#### 📌 Core Algorithm Flow

| Step | Function | Description |
|:---:|:---|:---|
| 1️⃣ | **Population Initialization** | Sobol low-discrepancy sequence + Gaussian-Cauchy mixed distribution + Feature priority weights |
| 2️⃣ | **Affinity Calculation** | KNN cross-validation error + Drift term + Cost term |
| 3️⃣ | **Cloning and Mutation** | Gaussian-Cauchy heavy-tailed mutation kernel |
| 4️⃣ | **Adaptive Adjustment** | Population size + Mutation step size adjusted by diversity feedback |
| 5️⃣ | **Gaussian Walk** | Generate new candidates around elite individuals when diversity is insufficient |

#### 🎯 Key Features

> **Drift Penalty and Cost Penalty**: Adding drift penalty and feature cost penalty on top of relevance and redundancy, aiming to maintain discriminative power while reducing the influence of unstable and high-cost features.

### 2.3 🔧 Feature Selection Hyperparameters

#### 📊 Population Control Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `population_init` | 50 | N_init | Initial population size |
| `population_min` | 20 | N_min | Minimum population size |
| `selection_size` | 10 | - | Selection size |
| `clone_rate` | 0.5 | - | Cloning rate |
| `stop_condition` | 50 | G_max | Maximum iterations |

#### 🔄 Mutation Kernel Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `tau` | 0.5 | τ | Gaussian-Cauchy mixing weight |
| `tau_init` | 0.7 | τ_init | Initial mutation kernel τ value |
| `tau_final` | 0.3 | τ_final | Final mutation kernel τ value |
| `sigma_min` | 0.05 | σ_min | Minimum mutation step |
| `sigma_max` | 0.5 | σ_max | Maximum mutation step |

#### 📈 Adaptive Factor Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `kappa_p1` | 5.0 | κ₁ | Time sensitivity |
| `kappa_p2` | 3.0 | κ₂ | Signal sensitivity |
| `theta` | 0.5 | θ | Annealing midpoint position |

#### 📉 Population Reduction Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `beta` | 1.5 | β | Population reduction curve shape |
| `kappa_tanh` | 0.5 | κ | Diversity feedback amplitude |
| `nu` | 10.0 | ν | Diversity feedback sensitivity |
| `delta_c` | 0.01 | δ_c | Diversity threshold |

#### ⚖️ mRMR Weight Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `lambda_g` | 0.5 | λ_g | Redundancy penalty weight |
| `rho_g` | 0.3 | ρ_g | Drift penalty weight |
| `xi_g` | 0.2 | ξ_g | Cost penalty weight |

#### 🔧 Other Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `amplification` | 1.0 | α | Priority amplification factor |
| `cov_scale` | 0.1 | σ₀² | Gaussian walk covariance scaling |

---

## 🧠 3. NetVax Optimized Model

### 3.1 Data Processing

| Processing Step | Description |
|:---|:---|
| 📊 Data Deduplication | Remove duplicate samples based on feature columns (excluding label) |
| 🏷️ Category Encoding | One-hot encoding for categorical features |
| 📐 Feature Scaling | MinMaxScaler for continuous features to [0, 1] |
| 🗑️ Field Filtering | Remove fields not directly involved in binary detection |

### 3.2 💡 Core Concept

> NetVax treats normal traffic as **"self"** and attack traffic as **"non-self"**. The model generates detector radii based on distances between normal sample centers and abnormal samples, then determines test traffic classification using hyper-sphere coverage: **classified as normal if falling within any mature detector's coverage, otherwise classified as abnormal**.

### 3.3 🔬 Main Methods

| Method | Description |
|:---|:---|
| 📐 **Mahalanobis Distance** | Calculate Mahalanobis distance using covariance matrix inverse |
| 🎯 **Geometric Center** | Use geometric median as vaccine center |
| 📏 **Adaptive Radius** | Determine vaccine radius using chi-square distribution quantile / sample distance quantile |
| 🛡️ **Instantiation Check** | Check if candidate detector provides new coverage, avoid conflicts with self-vaccine |
| 🧹 **Controlled Forgetting** | Update detector memory strength S based on conflict ratio, aging status, and consistency |

### 3.4 🔄 Training/Detection Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    NetVax Training Flow                      │
├──────────────────────────────────────────────────────────────
│  1️⃣ Sample Split → Normaldata & Abnormaldata                │
│  2️⃣ Batch Division → Multi-batch training, last batch test  │
│  3️⃣ Initial Detector → Build from first batch               │
│  4️⃣ Radius Calculation → 0.5 × min distance to anomalies    │
│  5️⃣ Vaccine Injection → Shrink old detectors + New detectors│
│  6️⃣ Quality Control → Instantiation check + Conflict proj.  │
│  7️⃣ Controlled Forgetting → Update memory + Prune weak      │
└──────────────────────────────────────────────────────────────┘
```

### 3.5 🔧 Model Hyperparameters

#### 📊 Core Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `num_vaccines` | 5 | - | Number of vaccine batches |
| `step` | 5000 | - | Batch processing step size |
| `budget` | 100 | - | Detector budget upper limit |

#### 📏 Detector Radius Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `alpha` | 0.95 | α | Vaccine radius confidence level |
| `r_min` | 0.01 | r_min | Detector radius lower bound |
| `r_max` | 10.0 | r_max | Detector radius upper bound |

#### 🧹 Controlled Forgetting Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `tau_del` | 0.1 | τ_del | Memory strength deletion threshold |
| `T_obsolete` | 50 | T_obsolete | Obsolescence threshold |
| `T_mature` | 20 | T_mature | Maturity threshold |
| `window_size` | 1000 | - | Sliding window size |

#### ⚖️ Pruning Weight Parameters

| Parameter | Default | Symbol | Description |
|:---|:---:|:---:|:---|
| `alpha_cf` | 0.5 | α | Conflict ratio weight (Eq.11) |
| `beta_cf` | 0.3 | β | Obsolescence weight (Eq.11) |

#### ⚡ Performance Optimization Parameters

| Parameter | Default | Description |
|:---|:---:|:---|
| `subsample_ratio` | 0.1 | Data subsampling ratio |
| `use_mahalanobis` | True | Whether to use Mahalanobis distance |
| `use_controlled_forgetting` | True | Whether to enable controlled forgetting |
| `GPU_CHUNK_SIZE` | 20000 | GPU chunk size |

---

## 📝 Quick Reference Guide

### Feature Selection Key Parameters

| Function | Parameter | Default | Tuning Suggestion |
|:---|:---|:---:|:---|
| Population size | `population_init` | 50 | Decrease for large datasets |
| Iterations | `stop_condition` | 50 | Increase if convergence is slow |
| Mutation strength | `sigma_max` | 0.5 | Increase for more exploration |
| Redundancy penalty | `lambda_g` | 0.5 | Increase for redundant features |
| Drift penalty | `rho_g` | 0.3 | Increase for non-stationary data |
| Cost penalty | `xi_g` | 0.2 | Increase for sparse solutions |

### Model Detection Key Parameters

| Function | Parameter | Default | Tuning Suggestion |
|:---|:---|:---:|:---|
| Vaccine count | `num_vaccines` | 5 | Increase for complex data |
| Batch size | `step` | 5000 | Decrease for insufficient memory |
| Confidence level | `alpha` | 0.95 | Increase for high false positives |
| Deletion threshold | `tau_del` | 0.1 | Decrease for too many detectors |
| Maturity period | `T_mature` | 20 | Decrease for faster response |

---
