# 🔥 Heat Exchanger Heat Load Prediction

A machine learning project that predicts the **heat load (kW)** of an industrial heat exchanger using both clean and sensor-noisy thermodynamic data. The core investigation explores how well ML models can filter real-world sensor noise to reconstruct true physical behavior.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

---

## Overview

Heat exchangers are critical components in industrial systems — from power plants to chemical processing facilities. Accurately predicting the **heat load** (Q, in kW) from sensor measurements enables real-time monitoring, fault detection, and energy optimization.

This project benchmarks four regression algorithms across three distinct scenarios to understand the effect of sensor noise on predictive accuracy and demonstrates how ensemble methods can act as **soft-computing signal filters** to recover the underlying thermodynamics from corrupted data.

---

## Problem Statement

Heat transfer through a counter-current heat exchanger follows:

```
Q = U * A * LMTD
```

Where the **Logarithmic Mean Temperature Difference (LMTD)** is:

```
LMTD = (ΔT1 - ΔT2) / ln(ΔT1 / ΔT2)
```

- ΔT1 = T_h,in  - T_c,out  (Temperature difference at one end)
- ΔT2 = T_h,out - T_c,in   (Temperature difference at the other end)
- U = Overall heat transfer coefficient (held constant in the clean dataset)
- A = Heat transfer surface area (held constant in the clean dataset)

In real industrial environments, sensor readings are corrupted by Gaussian noise, calibration drift, and telemetry drops. The key challenge is: **can ML models reconstruct the true heat load from noisy sensor inputs?**

---

## Dataset

**File:** `heat_exchanger_dataset_with_missing.csv`

| Property | Value |
|---|---|
| Total Rows | 10,000 |
| Total Columns | 20 |
| Missing Values | < 0.1% (sparse telemetry drops) |

### Features

The dataset contains **paired clean and noisy versions** of each physical measurement:

| Feature | Unit | Description |
|---|---|---|
| `hot_inlet_temperature_k` | K | Hot fluid inlet temperature |
| `cold_inlet_mass_flow_kg_s` | kg/s | Cold fluid inlet mass flow rate |
| `hot_outlet_temperature_k` | K | Hot fluid outlet temperature |
| `cold_outlet_temperature_k` | K | Cold fluid outlet temperature |
| `hot_outlet_pressure_pa` | Pa | Hot fluid outlet pressure |
| `cold_outlet_pressure_pa` | Pa | Cold fluid outlet pressure |
| `hot_outlet_mass_flow_kg_s` | kg/s | Hot fluid outlet mass flow rate |
| `cold_outlet_mass_flow_kg_s` | kg/s | Cold fluid outlet mass flow rate |
| `hx_1_logarithmic_mean_temperature_difference_lmtd_k` | K | LMTD across the heat exchanger |

Each feature above has a corresponding `*_noisy` column representing the corrupted sensor reading.

### Targets

| Target Column | Description |
|---|---|
| `hx_1_heat_load_kw` | True (clean) heat load in kW |
| `hx_1_heat_load_kw_noisy` | Sensor-measured (noisy) heat load in kW |

---

## Methodology

### 1. Data Preprocessing

- **Missing Value Imputation:**
  - Clean features → **Iterative Imputer with Linear Regression** (preserves deterministic physical relationships)
  - Noisy features → **Iterative Imputer with Bayesian Ridge** (handles stochastic sensor noise)
- **Outlier Removal:** IQR filtering (1.5x IQR rule) applied to the target columns to remove unphysical sensor spikes

### 2. Three Modeling Scenarios

| # | Scenario | Input Features | Target |
|---|---|---|---|
| 1 | **Clean Data** | Clean sensor readings | True heat load |
| 2 | **Noisy Data** | Noisy sensor readings | Noisy heat load |
| 3 | **Signal Reconstruction** | Noisy sensor readings | True (clean) heat load |

Scenario 3 is the most practically relevant — it simulates deploying an ML model on live industrial sensor data to recover the true physical quantity.

### 3. Training Pipeline

- **80/20 train-test split** with `random_state=42`
- **StandardScaler** normalization applied to all features
- Metrics: R², RMSE (kW), MAE (kW)

---

## Models & Results

Four regression algorithms were benchmarked:

- **Linear Regression**
- **Ridge Regression** (α = 1.0)
- **Random Forest** (100 estimators)
- **Gradient Boosting** (100 estimators)

### Performance Summary

| Scenario | Model | R² | RMSE (kW) | MAE (kW) |
|---|---|---|---|---|
| **Clean Data** | Linear Regression | **1.0000** | 0.0001 | 0.0000 |
| **Clean Data** | Ridge | 1.0000 | 0.0123 | 0.0100 |
| **Clean Data** | Random Forest | 1.0000 | 0.0276 | 0.0056 |
| **Clean Data** | Gradient Boosting | 0.9999 | 0.1435 | 0.1088 |
| | | | | |
| **Noisy Data** | Gradient Boosting | **0.5287** | 17.1103 | 13.4707 |
| **Noisy Data** | Linear Regression | 0.5232 | 17.2104 | 13.6032 |
| **Noisy Data** | Ridge | 0.5232 | 17.2103 | 13.6032 |
| **Noisy Data** | Random Forest | 0.5041 | 17.5521 | 13.8023 |
| | | | | |
| **Signal Reconstruction** | Gradient Boosting | **0.9107** | 5.7708 | 4.5820 |
| **Signal Reconstruction** | Random Forest | 0.9043 | 5.9742 | 4.7626 |
| **Signal Reconstruction** | Linear Regression | 0.8965 | 6.2116 | 4.9703 |
| **Signal Reconstruction** | Ridge | 0.8965 | 6.2116 | 4.9704 |

---

## Key Findings

1. **Perfect Clean Modeling** — Linear Regression achieves a flawless R² = 1.0000 on uncorrupted data, confirming the dataset is governed by exact physical equations (Q = U·A·LMTD).

2. **The Noise Barrier** — When both features and targets contain noise, all models plateau at R² ≈ 0.53. This is not model failure — it is **irreducible error**: the random component of the noisy target cannot be predicted by any deterministic model.

3. **Noise-Filtering Capability** — In Scenario 3, Gradient Boosting achieves R² = 0.9107 with RMSE = 5.77 kW, using only corrupted sensor inputs to predict the true physical heat load. This demonstrates that tree-based ensembles can act as **effective soft-computing signal filters**, recovering the underlying thermodynamics through the noise.

---

## Project Structure

```
ML Project-Heat Exchanger/
│
├── Heat_Load_Prediction.ipynb               # Main analysis notebook
├── heat_exchanger_dataset_with_missing.csv  # Dataset (10,000 rows, 20 columns)
└── README.md                                # This file
```

---

## Getting Started

### Prerequisites

Ensure you have **Python 3.8+** and the packages listed below installed.

### Running the Notebook

1. Clone or download this project directory.
2. Install the required dependencies (see below).
3. Launch Jupyter:
   ```bash
   jupyter notebook Heat_Load_Prediction.ipynb
   ```
4. Run all cells sequentially (`Kernel → Restart & Run All`).

> **Note:** The dataset file `heat_exchanger_dataset_with_missing.csv` must be in the same directory as the notebook.

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Imputation, scaling, ML models, and metrics |
| `jupyter` | Notebook environment |

---

## Physics Background

A **counter-current heat exchanger** transfers thermal energy between two fluid streams flowing in opposite directions. This arrangement maximizes the temperature gradient throughout the exchanger, making it more efficient than parallel-flow designs.

The thermodynamic relationship used in this project:

```
Q = m_dot_c * c_p,c * (T_c,out - T_c,in)
  = m_dot_h * c_p,h * (T_h,in  - T_h,out)
```

In the clean dataset, U·A is held constant, creating a perfect linear relationship between LMTD and heat load that linear regression can recover exactly.
