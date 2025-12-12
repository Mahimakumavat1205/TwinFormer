# TwinFormer: A Dual-Level Transformer for Long-Sequence Time-Series Forecasting

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**TwinFormer** is a hierarchical Transformer architecture designed for efficient and accurate Long-Sequence Time-Series Forecasting (LSTSF). It addresses the quadratic complexity limitations of vanilla Transformers by introducing a dual-level processing mechanism that captures both fine-grained local dynamics and long-range global dependencies with linear complexity $O(kLd)$.

> **Paper Title:** TwinFormer: A Dual-Level Transformer for Long-Sequence Time-Series Forecasting  
> **Authors:** Mahima Kumavat & Aditya Maheshwari (Indian Institute of Management Indore)

---

## Key Features

* **Hierarchical Architecture:** Processes data in two stages:
    1.  **Local Informer:** Models intra-patch dynamics using Top-k Sparse Attention.
    2.  **Global Informer:** Models inter-patch dependencies on compressed representations.
* **Linear Complexity:** Achieves $O(kLd)$ time and memory complexity, enabling training on sequences exceeding $10^5$ time steps.
* **Top-k Sparse Attention:** A deterministic attention mechanism that focuses only on the most relevant keys, offering superior stability compared to ProbSparse.
* **GRU Aggregation:** A lightweight Gated Recurrent Unit aggregates globally contextualized tokens for direct multi-horizon prediction.
* **SOTA Performance:** Validated on 8 real-world datasets across domains (Weather, Energy, Stock, Disease), achieving state-of-the-art results in 27/32 experimental settings.

---

## Model Architecture

The TwinFormer conceptually treats time series data as a hierarchy of patches.

1.  **Patching:** The input series is divided into non-overlapping patches.
2.  **Local Stage:** A shared Transformer encoder (Local Informer) processes tokens within each patch.
3.  **Patch Pooling:** Local representations are aggregated (mean pooling) to reduce sequence length.
4.  **Global Stage:** A Global Informer processes the sequence of patch embeddings to capture long-term trends.
5.  **Decoder:** A GRU aggregates the sequence, followed by a linear head for forecasting.

---

## Directory Structure

To run the code successfully, ensure your project directory is structured as follows. The code expects datasets to be in a `../Data/` folder relative to the script.

```text
TwinFormer/
├── Data/            
│   ├── temperatures.csv
│   ├── powerconsumption.csv
│   ├── weather_utf8.csv
│   ├── Electricity_load.csv
│   ├── powerconsumption.csv
│   ├── ETTh1.csv
│   ├── ETTm1.csv
│   ├── ETTh2.csv
│   ├── ETTm2.csv
│   ├── ILINet.csv
│   └── IDEA.csv
│
├── scripts/
│   └── main.py               # The provided training script
├── README.md
└── requirements.txt
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/TwinFormer.git](https://github.com/yourusername/TwinFormer.git)
    cd TwinFormer
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy torch scikit-learn matplotlib
    ```

---

## Datasets

This implementation includes loaders and preprocessing for the following domains:

| Dataset | Target | CSV Filename |
| :--- | :--- | :--- |
| **Temperature** | Univariate | `temperatures.csv` |
| **Power Consumption** | Multivariate | `powerconsumption.csv` |
| **Weather** | Multivariate (21 features) | `weather_utf8.csv` |
| **Electricity Load** | Multivariate (370 features) | `Electricity_load.csv` |
| **Stock Market** | Univariate (Close Price) | `IDEA.csv` |

*Note: Ensure your CSV files are formatted correctly (headers matching the code's expectations).*

---

## Usage

The main script runs experiments sequentially for different domains and prediction lengths. You can run the entire suite using:

```bash
python src/main.py
```

### Hyperparameters
The default configuration uses the settings described in the paper:
* **Input Sequence Length (`SEQ_LEN`):** 48
* **Prediction Lengths (`PRED_LEN`):** [96, 120, 336, 720]
* **Patch Size:** 6
* **Embedding Dimension:** 32 (variable per dataset)
* **Top-k:** 5

---

## Citation

If you use this code or model in your research, please cite the original paper:

```bibtex
@article{Kumavat2024TwinFormer,
  title={TwinFormer: A Dual-Level Transformer for Long-Sequence Time-Series Forecasting},
  author={Kumavat, Mahima and Maheshwari, Aditya},
  journal={},
  year={2025},
  url = {}
}
```
