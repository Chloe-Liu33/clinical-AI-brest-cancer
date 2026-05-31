# Clinical AI for Breast Cancer 🔬🧬

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Note**: This repository contains the official implementation of our causality-driven multi-modal framework for breast cancer survival prediction, validated on the METABRIC cohort.

## 📖 Overview

This repository provides an end-to-end multi-modal Artificial Intelligence framework designed for breast cancer prognosis and survival prediction. By effectively fusing structured clinical records, high-dimensional Gene Expression (mRNA), and Copy-Number Alterations (CNA) data, this project addresses the limitations of single-modality clinical models and naive late fusion techniques.

At its core, this framework leverages an **HSIC-based (Hilbert-Schmidt Independence Criterion) fusion mechanism** to compress high-dimensional genomic data into task-aware components. This method not only significantly outperforms clinical-only baselines and standard concatenation but also delivers superior probability calibration, which is critical for clinical decision-making.

### 🌟 Key Features
- **Multi-modal Data Fusion**: Integrates three distinct biological views: Clinical (58 features), Gene Expression (20,384 genes), and CNA (22,544 genes).
- **HSIC-based Bottleneck (Task-aware Selection)**: Replaces naive variance-based gene selection with dependency learning, achieving a clear performance sweet spot at $K=512$.
- **Robust Clinical Calibration**: Achieves state-of-the-art results not just in ranking (C-index), but in probability calibration (Brier Score, ICI, E50, E90), ensuring the predicted survival risks correspond to true clinical reality.
- **Graceful Modality Absorption**: Demonstrates robustness against weak modalities (like CNA) where naive concatenation degrades.
- **End-to-End Pipeline**: Comprehensive scripts for dataset splitting (stratified 70/15/15), training, and robust evaluation.

## 📊 Experimental Results (METABRIC Ablation)

We rigorously evaluate our framework on the METABRIC cohort ($n  pprox 1,900$ patients) for overall-survival prediction. All results are averaged across 5 random seeds on a held-out test split.

### 1. Held-out Test C-index (Breaking the Clinical Ceiling)
Only our HSIC fusion mechanism significantly breaks the hard clinical-only ceiling on METABRIC.

| Model Variant | Test C-index | Parameters |
|---------------|--------------|------------|
| Clinical-only (MLP) | 0.6800 ± 0.0023 | ~15K |
| Gene-only (MLP) | 0.6340 ± 0.0090 | ~2.6M |
| Late Fusion (Concat, bi-modal) | 0.6792 ± 0.0097 | ~2.6M |
| Late Fusion 3 (Concat, tri-modal) | 0.6705 ± 0.0122 | ~5.5M |
| Gene Var-Top 512 | 0.6057 ± 0.0191 | ~73K |
| **Ours (HSIC K=512)** | **0.7008 ± 0.0063** | **~2.7M** |
| Ours-Tri (HSIC K=512) | 0.7002 ± 0.0190 | ~5.8M |

*Note: HSIC K=512 provides a +0.0208 gain over the Clinical-only baseline ($p=0.0005$) and proves that how we fuse is more important than capacity (uses half the parameters of Late Fusion 3).*

### 2. Survival Probability Calibration
A model with a good C-index can still output clinically misleading probabilities (e.g., predicting 10% risk when truth is 35%). Our HSIC-based framework dominates naive concatenation across all calibration metrics at both 5-year and 10-year horizons.

| Model | 5y Brier (↓) | 5y ICI (↓) | 10y Brier (↓) | 10y ICI (↓) |
|-------|--------------|------------|---------------|-------------|
| Clinical-only (MLP) | 0.164 | 0.062 | 0.211 | 0.088 |
| Late Fusion (Concat) | 0.168 | 0.138 | 0.236 | 0.174 |
| **Ours (HSIC K=512)**| **0.149** | **0.046** | **0.204** | **0.064** |

*(Lower is better. Naive late fusion suffers from a systematic under-prediction of risk, making its probabilities misleading despite a competitive C-index. HSIC fusion guarantees actionable clinical probabilities.)*

## 🛠️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Chloe-Liu33/clinical-AI-brest-cancer.git
cd clinical-AI-brest-cancer
```

**2. Create a virtual environment:**
```bash
conda create -n bc_ai python=3.9
conda activate bc_ai
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📂 Repository Structure

```text
clinical-AI-brest-cancer/
├── data/                   # Data preprocessing (log-scale, z-score standardisation)
│   ├── clinical_loader.py  # Processing 58 clinical features (Demographics, IHC, etc.)
│   ├── gene_loader.py      # Processing continuous mRNA z-scores
│   └── cna_loader.py       # Processing discrete GISTIC calls
├── models/                 # Model architectures
│   ├── hsic_fusion.py      # HSIC kernel-based dependency learning layers
│   └── baselines.py        # Single-modality and Naive Concat MLPs
├── utils/                  # Helper functions 
│   └── metrics.py          # Harrell's C-index, Brier Score, ICI, E50, E90
├── train.py                # Main training script
├── evaluate.py             # Inference and calibration evaluation
├── requirements.txt        
└── README.md               
```

## 🚀 Quick Start

### Training the HSIC Fusion Model
To reproduce the optimal HSIC model ($K=512$) using bi-modal data (Clinical + Gene):
```bash
python train.py --model hsic --k 512 --modalities clinical gene --seed 42
```

### Running Baseline Comparisons
To run the clinical-only or naive concatenation baselines:
```bash
python train.py --model clinical_mlp --seed 42
python train.py --model late_fusion --modalities clinical gene cna
```

### Evaluation & Calibration
Evaluate the trained model to generate C-index and calibration metrics (Brier, ICI) at 5y/10y horizons:
```bash
python evaluate.py --checkpoint weights/hsic_k512_seed42.pth
```

## 🤝 Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## 📝 Citation
If you find this code or our concepts useful in your research, please consider citing our work:
```bibtex
@article{Liu2026BreastCancerAI,
  title={An HSIC-based Multimodal Fusion Framework on METABRIC in Breast-Cancer Survival Prediction},
  author={Liu, Limei and others},
  journal={Journal Name},
  year={2026}
}
```

## ✉️ Contact
For any questions regarding the code or paper, please contact:
**Limei Liu** - limeiliu.it@gmail.com;limei.liu@monash.edu
