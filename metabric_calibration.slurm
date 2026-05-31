#!/bin/bash
#SBATCH --job-name=metabric_calib
#SBATCH --output=logs/metabric_calibration.out
#SBATCH --error=logs/metabric_calibration.err
#SBATCH --partition=comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --account=ek04
#SBATCH --qos=normal

set -euo pipefail
mkdir -p logs/calibration

source ~/ek04_scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/limeil/ek04_scratch/miniconda3/envs/torch-med

# Post-hoc calibration analysis (CPU only).
# Requires: ablation_results_cindex.pkl already produced by main_C-index.py
# AFTER patching main_C-index.py to dump per-seed train/test (risks, times, events).
#
# Outputs land in logs/calibration/:
#   - calib_<model>.png            : per-model 5y + 10y panel
#   - calib_compare_{60,120}m.png  : smoothed-curve overlay across models
#   - calibration_summary.csv      : Brier / ICI / E50 / E90, mean ± std

python make_calibration.py \
    --pkl       processed_metabric/ablation_results_cindex.pkl \
    --out_dir   logs/calibration \
    --models    "Clinical-only (MLP)" \
                "Late Fusion (Concat)" \
                "Late Fusion 3 (Concat)" \
                "Ours (HSIC K=512)" \
                "Ours-Tri (HSIC K=512)" \
    --horizons  60 120
