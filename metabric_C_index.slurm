#!/bin/bash
#SBATCH --job-name=metabric_cidx
#SBATCH --output=logs/metabric_cindex.out
#SBATCH --error=logs/metabric_cindex.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --account=ek04
#SBATCH --qos=normal

set -euo pipefail
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

source ~/ek04_scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/limeil/ek04_scratch/miniconda3/envs/torch-med

# Cox proportional-hazards survival prediction (DeepSurv-style) with C-index.
# Reads raw cBioPortal files directly (keeps early-censored samples vs the
# AUROC variant which drops them). Single GPU is plenty for METABRIC.
python main_C-index.py --raw_dir ./raw
