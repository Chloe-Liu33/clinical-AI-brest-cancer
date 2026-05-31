#!/bin/bash
#SBATCH --job-name=metabric
#SBATCH --output=logs/metabric_auroc.out
#SBATCH --error=logs/metabric_auroc.err
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

# Single GPU is enough for METABRIC (~1300 train samples).
# To use 2 GPUs: set --gres=gpu:L40S:2 above and:
#   torchrun --standalone --nproc_per_node=2 main.py
python main_auroc.py --processed_dir ./processed_metabric
