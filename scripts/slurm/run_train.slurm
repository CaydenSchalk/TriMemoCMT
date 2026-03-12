#!/bin/bash
#SBATCH --job-name=trimemocmt
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgok-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=/projects/bgok/cschalk/Quals/logs/%x-%j.out
#SBATCH --error=/projects/bgok/cschalk/Quals/logs/%x-%j.err
#SBATCH --mail-user=chs3u@mtmail.mtsu.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

mkdir -p /projects/bgok/cschalk/Quals/logs
mkdir -p /projects/bgok/cschalk/Quals/checkpoints

cd ~/Quals
source ~/.bashrc
conda activate TriMemoCMT

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

cd ~/Quals/scripts

python train.py \
  --config ../src/configs/mae_base.py \
  --raw_root /projects/bgok/cschalk/Quals/data/raw \
  --processed_root /projects/bgok/cschalk/Quals/data/processed \
  --checkpoint_root /projects/bgok/cschalk/Quals/checkpoints