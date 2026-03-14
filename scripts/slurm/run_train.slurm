#!/bin/bash
#SBATCH --job-name=temporal-trimemocmt
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgok-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bgok/cschalk/temporal/logs/%x-%j.out
#SBATCH --error=/projects/bgok/cschalk/temporal/logs/%x-%j.err
#SBATCH --mail-user=chs3u@mtmail.mtsu.edu
#SBATCH --mail-type=END,FAIL

set -eo pipefail

mkdir -p /projects/bgok/cschalk/temporal/logs
mkdir -p /projects/bgok/cschalk/temporal/checkpoints

cd ~/Quals

export BASHRCSOURCED=1
source ~/.bashrc
conda activate TriMemoCMT

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi

cd ~/temporal/scripts

python train.py \
  --config ../src/configs/mae_base.py \
  --raw_root /projects/bgok/cschalk/Quals/data/raw \
  --processed_root /projects/bgok/cschalk/Quals/data/processed_batched \
  --checkpoint_root /projects/bgok/cschalk/Quals/checkpoints