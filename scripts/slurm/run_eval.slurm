#!/bin/bash
#SBATCH --job-name=trimemocmt_eval
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgok-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --output=/projects/bgok/cschalk/Quals/logs/%x-%j.out
#SBATCH --error=/projects/bgok/cschalk/Quals/logs/%x-%j.err
#SBATCH --mail-user=chs3u@mtmail.mtsu.edu
#SBATCH --mail-type=END,FAIL

set -eo pipefail

mkdir -p /projects/bgok/cschalk/Quals/logs

cd ~/Quals
export BASHRCSOURCED=1
source ~/.bashrc
conda activate TriMemoCMT

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

cd ~/Quals/scripts

python -u eval.py \
  --checkpoint_path /projects/bgok/cschalk/Quals/checkpoints/TriMemoCMT_bert_hubert_base_VideoMAEv2/20260312-040619 \
  --test_set test.pkl \
  --confusion_matrix