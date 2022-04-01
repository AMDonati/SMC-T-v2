#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/preprocess-%j.out
#SBATCH --error=slurm_out/preprocess-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

srun python -u src/preprocessing/create_synthetic_dataset.py -model 1 -num_samples 1000 -seq_len 24 -num_features 1


