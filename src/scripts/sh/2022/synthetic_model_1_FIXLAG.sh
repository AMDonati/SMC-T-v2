#!/bin/bash
#SBATCH --job-name=fix-lag-smct
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/fixlag-synthetic-%j.out
#SBATCH --error=slurm_out/fixlag-synthetic-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

#srun python -u src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 4
#srun python -u src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 8
#srun python -u src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 12
#srun python -u src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 16
srun python -u src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 20
srun python -u src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 24


