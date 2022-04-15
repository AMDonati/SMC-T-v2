#!/bin/bash
#SBATCH --job-name=lag3.30w-weather-attn-w-smct
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/attn-fixlag3-30w-%j.out
#SBATCH --error=slurm_out/attn-fixlag3-30w-%j.err
#SBATCH --time=100:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}


srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 20 -particles 10 -smc True -output_path "output/WEATHER_LONG/fix_lag" -alpha 0.8 -mc_samples 1000 -ess 0 -cv 0 -attn_w 30 -multistep 1 -past_len 120 -max_samples 10000 -fix_lag 3
