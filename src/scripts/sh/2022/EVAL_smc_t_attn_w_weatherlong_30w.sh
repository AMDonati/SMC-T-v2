#!/bin/bash
#SBATCH --job-name=EVAL30weather-attn-w-smct
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/EVAL-attn-w-30w-%j.out
#SBATCH --error=slurm_out/EVAL-attn-w-30w-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}


srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 0 -particles 10 -smc True -output_path "output/WEATHER_LONG/attn_window" -alpha 0.8 -mc_samples 1000 -ess 0 -cv 0 -attn_w 30 -multistep 1 -past_len 120 -max_samples 10000 -save_path "output/WEATHER_LONG/attn_window/w30/1" -fix_lag 144
