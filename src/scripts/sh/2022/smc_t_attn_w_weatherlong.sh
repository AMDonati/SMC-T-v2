#!/bin/bash
#SBATCH --job-name=1weather-attn-w-smct
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/attn-w-synthetic-%j.out
#SBATCH --error=slurm_out/attn-w-synthetic-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -particles 10 -smc True -output_path "output/WEATHER_LONG/attn_window" -alpha 0.8 -mc_samples 1000 -ess 0 -cv 0 -attn_w 10 -multistep 1 -past_len 120
srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -particles 10 -smc True -output_path "output/WEATHER_LONG/attn_window" -alpha 0.8 -mc_samples 1000 -ess True -cv 0 -attn_w 10 -multistep 1 -past_len 120
srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -particles 10 -smc True -output_path "output/WEATHER_LONG/attn_window" -alpha 0.8 -mc_samples 1000 -ess 0 -cv 0 -attn_w 25 -multistep 1 -past_len 120
srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -particles 10 -smc True -output_path "output/WEATHER_LONG/attn_window" -alpha 0.8 -mc_samples 1000 -ess True -cv 0 -attn_w 25 -multistep 1 -past_len 120
#python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -particles 10 -smc True -mc_samples 1000 -past_len 12 -multistep 1 -max_samples 50000