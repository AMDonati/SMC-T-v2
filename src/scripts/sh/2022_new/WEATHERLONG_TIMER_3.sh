#!/bin/bash
#SBATCH --job-name=2TIMER
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/TIMER-weather-%j.out
#SBATCH --error=slurm_out/TIMER-weather-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

# no lag, no attention window
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 0 -cv 0 -multistep 0 -past_len 120 -max_samples 10000
# fixlag = 3, attn_w = 3
srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 0 -cv 0 -attn_w 30 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 3
srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 1 -cv 0 -attn_w 30 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 3
# fixlag = 5, no attn_window.
srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 0 -cv 0 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 5
srun python -u src/scripts/run.py -dataset "weather_long" -dataset_model 1 -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 1 -cv 0 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 5





