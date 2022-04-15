#!/bin/bash
#SBATCH --job-name=TIMER
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/TIMER-weather-%j.out
#SBATCH --error=slurm_out/TIMER-weather-%j.err
#SBATCH --time=100:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

# attentionwindow = 30
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess True -cv 0 -attn_w 30 -multistep 0 -past_len 120 -max_samples 10000
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 0 -cv 0 -attn_w 30 -multistep 0 -past_len 120 -max_samples 10000
# attentionwindow=30, fix_lag=30.
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess True -cv 0 -attn_w 30 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 30
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 0 -cv 0 -attn_w 30 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 30
# fixlag = 30
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess True -cv 0 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 30
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 0 -cv 0 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 30
# no lag, no attention window
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess True -cv 0 -multistep 0 -past_len 120 -max_samples 10000
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 0 -cv 0 -multistep 0 -past_len 120 -max_samples 10000
# Baselines
python src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 1 -output_path "output/WEATHER_LONG/TIMER_3" -p_drop 0.1 -mc_samples 1000 -max_samples 10000 -past_len 120 -multistep 1 -pe 500
# attentionwindow = 120
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess True -cv 0 -attn_w 120 -multistep 0 -past_len 120 -max_samples 10000
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess 0 -cv 0 -attn_w 120 -multistep 0 -past_len 120 -max_samples 10000
# attentionwindow=60, fix_lag=60, ESS.
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess True -cv 0 -attn_w 60 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 60
# fixlag 60, ESS
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess True -cv 0 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 60
# fixlag 120, ESS
srun python -u src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 1 -particles 10 -smc True -output_path "output/WEATHER_LONG/TIMER_3" -alpha 0.8 -mc_samples 5 -ess True -cv 0 -multistep 0 -past_len 120 -max_samples 10000 -fix_lag 120





