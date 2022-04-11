#!/usr/bin/env bash
#SBATCH --job-name=BASELINES-w
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/BASELINES-w-%j.out
#SBATCH --error=slurm_out/BASELINES-w-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

python src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 20 -output_path "output/WEATHER_LONG/BASELINES" -max_samples 10000 -p_drop 0.
python src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 20 -output_path "output/WEATHER_LONG/BASELINES" -p_drop 0.1 -mc_samples 1000 -max_samples 10000 -past_len 120 -multistep 1
python src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 20 -output_path "output/WEATHER_LONG/BASELINES" -p_drop 0.2 -mc_samples 1000 -max_samples 10000 -past_len 120 -multistep 1
python src/scripts/run.py -dataset "weather_long" -data_path "data/weather_long" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 20 -output_path "output/WEATHER_LONG/BASELINES" -p_drop 0.5 -mc_samples 1000 -max_samples 10000 -past_len 120 -multistep 1