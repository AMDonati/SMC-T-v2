#!/bin/bash
#SBATCH --job-name=dec-sst
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/dec-sst-%j.out
#SBATCH --error=slurm_out/sst/dec-sst-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

DATASET="sst"
DATA_PATH="data/sst/all_data"
OUTPUT_PATH="output/DEBUG"
D_MODEL=32
DFF=32
BS=32

#python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep 1 -output_path $OUTPUT_PATH -smc True -particles 10
python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep 1 -output_path $OUTPUT_PATH -smc False -particles 1