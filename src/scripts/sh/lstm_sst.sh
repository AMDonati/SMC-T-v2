#!/bin/bash
#SBATCH --job-name=smc-t-None
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/lstm-%j.out
#SBATCH --error=slurm_out/sst/lstm-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

DATASET="sst"
DATA_PATH="data/sst/all_data"
OUTPUT_PATH="output/NLP"
D_MODEL=32
RNN_UNITS=32
BS=32
PARTICLES=1
EP=50

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "lstm" -d_model $D_MODEL -rnn_units $RNN_UNITS -bs $BS -ep $EP -output_path $OUTPUT_PATH
