#!/bin/bash
#SBATCH --job-name=30MAXSEQ-lstm
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

DATASET="roc"
DATA_PATH="data/ROC"
OUTPUT_PATH="output/NLP/ROC/LSTM"
D_MODEL=128
RNN_UNITS=128
BS=32
PARTICLES=1
EP=20

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "lstm" -d_model $D_MODEL -rnn_units $RNN_UNITS -bs $BS -ep $EP -output_path $OUTPUT_PATH -max_seq_len 20 -temp 0.7

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "lstm" -d_model $D_MODEL -rnn_units $RNN_UNITS -bs $BS -ep $EP -output_path $OUTPUT_PATH -max_seq_len 20 -pdrop 0.1 -temp 0.7

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "lstm" -d_model $D_MODEL -rnn_units $RNN_UNITS -bs $BS -ep $EP -output_path $OUTPUT_PATH -max_seq_len 20 -pdrop 0.5 -temp 0.7



