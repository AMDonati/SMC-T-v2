#!/bin/bash
#SBATCH --job-name=gpt2dec-ROC-smc-t-sigma0.5
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16g
#SBATCH --output=slurm_out/sst/ROC-smc_t-0.5-%j.out
#SBATCH --error=slurm_out/sst/ROC-smc_t-0.5-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

DATASET="roc"
DATA_PATH="data/ROC"
OUTPUT_PATH="output/NLP/ROC/gpt2dec"
D_MODEL=768
DFF=3072
BS=16
PARTICLES=10
EP=0

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc True -particles $PARTICLES -max_seq_len 20 -num_layers 0 -sigmas 0.5
