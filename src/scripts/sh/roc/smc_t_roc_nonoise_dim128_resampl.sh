#!/bin/bash
#SBATCH --job-name=dim128-ROC-smc-t-None
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/ROC-smc_t-None-%j.out
#SBATCH --error=slurm_out/sst/ROC-smc_t-None-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

DATASET="roc"
DATA_PATH="data/ROC"
OUTPUT_PATH="output/NLP/ROC_Feb2022"
D_MODEL=128
DFF=128
BS=32
PARTICLES=1
EP=40

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc False -particles $PARTICLES -max_seq_len 20 -temp 0.7 -inference_resample 1
