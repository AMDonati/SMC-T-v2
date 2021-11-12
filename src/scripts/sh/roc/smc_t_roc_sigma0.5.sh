#!/bin/bash
#SBATCH --job-name=ROC-CLEVR-smc-t0.5
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/ROC-smc_t0.5-%j.out
#SBATCH --error=slurm_out/sst/ROC-smc_t0.5-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

DATASET="clevr"
DATA_PATH="data/clevr"
OUTPUT_PATH="output/NLP/ROC"
D_MODEL=32
DFF=32
BS=32
PARTICLES=10
EP=20

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc True -particles $PARTICLES -sigmas 0.5 -max_seq_len 20 -full_model True