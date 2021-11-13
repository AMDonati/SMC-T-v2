#!/bin/bash
#SBATCH --job-name=CLEVR-smc-t-None
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --array=1-4
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_out/sst/smc_t-None-%j.out
#SBATCH --error=slurm_out/sst/smc_t-None-%j.err
#SBATCH --time=20:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

DATASET="clevr"
DATA_PATH="data/clevr"
OUTPUT_PATH="output/NLP/CLEVR/NEW_EXP"
D_MODEL=32
DFF=32
BS=32
PARTICLES=1
EP=20

set -x
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
OUT_PATH=output/NLP/CLEVR/NEW_EXP/no_noise/${SLURM_ARRAY_TASK_ID}

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path ${OUT_PATH} -smc False -particles $PARTICLES -max_seq_len 20
