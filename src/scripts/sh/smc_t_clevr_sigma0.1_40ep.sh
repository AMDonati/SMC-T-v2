#!/bin/bash
#SBATCH --job-name=40ep-CLEVR-smc-t0.1
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --array=1-4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/CLEVR-smc_t0.1-40ep-%j.out
#SBATCH --error=slurm_out/sst/CLEVR-smc_t0.1-40ep-%j.err
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
PARTICLES=10
EP=40

set -x
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
OUT_PATH=output/NLP/CLEVR/NEW_EXP/${SLURM_ARRAY_TASK_ID}

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path ${OUT_PATH} -smc True -particles $PARTICLES -sigmas 0.1 -max_seq_len 20 -full_model True
