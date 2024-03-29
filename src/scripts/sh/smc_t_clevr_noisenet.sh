#!/bin/bash
#SBATCH --job-name=CLEVR-noisenet
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --array=1-4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/CLEVR-noisenet-%j.out
#SBATCH --error=slurm_out/sst/CLEVR-noisenet-%j.err
#SBATCH --time=100:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

DATASET="clevr"
DATA_PATH="data/clevr"
OUTPUT_PATH="output/NLP/CLEVR/NEW_EXP/noise_net"
D_MODEL=32
DFF=32
BS=32
PARTICLES=10
EP=20

set -x
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
OUT_PATH=output/NLP/CLEVR/NEW_EXP/noise_net/${SLURM_ARRAY_TASK_ID}

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path ${OUT_PATH} -smc True -particles $PARTICLES -sigmas 0.1 -max_seq_len 20 -full_model True

#srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep 40 -output_path $OUTPUT_PATH -smc True -particles $PARTICLES -sigmas 0.1 -max_seq_len 20 -full_model True
