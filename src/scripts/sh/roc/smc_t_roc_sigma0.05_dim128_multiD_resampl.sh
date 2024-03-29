#!/bin/bash
#SBATCH --job-name=multiD0.05-resampl-40ep-dim128-ROC-CLEVR-smc-t0.1
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_out/sst/ROC-smc_t0.05-multi%j.out
#SBATCH --error=slurm_out/sst/ROC-smc_t0.05-multi-%j.err
#SBATCH --time=100:00:00
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
PARTICLES=10
EP=40

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc True -particles $PARTICLES -sigmas 0.05 -max_seq_len 20 -full_model True -noise_dim "multi" -temp 0.7 -inference_resample 1