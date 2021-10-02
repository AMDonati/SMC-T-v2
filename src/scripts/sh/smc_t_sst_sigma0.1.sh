#!/bin/bash
#SBATCH --job-name=smc-t0.1
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/sst/smc_t0.1-%j.out
#SBATCH --error=slurm_out/sst/smc_t0.1-%j.err
#SBATCH --time=100:00:00
#SBATCH -A ktz@gpu

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

DATASET="sst"
DATA_PATH="data/sst/all_data"
OUTPUT_PATH="output/NLP/vocab2/maxlen30"
D_MODEL=32
DFF=32
BS=32
PARTICLES=10
EP=50

srun python -u src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc True -particles $PARTICLES -sigmas 0.1 -max_seq_len 30 -min_token_count 2 -test_samples 30
