#!/bin/bash
DATASET="sst"
DATA_PATH="data/sst/all_data"
OUTPUT_PATH="output/timer"
D_MODEL=8
DFF=8
BS=32
EP=1
MAX_SEQ_LEN=20

python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -max_seq_len $MAX_SEQ_LEN -ep $EP -output_path $OUTPUT_PATH -smc False -particles 1 > "output/timer/out_nosmc_logvar_1p_bs32_d8_20seq.txt"
python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -max_seq_len $MAX_SEQ_LEN -ep $EP -output_path $OUTPUT_PATH -smc True -particles 1 > "output/timer/out_smc_logvar_1p_bs32_d8_20seq.txt"
python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -max_seq_len $MAX_SEQ_LEN -ep $EP -output_path $OUTPUT_PATH -smc True -particles 10 > "output/timer/out_smc_logvar_10p_bs32_d8_20seq.txt"