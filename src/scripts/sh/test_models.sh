#!/bin/bash
DATASET="sst"
DATA_PATH="data/sst/all_data"
OUTPUT_PATH="output/NLP"
D_MODEL=32
DFF=32
RNN_UNITS=32
BS=32
PARTICLES=10
EP=0

python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "lstm" -d_model $D_MODEL -rnn_units $RNN_UNITS -bs $BS -ep $EP -output_path $OUTPUT_PATH -save_path "output/NLP/lstm_d32_p0.0/1" -test_samples 30

#python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc True -particles $PARTICLES -sigmas 0.05 -save_path "output/NLP/smc_t_l1_h1_d32_10p_sigmas0.05/1"

#python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc True -particles $PARTICLES -sigmas 0.1 -save_path "output/NLP/smc_t_l1_h1_d32_10p_sigmas0.1/1"

#python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc True -particles $PARTICLES -sigmas 0.5 -save_path "output/NLP/smc_t_l1_h1_d32_10p_sigmas0.5/1"

python src/scripts/run.py -dataset $DATASET -data_path $DATA_PATH -algo "smc_t" -d_model $D_MODEL -dff $DFF -bs $BS -ep $EP -output_path $OUTPUT_PATH -smc False -particles 1 -save_path "output/NLP/smc_t_l1_h1_d32_1p_sigmas0.5/1" -test_samples 30