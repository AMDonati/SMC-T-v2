#!/usr/bin/env bash
echo "-------- synthetic model 1 ---------------------------------------------"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 8 -dff 8 -bs 32 -ep 1 -output_path "output/temp" -p_drop 0.1 -inference 1 -mc_samples 5
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 8 -p_drop 0.1 -bs 32 -ep 1 -output_path "output/temp" -inference 1 -mc_samples 5
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 8 -dff 8 -bs 100 -ep 1 -particles 2 -smc True -output_path "output/temp" -num_layers 2
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "bayesian_lstm" -rnn_units 8 -bs 32 -ep 1 -output_path "output/temp" -mc_samples 5 -particles 3
echo "-------- covid ---------------------------------------------"
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 8 -dff 32 -bs 32 -pe 100 -ep 1 -output_path "output/temp" -p_drop 0.5 -mc_samples 5 -multistep 1 -past_len 40
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "lstm" -rnn_units 8 -bs 32 -ep 1 -output_path "output/temp" -p_drop 0.1 -mc_samples 5
python src/scripts/run.py -dataset "covid" -dataset_model 1 -data_path "data/covid" -algo "bayesian_lstm" -rnn_units 8 -bs 32 -ep 1 -output_path "output/temp" -mc_samples 5 -particles 3
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 8 -dff 8 -num_heads 4 -bs 64 -ep 1 -output_path "output/temp" -particles 2 -smc True -multistep 1 -past_len 40 -mc_samples 5 -num_layers 2