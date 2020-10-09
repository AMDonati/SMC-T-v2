#!/usr/bin/env bash
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.1 -bs 32 -ep 50 -output_path "exp_synthetic_model_1_plots" -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.2 -bs 32 -ep 50 -output_path "exp_synthetic_model_1_plots" -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.5 -bs 32 -ep 50 -output_path "exp_synthetic_model_1_plots" -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "bayesian_lstm" -rnn_units 32 -bs 32 -ep 150 -output_path "output/exp_synthetic_model_1_plots" -mc_samples 1000 -particles 10
