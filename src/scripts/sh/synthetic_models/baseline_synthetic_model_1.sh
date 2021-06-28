#!/usr/bin/env bash
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.1 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.2 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.5 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -bs 32 -ep 0 -output_path "exp_synthetic_model_1"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.1 -bs 32 -ep 0 -output_path "exp_synthetic_model_1" -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.2 -bs 32 -ep 0 -output_path "exp_synthetic_model_1" -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.5 -bs 32 -ep 0 -output_path "exp_synthetic_model_1" -inference 1 -mc_samples 1000
