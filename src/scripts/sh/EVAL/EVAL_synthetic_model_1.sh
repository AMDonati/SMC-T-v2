#!/usr/bin/env bash
#python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 16 -dff 16 -bs 32 -ep 0 -particles 30 -smc True -output_path "output/exp_synthetic_model_1" -save_path "output/exp_synthetic_model_1/smc_t/smc_t_d16_p30/1"  -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 16 -dff 16 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.1 -mc_samples 1000 -save_path "output/exp_synthetic_model_1/baseline_t/baseline_t_d16_p0.1/1"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 16 -dff 16 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.2 -mc_samples 1000 -save_path "output/exp_synthetic_model_1/baseline_t/baseline_t_d16_p0.2/1"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 16 -dff 16 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.5 -mc_samples 1000 -save_path "output/exp_synthetic_model_1/baseline_t/baseline_t_d16_p0.5/1"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.1 -bs 32 -ep 0 -output_path "exp_synthetic_model_1" -mc_samples 1000 -save_path "output/exp_synthetic_model_1/lstm/lstm_d32_p0.1/1"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.2 -bs 32 -ep 0 -output_path "exp_synthetic_model_1" -mc_samples 1000 -save_path "output/exp_synthetic_model_1/lstm/lstm_d32_p0.2/1"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.5 -bs 32 -ep 0 -output_path "exp_synthetic_model_1" -mc_samples 1000 -save_path "output/exp_synthetic_model_1/lstm/lstm_d32_p0.5/1"

python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -particles 30 -smc True -output_path "output/exp_synthetic_model_1" -cv 1 -alpha 0.8 -mc_samples 1000 -save_path "output/exp_synthetic_model_1/smc_t_d32_30p/1"
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -particles 10 -smc True -output_path "output/exp_synthetic_model_1" -cv 1 -alpha 0.8 -mc_samples 1000 -save_path "output/exp_synthetic_model_1/smc_t_d32_10p/1"