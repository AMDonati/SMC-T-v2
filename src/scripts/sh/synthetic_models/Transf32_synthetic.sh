#!/usr/bin/env bash
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.1  -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.2  -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_1" -p_drop 0.5  -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "lstm" -rnn_units 32 -bs 32 -ep 0 -output_path "exp_synthetic_model_1" -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_2" -p_drop 0.1  -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_2" -p_drop 0.2  -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_synthetic_model_2" -p_drop 0.5  -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "lstm" -rnn_units 32 -bs 32 -ep 0 -output_path "exp_synthetic_model_2" -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -smc False -output_path "output/exp_synthetic_model_1" -cv 1 -alpha 0.8
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -particles 10 -smc True -output_path "output/exp_synthetic_model_1" -cv 1 -alpha 0.8 -mc_samples 1000
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -particles 30 -smc True -output_path "output/exp_synthetic_model_1" -cv 1 -alpha 0.8 -mc_samples 1000



