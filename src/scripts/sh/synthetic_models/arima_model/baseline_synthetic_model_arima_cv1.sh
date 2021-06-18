#!/usr/bin/env bash
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 50 -output_path "output/exp_arima_model/cv" -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 50 -output_path "output/exp_arima_model/cv" -p_drop 0.1 -inference 1 -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 50 -output_path "output/exp_arima_model/cv" -p_drop 0.2 -inference 1 -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -ep 50 -output_path "output/exp_arima_model/cv" -p_drop 0.5 -inference 1 -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "lstm" -rnn_units 32 -bs 32 -ep 50 -output_path "output/exp_arima_model/cv" -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "lstm" -rnn_units 32 -p_drop 0.1 -bs 32 -ep 50 -output_path "output/exp_arima_model/cv" -inference 1 -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "lstm" -rnn_units 32 -p_drop 0.2 -bs 32 -ep 50 -output_path "output/exp_arima_model/cv" -inference 1 -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "lstm" -rnn_units 32 -p_drop 0.5 -bs 32 -ep 50 -output_path "output/exp_arima_model/cv" -inference 1 -mc_samples 1000 -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "bayesian_lstm" -rnn_units 32 -bs 32 -ep 150 -output_path "output/exp_arima_model/cv" -mc_samples 1000 -particles 3 -cv 1