#!/usr/bin/env bash
python src/scripts/run.py -dataset "covid" -dataset_model 1 -data_path "data/covid" -algo "bayesian_lstm" -rnn_units 32 -bs 32 -ep 150 -output_path "output/exp_covid" -mc_samples 1000 -particles 3
python src/scripts/run.py -dataset "air_quality" -dataset_model 1 -data_path "data/air_quality" -algo "bayesian_lstm" -rnn_units 32 -bs 64 -ep 150 -output_path "output/exp_air_quality" -mc_samples 1000 -particles 3
python src/scripts/run.py -dataset "weather" -dataset_model 1 -data_path "data/weather" -algo "bayesian_lstm" -rnn_units 32 -bs 256 -ep 0 -output_path "output/exp_weather" -mc_samples 1000 -particles 3 -max_samples 50000 -save_path "output/exp_weather/bayesian_lstm_d32/1" -past_len 12 -multistep 1
python src/scripts/run.py -dataset "energy" -dataset_model 1 -data_path "data/energy" -algo "bayesian_lstm" -rnn_units 32 -bs 128 -ep 150 -output_path "output/exp_energy" -mc_samples 1000 -particles 3 -multistep 1 -past_len 6
