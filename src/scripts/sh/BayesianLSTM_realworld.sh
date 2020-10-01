#!/usr/bin/env bash
python src/scripts/run.py -dataset "covid" -dataset_model 1 -data_path "data/covid" -algo "bayesian_lstm" -rnn_units 32 -bs 32 -ep 150 -output_path "output/exp_covid" -mc_samples 1000 -particles 3
python src/scripts/run.py -dataset "air_quality" -dataset_model 1 -data_path "data/air_quality" -algo "bayesian_lstm" -rnn_units 32 -bs 64 -ep 150 -output_path "output/exp_air_quality" -mc_samples 1000 -particles 3
python src/scripts/run.py -dataset "weather" -dataset_model 1 -data_path "data/weather" -algo "bayesian_lstm" -rnn_units 32 -bs 256 -ep 150 -output_path "output/exp_weather" -mc_samples 1000 -particles 3
python src/scripts/run.py -dataset "weather" -dataset_model 1 -data_path "data/weather" -algo "bayesian_lstm" -rnn_units 32 -bs 256 -ep 1 -output_path "output/temp" -mc_samples 1000 -particles 3
