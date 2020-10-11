#!/usr/bin/env bash
#python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "baseline_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather"
#python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "baseline_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -p_drop 0.1 -mc_samples 1000
#python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "baseline_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -p_drop 0.2 -mc_samples 1000
#python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "baseline_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -p_drop 0.5 -mc_samples 1000
#python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "lstm" -rnn_units 32 -bs 256 -ep 50 -output_path "output/exp_weather"
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "lstm" -rnn_units 32 -bs 256 -ep 50 -output_path "output/exp_weather_lstm" -p_drop 0.1 -mc_samples 1000 -multistep 1 -past_len 12
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "lstm" -rnn_units 32 -bs 256 -ep 50 -output_path "output/exp_weather_lstm" -p_drop 0.2 -mc_samples 1000 -multistep 1 -past_len 12
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "lstm" -rnn_units 32 -bs 256 -ep 50 -output_path "output/exp_weather_lstm" -p_drop 0.5 -mc_samples 1000 -multistep 1 -past_len 12