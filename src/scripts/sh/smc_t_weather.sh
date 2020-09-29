#!/usr/bin/env bash
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -smc False
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -particles 10 -smc True
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -particles 30 -smc True -multistep 1 -past_len 12