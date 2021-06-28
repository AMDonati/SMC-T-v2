#!/usr/bin/env bash
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -smc False -output_path "output/exp_arima_model"
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -particles 10 -smc True -output_path "output/exp_arima_model"
python src/scripts/run.py -dataset "synthetic" -dataset_model 3 -data_path "data/arima_model" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -particles 30 -smc True -output_path "output/exp_arima_model"
