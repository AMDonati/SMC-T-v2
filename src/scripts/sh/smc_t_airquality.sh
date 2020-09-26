#!/usr/bin/env bash
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "smc_t" -d_model 16 -dff 16 -bs 64 -ep 50 -output_path "output/exp_airquality" -smc False
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 50 -output_path "output/exp_airquality" -smc False
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "smc_t" -d_model 16 -dff 16 -bs 64 -ep 50 -output_path "output/exp_airquality" -particles 10 -smc True
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "smc_t" -d_model 32 -dff 32 -bs 64 -ep 50 -output_path "output/exp_airquality" -particles 10 -smc True