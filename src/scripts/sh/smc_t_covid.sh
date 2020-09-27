#!/usr/bin/env bash
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 8 -dff 16 -bs 32 -ep 50 -output_path "output/exp_covid" -smc False
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 8 -dff 16 -bs 32 -ep 50 -output_path "output/exp_covid" -particles 10 -smc True