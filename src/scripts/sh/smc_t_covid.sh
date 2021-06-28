#!/usr/bin/env bash
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 16 -dff 16 -bs 32 -ep 0 -output_path "output/exp_covid" -smc False
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 16 -dff 16 -bs 32 -ep 0 -output_path "output/exp_covid" -particles 10 -smc True
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_covid" -smc False
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_covid" -particles 10 -smc True
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_covid" -particles 60 -smc True
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_covid" -particles 100 -smc True -multistep 1 -past_len 40 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_covid" -particles 30 -smc True -multistep 1 -past_len 40 -mc_samples 1000