#!/usr/bin/env bash
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_covid" -particles 10 -smc True -save_path "output/exp_covid/smc_t/smc_t_d32_p10/1" -mc_samples 1000 -multistep 1 -past_len 40
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/exp_covid" -particles 60 -smc True -multistep 1 -past_len 40 -save_path "output/exp_covid/smc_t/smc_t_d32_p60/1" -mc_samples 1000 -multistep 1