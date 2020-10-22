#!/usr/bin/env bash
python src/scripts/run.py -dataset "energy" -data_path "data/energy" -algo "smc_t" -d_model 32 -dff 32 -bs 128 -ep 50 -output_path "output/exp_energy" -smc False
python src/scripts/run.py -dataset "energy" -data_path "data/energy" -algo "smc_t" -d_model 32 -dff 32 -bs 128 -ep 50 -output_path "output/exp_energy" -particles 10 -smc True -multistep 1 -past_len 6 -mc_samples 1000
python src/scripts/run.py -dataset "energy" -data_path "data/energy" -algo "smc_t" -d_model 32 -dff 32 -bs 128 -ep 50 -output_path "output/exp_energy" -particles 30 -smc True -multistep 1 -past_len 6 -mc_samples 1000
python src/scripts/run.py -dataset "energy" -data_path "data/energy" -algo "smc_t" -d_model 32 -dff 32 -bs 128 -ep 50 -output_path "output/exp_energy" -particles 60 -smc True -multistep 1 -past_len 6 -mc_samples 1000 -save_path "output/exp_energy/smc_t_d32_60p/1"

python src/scripts/run.py -dataset "energy" -data_path "data/energy" -algo "smc_t" -d_model 32 -dff 32 -bs 128 -ep 0 -output_path "output/exp_energy" -particles 10 -smc True -multistep 1 -past_len 6 -mc_samples 1000 -save_path "output/exp_energy/smc_t_d32_p10/1"