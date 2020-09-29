#!/usr/bin/env bash
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 16 -dff 16 -bs 32 -ep 0 -particles 30 -smc True -output_path "output/exp_synthetic_model_1" -save_path "output/exp_synthetic_model_1/smc_t/smc_t_d16_p30/1"  -mc_samples 1000
