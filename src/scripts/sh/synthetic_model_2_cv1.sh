#!/usr/bin/env bash
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -smc False -output_path "output/exp_synthetic_model_2" -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/exp_synthetic_model_2" -cv 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 30 -smc True -output_path "output/exp_synthetic_model_2" -cv 1
