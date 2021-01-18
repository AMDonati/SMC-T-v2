#!/usr/bin/env bash
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 8 -dff 8 -bs 32 -ep 50 -smc False -output_path "output/exp_synthetic_model_2"
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 16 -dff 16 -bs 32 -ep 50 -smc False -output_path "output/exp_synthetic_model_2"
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 8 -dff 8 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/exp_synthetic_model_2"
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 8 -dff 8 -bs 32 -ep 50 -particles 30 -smc True -output_path "output/exp_synthetic_model_2"
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 16 -dff 16 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/exp_synthetic_model_2"
python src/scripts/run.py -dataset "synthetic" -dataset_model 2 -data_path "data/synthetic_model_2" -algo "smc_t" -d_model 16 -dff 16 -bs 32 -ep 50 -particles 30 -smc True -output_path "output/exp_synthetic_model_2"
