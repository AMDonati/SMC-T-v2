#!/bin/bash
#conda activate smc-t

export PYTHONPATH=src:${PYTHONPATH}

#python  src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG_ESS/cv/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 4 -cv 1 -ess 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG_ESS/cv/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 8 -cv 1 -ess 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG_ESS/cv/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 12 -cv 1 -ess 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG_ESS/cv/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 16 -cv 1 -ess 1
python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "data/synthetic_model_1" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 50 -particles 10 -smc True -output_path "output/FIX_LAG_ESS/cv/exp_synthetic_model_1" -alpha 0.8 -mc_samples 1000 -fix_lag 20 -cv 1 -ess 1


