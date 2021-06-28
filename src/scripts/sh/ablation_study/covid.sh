#!/usr/bin/env bash
echo "------------ no full model-------------------------------"
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/no_full_model/no_smc" -full_model False -particles 1 -smc False -multistep 1 -past_len 40 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/no_full_model" -full_model False -particles 1 -smc True -multistep 1 -past_len 40 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/no_full_model" -full_model False -particles 10 -smc True -multistep 1 -past_len 40 -mc_samples 1000
echo "------------ full model ----------------------------------"
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/full_model/no_smc" -particles 1 -smc False -multistep 1 -past_len 40 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/full_model" -particles 1 -smc True -multistep 1 -past_len 40 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/full_model" -particles 10 -smc True -multistep 1 -past_len 40 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/full_model" -particles 10 -smc True -multistep 1 -past_len 40 -mc_samples 1000 -num_heads 4
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/full_model" -particles 10 -smc True -multistep 1 -past_len 40 -mc_samples 1000 -num_heads 4 -num_layers 2
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "smc_t" -d_model 32 -dff 32 -bs 32 -ep 0 -output_path "output/ablation_study/exp_covid/full_model" -particles 30 -smc True -multistep 1 -past_len 40 -mc_samples 1000 -num_heads 4 -num_layers 2

