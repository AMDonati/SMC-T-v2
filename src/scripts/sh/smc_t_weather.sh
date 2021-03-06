#!/usr/bin/env bash
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -smc False
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -particles 10 -smc True -save_path "output/exp_weather/weather_Recurrent_T_depth_32_bs_256_fullmodel_True_dff_32_attn_w_None__p_10_SigmaObs_0.5_sigmas_0.5/1"
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -particles 30 -smc True -multistep 1 -past_len 12
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 128 -ep 50 -output_path "output/exp_weather" -particles 60 -smc True -multistep 1 -past_len 12
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 0 -output_path "output/exp_weather" -particles 10 -smc True -save_path "output/exp_weather/smc_t_d32_p10/1" -mc_samples 1000 -past_len 12 -multistep 1

python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -particles 10 -smc True -mc_samples 1000 -past_len 12 -multistep 1 -max_samples 50000 -num_layers 2 -num_heads 4

python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "smc_t" -d_model 32 -dff 32 -bs 256 -ep 50 -output_path "output/exp_weather" -particles 10 -smc True -mc_samples 1000 -past_len 12 -multistep 1 -max_samples 50000