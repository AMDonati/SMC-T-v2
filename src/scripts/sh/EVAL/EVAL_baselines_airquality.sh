#!/usr/bin/env bash
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.1 -mc_samples 1000 -save_path "output/exp_airquality/baseline_t/baseline_t_d32_p0.1/1" -multistep 1 -past_len 6
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.2 -mc_samples 1000 -save_path "output/exp_airquality/baseline_t/baseline_t_d32_p0.2/1" -multistep 1 -past_len 6
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.5 -mc_samples 1000 -save_path "output/exp_airquality/baseline_t/baseline_t_d32_p0.5/1" -multistep 1 -past_len 6
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.1 -mc_samples 1000 -save_path "output/exp_airquality/lstm/lstm_d32_p0.1/1" -multistep 1 -past_len 6
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.2 -mc_samples 1000 -save_path "output/exp_airquality/lstm/lstm_d32_p0.2/1" -multistep 1 -past_len 6
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.5 -mc_samples 1000 -save_path "output/exp_airquality/lstm/lstm_d32_p0.5/1" -multistep 1 -past_len 6