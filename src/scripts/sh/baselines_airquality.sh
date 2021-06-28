#!/usr/bin/env bash
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 16 -dff 16 -bs 64 -ep 0 -output_path "output/exp_airquality"
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 16 -dff 16 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.1 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 16 -dff 16 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.2 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 16 -dff 16 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.5 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_airquality"
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.1 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.2 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.5 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_airquality"
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.1 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.2 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.5 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 64 -bs 64 -ep 0 -output_path "output/exp_airquality"
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 64 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.1 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 64 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.2 -inference 1 -mc_samples 1000
python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "lstm" -rnn_units 64 -bs 64 -ep 0 -output_path "output/exp_airquality" -p_drop 0.5 -inference 1 -mc_samples 1000