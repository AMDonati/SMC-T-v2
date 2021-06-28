#!/usr/bin/env bash
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_stock"
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_stock" -p_drop 0.1 -mc_samples 1000 -multistep 1 -past_len 20
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_stock" -p_drop 0.2 -mc_samples 1000 -multistep 1 -past_len 20
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "baseline_t" -d_model 32 -dff 32 -bs 64 -ep 0 -output_path "output/exp_stock" -p_drop 0.5 -mc_samples 1000 -multistep 1 -past_len 20
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_stock" -multistep 1 -past_len 20
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_stock" -p_drop 0.1 -mc_samples 1000 -multistep 1 -past_len 20
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_stock" -p_drop 0.2 -mc_samples 1000 -multistep 1 -past_len 20
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "lstm" -rnn_units 32 -bs 64 -ep 0 -output_path "output/exp_stock" -p_drop 0.5 -mc_samples 1000 -multistep 1 -past_len 20
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "bayesian_lstm" -rnn_units 32 -bs 64 -ep 150 -output_path "output/exp_stock" -mc_samples 1000 -particles 3 -multistep 1 -past_len 20