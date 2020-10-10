#!/usr/bin/env bash
python src/scripts/run.py -dataset "energy" -data_path "data/energy_vm6" -algo "lstm" -rnn_units 32 -bs 128 -ep 0 -output_path "output/exp_energy" -p_drop 0.1 -mc_samples 1000 -multistep 1 -past_len 6 -save_path "output/exp_energy/lstm_d32_p0.1/1"
python src/scripts/run.py -dataset "energy" -data_path "data/energy_vm6" -algo "lstm" -rnn_units 32 -bs 128 -ep 0 -output_path "output/exp_energy" -p_drop 0.2 -mc_samples 1000 -multistep 1 -past_len 6 -save_path "output/exp_energy/lstm_d32_p0.2/1"
python src/scripts/run.py -dataset "energy" -data_path "data/energy_vm6" -algo "lstm" -rnn_units 32 -bs 128 -ep 0 -output_path "output/exp_energy" -p_drop 0.5 -mc_samples 1000 -multistep 1 -past_len 6 -save_path "output/exp_energy/lstm_d32_p0.5/1"
python src/scripts/run.py -dataset "energy" -dataset_model 1 -data_path "data/energy_vm6" -algo "bayesian_lstm" -rnn_units 32 -bs 128 -ep 0 -output_path "output/exp_energy" -mc_samples 1000 -particles 3 -multistep 1 -past_len 6 -save_path "output/exp_energy/bayesian_lstm_d32/1"