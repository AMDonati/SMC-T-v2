#!/usr/bin/env bash
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -pe 100 -ep 0 -output_path "output/covid" -p_drop 0.1 -mc_samples 1000 -save_path "output/exp_covid/baseline_t/classic_T_d32_p0.1/1" -multistep 1 -past_len 40
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -pe 100 -ep 0 -output_path "output/covid" -p_drop 0.2 -mc_samples 1000 -save_path "output/exp_covid/baseline_t/classic_T_d32_p0.2/1" -multistep 1 -past_len 40
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -pe 100 -ep 0 -output_path "output/covid" -p_drop 0.5 -mc_samples 1000 -save_path "output/exp_covid/baseline_t/classic_T_d32_p0.5/1" -multistep 1 -past_len 40
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "lstm" -rnn_units 32 -bs 32 -ep 0 -output_path "output/exp_covid" -p_drop 0.1 -mc_samples 1000 -save_path "output/exp_covid/lstm/lstm_d32_p0.1/1" -multistep 1 -past_len 40
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "lstm" -rnn_units 32 -bs 32 -ep 0 -output_path "output/exp_covid" -p_drop 0.2 -mc_samples 1000 -save_path "output/exp_covid/lstm/lstm_d32_p0.2/1" -multistep 1 -past_len 40
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "lstm" -rnn_units 32 -bs 32 -ep 0 -output_path "output/exp_covid" -p_drop 0.5 -mc_samples 1000 -save_path "output/exp_covid/lstm/lstm_d32_p0.5/1" -multistep 1 -past_len 40
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "bayesian_lstm" -rnn_units 32 -bs 32 -ep 0 -output_path "output/exp_covid" -mc_samples 1000 -particles 3 -save_path "output/exp_covid/covid_bayes" -multistep 1 -past_len 40
