#!/usr/bin/env bash
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 16 -dff 16 -bs 32 -pe 100 -ep 50 -output_path "output/covid"
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 16 -dff 16 -bs 32 -pe 100 -ep 50 -output_path "output/covid" -p_drop 0.1 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 16 -dff 16 -bs 32 -pe 100 -ep 50 -output_path "output/covid" -p_drop 0.2 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -pe 100 -ep 50 -output_path "output/covid" -p_drop 0.5 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -pe 100 -ep 50 -output_path "output/covid"
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -pe 100 -ep 50 -output_path "output/covid" -p_drop 0.1 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -pe 100 -ep 50 -output_path "output/covid" -p_drop 0.2 -mc_samples 1000
python src/scripts/run.py -dataset "covid" -data_path "data/covid" -algo "baseline_t" -d_model 32 -dff 32 -bs 32 -pe 100 -ep 50 -output_path "output/covid" -p_drop 0.5 -mc_samples 1000
