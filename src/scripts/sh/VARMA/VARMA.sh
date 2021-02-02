#!/usr/bin/env bash
echo "-------- air quality ---------------------------------------------"
#python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "varma" -p_model 1 -q 0 -ep 500 -output_path "output/exp_air_quality" -multistep 1 -past_len 6
#python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "varma" -p_model 1 -q 0 -ep 1000 -output_path "output/exp_air_quality" -multistep 1 -past_len 6
#python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "varma" -p_model 3 -q 0 -ep 1000 -output_path "output/exp_air_quality" -multistep 1 -past_len 6
#python src/scripts/run.py -dataset "air_quality" -data_path "data/air_quality" -algo "varma" -p_model 5 -q 0 -ep 1000 -output_path "output/exp_air_quality" -multistep 1 -past_len 6
echo "-------- stock ---------------------------------------------"
python src/scripts/run.py -dataset "stock" -data_path "data/stock" -algo "varma" -p_model 1 -q 0 -ep 1000 -output_path "output/exp_stock" -multistep 1 -past_len 20
echo "-------- energy ---------------------------------------------"
python src/scripts/run.py -dataset "energy" -data_path "data/energy" -algo "varma" -p_model 1 -q 0 -ep 1000 -output_path "output/exp_energy" -multistep 1 -past_len 6
echo "-------- weather ---------------------------------------------"
python src/scripts/run.py -dataset "weather" -data_path "data/weather" -algo "varma" -p_model 1 -q 0 -ep 1000 -output_path "output/exp_weather" -mc_samples 1000 -past_len 12 -multistep 1 -max_samples 50000