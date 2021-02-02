#!/usr/bin/env bash
python src/preprocessing/create_synthetic_dataset.py -model 1 -num_samples 1000 -seq_len 24 -num_features 1
python src/preprocessing/create_synthetic_dataset.py -model 2 -alpha 0.9 -variance 0.3 -num_samples 1000 -seq_len 24
python src/preprocessing/preprocess_covid_data.py
python src/preprocessing/preprocess_weather_dataset.py
python src/preprocessing/preprocess_air_quality.py
python src/preprocessing/preprocess_stock.py
python src/preprocessing/preprocess_energy.py
