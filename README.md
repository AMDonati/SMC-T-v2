## Code for the paper: The Monte Carlo Transformer: a stochastic self-attention model for sequence prediction

### Preprocessing the datasets
#### Synthetic datasets
* `python src/preprocessing/create_synthetic_dataset.py -model 1 -num_samples 1000 -seq_len 24 -num_features 1`
* `python src/preprocessing/create_synthetic_dataset.py -model 2 -alpha 0.9 -variance 0.3 -num_samples 1000 -seq_len 24 -num_features 1`
#### Covid dataset
`python src/preprocessing/preprocess_covid_data.py`
#### Weather Dataset
`python src/preprocessing/preprocess_weather_dataset.py`
#### Air Quality Dataset
`python src/preprocessing/preprocess_air_quality.py`


### Training the Baselines
#### LSTM with Dropout
* python src/scripts/run.py -dataset "synthetic" -dataset "synthetic" -dataset_model 1 -data_path "../../data/synthetic_model_1" -algo "lstm" -rnn_units 32 -p_drop 0.1 -rnn_drop 0.1 -bs 32 -ep 50 -output_path "exp_synthetic_model_1"
#### Classic Transformer
* python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "../../data/synthetic_model_1" -algo "baseline_t" -d_model 8 -dff 8 -pe 50 -bs 32 -ep 50 -output_path "exp_synthetic_model_1"

### Training the SMC-Transformer
#### Recurrent Transformer without noise and smc algo
* python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "../../data/synthetic_model_1" -algo "smc_t" -d_model 8 -dff 8 -bs 32 -ep 50 -smc False -output_path "exp_synthetic_model_1"
#### SMC Transformer
* python src/scripts/run.py -dataset "synthetic" -dataset_model 1 -data_path "../../data/synthetic_model_1" -algo "smc_t" -d_model 8 -dff 8 -bs 32 -ep 50 -particles 10 -smc True -output_path "exp_synthetic_model_1"
