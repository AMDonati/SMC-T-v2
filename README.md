## Code for the paper: The Monte Carlo Transformer: a stochastic self-attention model for sequence prediction

### Preprocessing the datasets
#### Synthetic datasets
* `python src/preprocessing/create_synthetic_dataset.py -model 1 -num_samples 1000 -seq_len 24 -num_features 1`
* `python src/preprocessing/create_synthetic_dataset.py -model 2 -alpha 0.9 -variance 0.3 -num_samples 1000 -seq_len 24 -num_features 1`
#### Covid dataset
`python preprocess_covid_data.py -data_path "../../data/covid/covid_preprocess.npy"`
