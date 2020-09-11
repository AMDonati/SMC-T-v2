## Code for the paper: The Monte Carlo Transformer: a stochastic self-attention model

### Preprocessing the datasets
#### Synthetic datasets
`python create_synthetic_dataset.py -model 1 -num_samples 10,000 -seq_len 24 -num_features 1`
`python create_synthetic_dataset.py -model 2 -num_samples 10,000 -seq_len 24 -num_features 1`
#### Covid dataset
`python preprocess_covid_data.py -data_path "../../data/covid/covid_preprocess.npy"`
