python src/train/train_SMC_T.py -bs 256 -d_model 32 -ep 100 -smc True -particles 10 -sigmas 0.5 -sigma_obs 0.5 -data_path "data" -output_path "output"

python src/train/train_transformer.py -bs 64 -d_model 12 -ep 20 -data_path "data" -output_path "output" -pe 50 -dff 48 -launch_smc False

python src/train/train_RNN.py -bs 256 -rnn_units 32 -ep 50 -p_drop 0.1 -cv True -data_path "data" -output_path "output/RNN_exp_weather"
