python src/train/train_SMC_T.py -bs 256 -d_model 12 -ep 100 -smc True -full_model True -particles 10 -sigmas 0.5 -sigma_obs 0.5 -data_path "data" -output_path "output/exp_w_EM"

python src/train/train_SMC_T.py -bs 256 -d_model 32 -ep 30 -smc False -full_model True  -data_path "data" -output_path "output/exp_2052020"

python src/train/train_SMC_T.py -bs 256 -d_model 32 -ep 30 -smc False -full_model True  -data_path "data" -output_path "output/Recurrent_T_weather"

python src/train/train_transformer.py -bs 256 -d_model 12 -ep 30 -full_model True -pe 50 -data_path "data" -output_path "output/classic_T_weather"
python src/train/train_RNN.py -bs 256 -rnn_units 32 -ep 50 -p_drop 0.1 -cv True -data_path "data" -output_path "output/RNN_exp_weather"

# covid:

python src/train/train_SMC_T.py -bs 32 -d_model 8 -dff 16 -ep 50 -smc True -full_model True -particles 30 -data_path "data" -output_path "output/covid"

python src/train/train_SMC_T.py -bs 32 -d_model 8 -dff 16 -ep 30 -smc False -full_model True -data_path "data" -output_path "output/covid"

python src/train/train_RNN.py -bs 32 -rnn_units 16 -ep 30 -p_drop 0.1 -cv True -data_path "data" -output_path "output/covid_rnn"

python src/train/train_RNN.py -bs 32 -rnn_units 16 -ep 30 -p_drop 0.1 -rnn_drop 0.1 -cv True -data_path "data" -output_path "output/covid_rnn"