python src/train/train_SMC_T.py -bs 64 -d_model 6 -ep 30 -smc True -particles 10 -data_path "data" -output_path "output"

python src/train/train_transformer.py -bs 64 -d_model 12 -ep 20 -data_path "data" -output_path "output" -pe 50 -dff 48 -launch_smc False

python src/train/train_RNN.py -bs 64 -rnn_units 20 -ep 20 -data_path "data" -output_path "output"
