python src/train/train_SMC_T_simple.py -bs 64 -d_model 12 -ep 20 -data_path "data" -output_path "output" -full_model True -dff 48

python src/train/train_transformer.py -bs 64 -d_model 12 -ep 20 -data_path "data" -output_path "output" -pe 50 -dff 48 -launch_smc False

python src/train/train_RNN.py -bs 64 -rnn_units 20 -ep 20 -data_path "data" -output_path "output"
