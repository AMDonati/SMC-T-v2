import tensorflow as tf
import os, argparse
import numpy as np
from preprocessing.time_series.df_to_dataset_synthetic import split_synthetic_dataset, data_to_dataset_3D, split_input_target
from utils.utils_train import create_logger
from models.Baselines.RNNs import build_LSTM_for_regression
from train.train_functions import train_LSTM

if __name__ == '__main__':

  #trick for boolean parser args.
  def str2bool(v):
    if isinstance(v, bool):
      return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

  parser = argparse.ArgumentParser()

  parser.add_argument("-rnn_units", type=int, required=True, help="number of rnn units")
  parser.add_argument("-bs", type=int, default=128, help="batch size")
  parser.add_argument("-ep", type=int, default=20, help="number of epochs")
  parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
  parser.add_argument("-data_path", type=str, required=True, help="path for saving data")
  parser.add_argument("-output_path", type=str, required=True, help="path for output folder")

  args = parser.parse_args()

  # ------------------- Upload synthetic dataset ----------------------------------------------------------------------------------
  BUFFER_SIZE = 500
  BATCH_SIZE = args.bs
  TRAIN_SPLIT = 0.7

  data_path = os.path.join(args.data_path, 'synthetic_dataset_1_feat.npy')
  input_data = np.load(data_path)
  train_data, val_data, test_data = split_synthetic_dataset(x_data=input_data, TRAIN_SPLIT=TRAIN_SPLIT, cv=False)

  val_data_path = os.path.join(args.data_path, 'val_data_synthetic_1_feat.npy')
  train_data_path = os.path.join(args.data_path, 'train_data_synthetic_1_feat.npy')
  test_data_path = os.path.join(args.data_path, 'test_data_synthetic_1_feat.npy')

  np.save(val_data_path, val_data)
  np.save(train_data_path, train_data)
  np.save(test_data_path, test_data)

  train_dataset, val_dataset, test_dataset = data_to_dataset_3D(train_data=train_data,
                                                                val_data=val_data,
                                                                test_data=test_data,
                                                                split_fn=split_input_target,
                                                                BUFFER_SIZE=BUFFER_SIZE,
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                target_feature=None,
                                                                cv=False)

  # -------------------- define hyperparameters -----------------------------------------------------------------------------------
  rnn_units = args.rnn_units
  learning_rate = args.lr
  EPOCHS = args.ep
  # define optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)
  output_path = args.output_path
  out_file = 'LSTM_units_{}_lr_{}_bs_{}'.format(rnn_units, learning_rate, BATCH_SIZE)
  output_path = os.path.join(output_path, out_file)
  if not os.path.isdir(output_path):
    os.makedirs(output_path)

  for inp, tar in train_dataset.take(1):
    seq_len = tf.shape(inp)[1].numpy()
    num_features = tf.shape(inp)[-1].numpy()
    output_size = tf.shape(tar)[-1].numpy()

  # -------------------- create logger and checkpoint saver ----------------------------------------------------------------------------------------------------

  out_file_log = output_path + '/' + 'training_log.log'
  logger = create_logger(out_file_log=out_file_log)
  #  creating the checkpoint manager:
  checkpoint_path = os.path.join(output_path, "checkpoints")
  if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

  # -------------------- Build the RNN model -----------------------------------------------------------------------------------------
  model = build_LSTM_for_regression(shape_input_1=seq_len,
                                    shape_input_2=num_features,
                                    shape_output=output_size,
                                    rnn_units=rnn_units,
                                    dropout_rate=0,
                                    training=True)

  train_LSTM(model=model,
             optimizer=optimizer,
             EPOCHS=EPOCHS,
             train_dataset=train_dataset,
             val_dataset=val_dataset,
             checkpoint_path=checkpoint_path,
             output_path=output_path,
             logger=logger,
             num_train=1)





