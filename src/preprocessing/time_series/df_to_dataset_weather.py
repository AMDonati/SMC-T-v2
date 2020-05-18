import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from preprocessing.time_series.df_to_dataset_synthetic import data_to_dataset_3D, split_input_target
import numpy as np

def split_dataset_into_seq(dataset, start_index, end_index, history_size, step):
  data = []
  start_index = start_index + history_size

  if end_index is None:
    end_index=len(dataset)

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

  return np.array(data)

def df_to_data_regression(file_path, fname, col_name, index_name, history, step, TRAIN_SPLIT, VAL_SPLIT=0.5, VAL_SPLIT_cv=0.9, cv=False, max_samples=None):
  zip_path = tf.keras.utils.get_file(
      origin=file_path,
      fname=fname,
      extract=True)

  csv_path, _ = os.path.splitext(zip_path)
  df = pd.read_csv(csv_path)
  uni_data_df = df[col_name]
  uni_data_df.index = df[index_name]
  print('length of original continuous dataset: {}'.format(len(uni_data_df)))

  uni_data = uni_data_df.values

  # normalization
  data_mean = uni_data.mean(axis=0)
  data_std = uni_data.std(axis=0)
  uni_data = (uni_data - data_mean) / data_std

  stats = (data_mean, data_std)

  data_in_seq = split_dataset_into_seq(uni_data, 0, None, history, step)
  if max_samples is not None:
      data_in_seq = data_in_seq[:max_samples, :, :]
  if not cv:
    # split between validation dataset and test set:
    train_data, val_data = train_test_split(data_in_seq, train_size=TRAIN_SPLIT, shuffle=True)
    val_data, test_data = train_test_split(val_data, train_size=VAL_SPLIT, shuffle=True)

    # reshaping arrays to have a (future shape) of (B,S,1):
    if len(col_name) == 1:
      train_data = np.reshape(train_data, newshape=(train_data.shape[0], train_data.shape[1], 1))
      val_data = np.reshape(val_data, newshape=(val_data.shape[0], val_data.shape[1], 1))
      test_data = np.reshape(test_data, newshape=(test_data.shape[0], test_data.shape[1], 1))

    return (train_data, val_data, test_data), uni_data_df, stats

  else:
    train_val_data, test_data = train_test_split(data_in_seq, train_size=VAL_SPLIT_cv)
    kf = KFold(n_splits=5)
    list_train_data, list_val_data = [], []
    for train_index, val_index in kf.split(train_val_data):
      train_data = train_val_data[train_index, :, :]
      val_data = train_val_data[val_index, :, :]
      list_train_data.append(train_data)
      list_val_data.append(val_data)

    if len(col_name) == 1:
      list_train_data = [np.reshape(d, newshape=(d.shape[0], d.shape[1], 1)) for d in list_train_data]
      list_val_data = [np.reshape(d, newshape=(d.shape[0], d.shape[1], 1)) for d in list_val_data]
      test_data = np.reshape(test_data, newshape=(test_data.shape[0], test_data.shape[1], 1))

    return (list_train_data, list_val_data, test_data), uni_data_df, stats



if __name__ == '__main__':
    file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
    fname = 'jena_climate_2009_2016.csv.zip'
    col_name = ['p (mbar)', 'T (degC)', 'rh (%)', 'rho (g/m**3)']
    index_name = 'Date Time'
    # temperature recorded every 10 minutes.
    TRAIN_SPLIT = 0.7
    history = 6*24*4 + 6*4
    step = 6*4  # sample a temperature every 4 hours.
    cv = False

    (train_data, val_data, test_data), original_df, stats = df_to_data_regression(file_path=file_path,
                                                                                  fname=fname,
                                                                                  col_name=col_name,
                                                                                  index_name=index_name,
                                                                                  TRAIN_SPLIT=TRAIN_SPLIT,
                                                                                  history=history,
                                                                                  step=step,
                                                                                  cv=cv,
                                                                                  max_samples=70000)

    print(train_data[:5])

    BUFFER_SIZE = 5000
    BATCH_SIZE = 64

    train_dataset, val_dataset, test_dataset = data_to_dataset_3D(train_data=train_data,
                                                                  val_data=val_data,
                                                                  test_data=test_data,
                                                                   split_fn=split_input_target,
                                                                   BUFFER_SIZE=BUFFER_SIZE,
                                                                   BATCH_SIZE=BATCH_SIZE,
                                                                   cv=cv)
    print('train dataset', len(train_dataset))