import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np


def split_dataset_into_seq(dataset, start_index, end_index, history_size, step):
    data = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
    return np.array(data)


def split_weather_dataset(file_path, fname, col_name, index_name, history, step, TRAIN_SPLIT, VAL_SPLIT=0.5,
                          VAL_SPLIT_cv=0.9, cv=False, max_samples=None):
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
    if max_samples is not None:
        # index = list(np.random.randint(0, len(uni_data_df), size=max_samples))
        uni_data = uni_data[:max_samples, :]

    # normalization
    data_mean = uni_data.mean(axis=0)
    data_std = uni_data.std(axis=0)
    uni_data = (uni_data - data_mean) / data_std

    stats = (data_mean, data_std)

    data_in_seq = split_dataset_into_seq(uni_data, 0, None, history, step)

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

def split_synthetic_dataset(x_data, save_path, TRAIN_SPLIT, VAL_SPLIT=0.5, VAL_SPLIT_cv=0.9, cv=False):
  if not cv:
    train_data, val_test_data = train_test_split(x_data, train_size=TRAIN_SPLIT, shuffle=True)
    val_data, test_data = train_test_split(val_test_data, train_size=VAL_SPLIT, shuffle=True)
    train_data_path = os.path.join(save_path, "train")
    val_data_path = os.path.join(save_path, "val")
    test_data_path = os.path.join(save_path, "test")
    for path in [train_data_path, val_data_path, test_data_path]:
      if not os.path.isdir(path):
        os.makedirs(path)
    np.save(os.path.join(train_data_path, "synthetic.npy"), train_data)
    np.save(os.path.join(val_data_path, "synthetic.npy"), val_data)
    np.save(os.path.join(test_data_path, "synthetic.npy"), test_data)
    print("saving train, val, and test data into .npy files...")
    return train_data, val_data, test_data
  else:
    train_val_data, test_data = train_test_split(x_data, train_size=VAL_SPLIT_cv)
    kf = KFold(n_splits=5)
    list_train_data, list_val_data = [], []
    for train_index, val_index in kf.split(train_val_data):
      train_data = train_val_data[train_index, :, :]
      val_data = train_val_data[val_index, :, :]
      list_train_data.append(train_data)
      list_val_data.append(val_data)

    return list_train_data, list_val_data, test_data


def split_covid_data(arr_path, normalize=True, split=0.8):
    covid_data = np.load(arr_path)
    covid_data = covid_data.astype(np.float32)
    num_samples = covid_data.shape[0]
    TRAIN_SPLIT = int(num_samples * split)
    VAL_SPLIT = TRAIN_SPLIT + int(num_samples * (1 - split) * 0.5) + 1
    if normalize:
        data_mean = np.mean(covid_data, axis=1, keepdims=True)
        data_std = np.std(covid_data, axis=1, keepdims=True)
        covid_data = (covid_data - data_mean) / data_std
        stats_train = (data_mean[:TRAIN_SPLIT, :], data_std[:TRAIN_SPLIT, :])
        stats_val = (data_mean[TRAIN_SPLIT:VAL_SPLIT, :], data_std[TRAIN_SPLIT:VAL_SPLIT, :])
        stats_test = (data_mean[VAL_SPLIT:, :], data_std[VAL_SPLIT:, :])
        stats = (stats_train, stats_val, stats_test)
    else:
        stats = None

    train_data = covid_data[:TRAIN_SPLIT, :]
    val_data = covid_data[TRAIN_SPLIT:VAL_SPLIT, :]
    test_data = covid_data[VAL_SPLIT:, :]

    # reshaping arrays:
    train_data = np.reshape(train_data, newshape=(train_data.shape[0], train_data.shape[1], 1))
    val_data = np.reshape(val_data, newshape=(val_data.shape[0], val_data.shape[1], 1))
    test_data = np.reshape(test_data, newshape=(test_data.shape[0], test_data.shape[1], 1))

    folder_path = os.path.dirname(arr_path)
    train_data_path = os.path.join(folder_path, "train")
    val_data_path = os.path.join(folder_path, "val")
    test_data_path = os.path.join(folder_path, "test")
    for path in [train_data_path, val_data_path, test_data_path]:
        if not os.path.isdir(path):
            os.makedirs(path)
    np.save(os.path.join(train_data_path, "covid.npy"), train_data)
    np.save(os.path.join(val_data_path, "covid.npy"), val_data)
    np.save(os.path.join(test_data_path, "covid.npy"), test_data)
    print("saving train, val and test sets in npy files...")
    return train_data, val_data, test_data, stats


if __name__ == '__main__':
    file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
    fname = 'jena_climate_2009_2016.csv.zip'
    col_name = ['p (mbar)', 'T (degC)', 'rh (%)', 'rho (g/m**3)']
    index_name = 'Date Time'
    # temperature recorded every 10 minutes.
    TRAIN_SPLIT = 0.7
    history = 6 * 24 * 4 + 6 * 4
    step = 6 * 4  # sample a temperature every 4 hours.
    cv = False

    (train_data, val_data, test_data), original_df, stats = split_weather_dataset(file_path=file_path,
                                                                                  fname=fname,
                                                                                  col_name=col_name,
                                                                                  index_name=index_name,
                                                                                  TRAIN_SPLIT=TRAIN_SPLIT,
                                                                                  history=history,
                                                                                  step=step,
                                                                                  cv=cv,
                                                                                  max_samples=20000)

    print(train_data[:5])
