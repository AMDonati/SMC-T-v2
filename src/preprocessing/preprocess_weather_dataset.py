import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from src.preprocessing.utils import split_dataset_into_seq
import argparse

#TODO: finalize CV here.

def split_weather_dataset(file_path, fname, col_name, index_name, history, step, TRAIN_SPLIT, VAL_SPLIT=0.5,
                          VAL_SPLIT_cv=0.9, cv=False, max_samples=None, save_path=None):

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
        uni_data = uni_data[:max_samples, :]

    # # normalization
    # data_mean = uni_data.mean(axis=0)
    # data_std = uni_data.std(axis=0)
    # uni_data = (uni_data - data_mean) / data_std
    # stats = (data_mean, data_std)

    data_in_seq = split_dataset_into_seq(uni_data, 0, None, history, step)

    if not cv:
        # save_paths:
        if save_path is not None:
            train_path = os.path.join(save_path, "weather", "train")
            val_path = os.path.join(save_path, "weather", "val")
            test_path = os.path.join(save_path, "weather", "test")
            weather_path = os.path.join(save_path, "weather")
        for path in [train_path, val_path, test_path]:
            if not os.path.isdir(path):
                os.makedirs(path)

        # split between validation dataset and test set:
        train_data, val_data = train_test_split(data_in_seq, train_size=TRAIN_SPLIT, shuffle=True)
        val_data, test_data = train_test_split(val_data, train_size=VAL_SPLIT, shuffle=True)

        # normalization
        data_mean_seq = train_data.mean(axis=0)
        data_mean = data_mean_seq.mean(axis=0)
        data_std_seq = train_data.std(axis=0)
        data_std = data_std_seq.std(axis=0)
        stats = (data_mean, data_std)
        train_data = (train_data - data_mean) / data_std
        val_data = (val_data - data_mean) / data_std
        test_data = (test_data - data_mean) / data_std

        # reshaping arrays to have a (future shape) of (B,S,1):
        if len(col_name) == 1:
            train_data = np.reshape(train_data, newshape=(train_data.shape[0], train_data.shape[1], 1))
            val_data = np.reshape(val_data, newshape=(val_data.shape[0], val_data.shape[1], 1))
            test_data = np.reshape(test_data, newshape=(test_data.shape[0], test_data.shape[1], 1))

        # save datasets:
        print("train data shape", train_data.shape)
        print("val data shape", val_data.shape)
        print("test data shape", test_data.shape)
        if save_path is not None:
            print("saving datasets into .npy files...")
            np.save(os.path.join(train_path, "weather.npy"), train_data)
            np.save(os.path.join(val_path, "weather.npy"), val_data)
            np.save(os.path.join(test_path, "weather.npy"), test_data)
            np.save(os.path.join(weather_path, "means.npy"), data_mean)
            np.save(os.path.join(weather_path, "stds.npy"), data_std)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="data",
                        help="data folder to upload and save weather dataset.")
    parser.add_argument("-TRAIN_SPLIT", type=float, default=0.7,
                        help="train split for splitting between train and validation sets.")
    parser.add_argument("-history", type=int, default=600, help="history of past observations.")
    # history =  6 * 24 * 4 + 6 * 4 # 25 timesteps sampled every 4 hours (every 24 samples).  > time-window of 4 days for temperature sampled every 4 hours. temperature recorded every 10min.
    parser.add_argument("-step", type=int, default=24, help="sample step.")
    # step = 6 * 4  # sample a temperature every 4 hours (samples taken every 10 mins.)
    args = parser.parse_args()

    file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
    fname = 'jena_climate_2009_2016.csv.zip'
    col_name = ['p (mbar)', 'T (degC)', 'rh (%)', 'rho (g/m**3)']
    index_name = 'Date Time'

    (train_data, val_data, test_data), original_df, stats = split_weather_dataset(file_path=file_path,
                                                                                  fname=fname,
                                                                                  col_name=col_name,
                                                                                  index_name=index_name,
                                                                                  TRAIN_SPLIT=args.TRAIN_SPLIT,
                                                                                  history=args.history,
                                                                                  step=args.step,
                                                                                  cv=False,
                                                                                  max_samples=None,
                                                                                  save_path=args.data_path)
