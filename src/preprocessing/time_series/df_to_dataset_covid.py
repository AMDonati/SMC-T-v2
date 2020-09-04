import os
from preprocessing.time_series.df_to_dataset_synthetic import data_to_dataset_3D, data_to_dataset_4D, split_input_target
import numpy as np
import argparse

def split_covid_data(arr_path, normalize=True, split=0.8):
    covid_data = np.load(arr_path)
    covid_data = covid_data.astype(np.float32)
    num_samples = covid_data.shape[0]
    TRAIN_SPLIT = int(num_samples*split)
    VAL_SPLIT = TRAIN_SPLIT + int(num_samples*(1-split)*0.5) + 1

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

def rescale_covid_data(data_sample, stats, index):
    data_mean, data_std = stats
    mean, std = data_mean[index], data_std[index]
    data_sample = std * data_sample + mean
    data_sample = data_sample.astype(np.int32)
    return data_sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../../../data/covid/covid_preprocess.npy", help="data folder to upload and save covid dataset.")
    parser.add_argument("-TRAIN_SPLIT", type=float, default=0.7,
                        help="train split for splitting between train and validation sets.")
    parser.add_argument("-normalize", type=int, default=1, help="normalize the dataset.")
    args = parser.parse_args()

    train_data, val_data, test_data, stats = split_covid_data(arr_path=args.data_path, normalize=args.normalize, split=args.TRAIN_SPLIT)
    print('train_data shape', train_data.shape)
    print('train_data sample 0', train_data[0, :, :])
    print('val_data shape', val_data.shape)
    print('val_data sample 0', val_data[0, :, :])
    print('test_data shape', test_data.shape)
    print('test_data sample 0', test_data[0, :, :])
    stats_train, stats_val, stats_test = stats


    # first_sample = rescale_covid_data(train_data[0], stats_train, 0)
    # last_sample = rescale_covid_data(test_data[-1], stats_test, -1)
    # data_unnorm = np.load(arr_path)
    # checking mean and max values of each dataset:
    # test_sum = np.sum(data_unnorm[798:], axis=1)
    # test_max = np.max(test_sum)
    # test_mean = np.mean(test_sum)
    # val_sum = np.sum(data_unnorm[709:798], axis=1)
    # val_max = np.max(val_sum)
    # val_mean = np.max(val_sum)
    # train_sum = np.sum(data_unnorm[:709], axis=1)
    # train_mean = np.mean(train_sum)
    # train_max = np.max(train_sum)



