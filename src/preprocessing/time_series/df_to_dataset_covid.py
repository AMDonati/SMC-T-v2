import os
from preprocessing.time_series.df_to_dataset_synthetic import data_to_dataset_3D, data_to_dataset_4D, split_input_target
import numpy as np


def split_covid_data(arr_path):
    covid_data = np.load(arr_path)
    covid_data = covid_data.astype(np.float32)

    data_mean = np.mean(covid_data, axis=1, keepdims=True)
    data_std = np.std(covid_data, axis=1, keepdims=True)
    covid_data = (covid_data - data_mean) / data_std

    train_data = covid_data[:709,:]
    stats_train = (data_mean[:709,:], data_std[:709,:])
    val_data = covid_data[709:798,:]
    stats_val = (data_mean[709:798, :], data_std[709:798, :])
    test_data = covid_data[798:, :]
    stats_test = (data_mean[798:, :], data_std[798:,:])

    # reshaping arrays:
    train_data = np.reshape(train_data, newshape=(train_data.shape[0], train_data.shape[1], 1))
    val_data = np.reshape(val_data, newshape=(val_data.shape[0], val_data.shape[1], 1))
    test_data = np.reshape(test_data, newshape=(test_data.shape[0], test_data.shape[1], 1))

    return train_data, val_data, test_data, (stats_train, stats_val, stats_test)

def rescale_covid_data(data_sample, stats, index):
    data_mean, data_std = stats
    mean, std = data_mean[index], data_std[index]
    data_sample = std * data_sample + mean
    data_sample = data_sample.astype(np.int32)
    return data_sample

#TODO: add a function that split between input and target.


if __name__ == '__main__':
    arr_path = '../../../data/covid_preprocess.npy'
    train_data, val_data, test_data, stats = split_covid_data(arr_path=arr_path)
    stats_train, stats_val, stats_test = stats
    first_sample = rescale_covid_data(train_data[0], stats_train, 0)
    last_sample = rescale_covid_data(test_data[-1], stats_test, -1)
    data_unnorm = np.load(arr_path)

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

    # saving train, val
    data_path = '../../../data'
    train_data_path = os.path.join(data_path, 'covid_train_data.npy')
    val_data_path = os.path.join(data_path, 'covid_val_data.npy')
    test_data_path = os.path.join(data_path, 'covid_test_data.npy')

    np.save(val_data_path, val_data)
    np.save(train_data_path, train_data)
    np.save(test_data_path, test_data)

    print('train_data', train_data.shape)
    print('train_data', train_data[0, :, :])
    print('val_data', val_data.shape)
    print('val_data', val_data[0, :, :])
    print('test_data', test_data.shape)
    print('test_data', test_data[0, :, :])

    train_dataset, val_dataset, test_dataset = data_to_dataset_3D(train_data, val_data, test_data,
                                                                  split_fn=split_input_target, BUFFER_SIZE=50, BATCH_SIZE=32, cv=False)
    for (inp, tar) in train_dataset.take(1):
        print('input', inp.shape)
        print('input', inp[0,:,:])
        print('target', tar.shape)
        print('target', tar[0,:,:])
    train_dataset, val_dataset, test_dataset = data_to_dataset_4D(train_data, val_data, test_data,
                                                                  split_fn=split_input_target, BUFFER_SIZE=50,
                                                                  BATCH_SIZE=32, cv=False)
    for (inp, tar) in train_dataset.take(1):
        print('input', inp.shape)
        print('input', inp[0,:,:])
        print('target', tar.shape)
        print('target', tar[0,:,:])
