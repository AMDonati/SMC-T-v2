import os
from preprocessing.time_series.df_to_dataset_synthetic import data_to_dataset_3D, data_to_dataset_4D, split_input_target
import numpy as np

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

    return train_data, val_data, test_data, stats

def rescale_covid_data(data_sample, stats, index):
    data_mean, data_std = stats
    mean, std = data_mean[index], data_std[index]
    data_sample = std * data_sample + mean
    data_sample = data_sample.astype(np.int32)
    return data_sample


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


    # ---- covid data rescaled ------------------------------------------------

    print("data with rescaling...")
    arr_path = '../../../data/covid_preprocess_rescaled.npy'
    train_data_s, val_data_s, test_data_s, stats = split_covid_data(arr_path=arr_path, normalize=False)

    print('train_data', train_data_s.shape)
    print('train_data', train_data_s[0, :, :])
    print('val_data', val_data_s.shape)
    print('val_data', val_data_s[0, :, :])
    print('test_data', test_data_s.shape)
    print('test_data', test_data_s[0, :, :])

    # saving train, val
    data_path = '../../../data'
    train_data_path = os.path.join(data_path, 'covid_train_data_rescaled.npy')
    val_data_path = os.path.join(data_path, 'covid_val_data_rescaled.npy')
    test_data_path = os.path.join(data_path, 'covid_test_data_rescaled.npy')

    np.save(val_data_path, val_data_s)
    np.save(train_data_path, train_data_s)
    np.save(test_data_path, test_data_s)


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
