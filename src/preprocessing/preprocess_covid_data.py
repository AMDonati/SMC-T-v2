import argparse
import numpy as np
import os
import pandas as pd
from scipy import stats
import h5py


def preprocess_covid_data(csv_path):
    covid_data_df = pd.read_csv(csv_path)
    df = covid_data_df.iloc[:, 11:]
    df_60_days = df.iloc[:, 60:121]
    # removing rows with too few deaths.
    df_60_days = df_60_days[df_60_days.sum(axis=1) >= 100]
    z = np.abs(stats.zscore(df_60_days, axis=1))
    outliers = np.where((z > 6))
    df_60_days = df_60_days.drop(index=[3207, 3210], axis=0)
    final_arr = df_60_days.values
    print("data shape", final_arr.shape)
    return df_60_days, final_arr


def split_covid_data(covid_data, normalize=True, split=0.8):
    covid_data = covid_data.astype(np.float32)
    num_samples = covid_data.shape[0]
    TRAIN_SPLIT = int(num_samples * split)
    VAL_SPLIT = TRAIN_SPLIT + int(num_samples * (1 - split) * 0.5) + 1

    if normalize:
        data_mean = np.mean(covid_data, axis=1, keepdims=True)
        data_std = np.std(covid_data, axis=1, keepdims=True)
        covid_data = (covid_data - data_mean) / data_std
        stats_train = [data_mean[:TRAIN_SPLIT, :], data_std[:TRAIN_SPLIT, :]]
        stats_val = [data_mean[TRAIN_SPLIT:VAL_SPLIT, :], data_std[TRAIN_SPLIT:VAL_SPLIT, :]]
        stats_test = [data_mean[VAL_SPLIT:, :], data_std[VAL_SPLIT:, :]]
        stats = (stats_train, stats_val, stats_test)
    else:
        stats = None

    train_data = covid_data[:TRAIN_SPLIT, :]
    val_data = covid_data[TRAIN_SPLIT:VAL_SPLIT, :]
    test_data = covid_data[VAL_SPLIT:, :]

    # reshaping arrays:
    covid_data = np.reshape(covid_data, newshape=(covid_data.shape[0], covid_data.shape[1], 1))
    train_data = np.reshape(train_data, newshape=(train_data.shape[0], train_data.shape[1], 1))
    val_data = np.reshape(val_data, newshape=(val_data.shape[0], val_data.shape[1], 1))
    test_data = np.reshape(test_data, newshape=(test_data.shape[0], test_data.shape[1], 1))

    folder_path = os.path.join("data", "covid")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    np.save(os.path.join(folder_path, "covid_data.npy"), covid_data)
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
    # save statistics:
    stats_file = os.path.join(folder_path, "stats.h5")
    with h5py.File(stats_file, 'w') as f:
        f.create_dataset('train_mean', data=stats_train[0])
        f.create_dataset('train_std', data=stats_train[1])
        f.create_dataset('val_mean', data=stats_val[0])
        f.create_dataset('val_std', data=stats_val[1])
        f.create_dataset('test_mean', data=stats_test[0])
        f.create_dataset('test_std', data=stats_test[1])

    return train_data, val_data, test_data, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="data/covid/time_series_covid19_deaths_US.csv",
                        help="path of the csv file.")
    parser.add_argument("-TRAIN_SPLIT", type=float, default=0.8,
                        help="train split for splitting between train and validation sets.")
    parser.add_argument("-normalize", type=int, default=1, help="normalize the dataset.")
    args = parser.parse_args()

    final_df, covid_data = preprocess_covid_data(args.data_path)
    train_data, val_data, test_data, stats = split_covid_data(covid_data, normalize=args.normalize,
                                                              split=args.TRAIN_SPLIT)
    print('train_data shape', train_data.shape)
    print('train_data sample 0', train_data[0, :, :])
    print('val_data shape', val_data.shape)
    print('val_data sample 0', val_data[0, :, :])
    print('test_data shape', test_data.shape)
    print('test_data sample 0', test_data[0, :, :])
    stats_train, stats_val, stats_test = stats

