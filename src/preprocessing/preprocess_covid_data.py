import argparse
from preprocessing.utils import split_covid_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../../data/covid/covid_preprocess.npy", help="data folder to upload and save covid dataset.")
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



