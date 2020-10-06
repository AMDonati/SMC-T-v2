import argparse

from src.preprocessing.utils import *


def preprocess_dataframe(csv_path, save_path=None):
    df = pd.read_csv(csv_path, index_col='date', parse_dates=['date'])
    list_cols = list(df.columns)
    df = convert_col_into_float64(df, list_cols)

    assert df.isnull().any(axis=1).sum() == 0, "error in nan values"

    if save_path is not None:
        path = os.path.join(save_path, "energy")
        if not os.path.isdir(path):
            os.makedirs(path)
        df.to_csv(os.path.join(path, "raw_data.csv"))
        np.save(os.path.join(path, "raw_data.npy"), df.values)
    return df, df.values


def split_dataset(data, TRAIN_SPLIT, VAL_SPLIT=0.5, save_path=None):
    # normalization
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std
    stats = (data_mean, data_std)

    data_in_seq = split_dataset_into_seq(data, start_index=0, end_index=None, history_size=13, step=1)

    # save_paths:
    if save_path is not None:
        train_path = os.path.join(save_path, "energy", "train")
        val_path = os.path.join(save_path, "energy", "val")
        test_path = os.path.join(save_path, "energy", "test")
        aq_path = os.path.join(save_path, "energy")

        for path in [train_path, val_path, test_path]:
            if not os.path.isdir(path):
                os.makedirs(path)
    # split between validation dataset and test set:
    train_data, val_data = train_test_split(data_in_seq, train_size=TRAIN_SPLIT, shuffle=True)
    val_data, test_data = train_test_split(val_data, train_size=VAL_SPLIT, shuffle=True)

    # save datasets:
    if save_path is not None:
        print("saving datasets into .npy files...")
        np.save(os.path.join(train_path, "energy.npy"), train_data)
        np.save(os.path.join(val_path, "energy.npy"), val_data)
        np.save(os.path.join(test_path, "energy.npy"), test_data)
        np.save(os.path.join(aq_path, "means.npy"), data_mean)
        np.save(os.path.join(aq_path, "stds.npy"), data_std)

    return (train_data, val_data, test_data), data_in_seq, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv_path", type=str, default="data/energy/energydata_complete.csv")
    parser.add_argument("-data_path", type=str, default="data",
                        help="data folder to upload and save energy dataset.")
    parser.add_argument("-TRAIN_SPLIT", type=float, default=0.7,
                        help="train split for spliting between train and validation sets.")
    parser.add_argument("-history", type=int, default=13, help="history of past observations.")
    args = parser.parse_args()

    df, array = preprocess_dataframe(args.csv_path, args.data_path)
    (train_data, val_data, test_data), data_in_seq, (mean, std) = split_dataset(data=array, TRAIN_SPLIT=args.TRAIN_SPLIT,
                                                                                save_path=args.data_path)
    print("train data shape", train_data.shape)
    print("val data shape", val_data.shape)
    print("test data shape", test_data.shape)
    print("stats - mean:", mean)
    print("stats - std:", std)
