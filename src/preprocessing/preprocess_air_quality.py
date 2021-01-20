from src.preprocessing.utils import *
import os
from sklearn.model_selection import train_test_split
import argparse


def preprocess_dataframe(csv_path, save_path=None):
    df = pd.read_csv(csv_path, sep=';')
    df = df.iloc[:, :-2]
    list_cols = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
    df = convert_col_into_float64(df, list_cols)
    # removing columns with too much missing values.
    df = df.drop(labels=['CO(GT)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)'], axis=1)

    # fill-up missing values with mean
    list_cols = list(df.columns)[2:]
    df, rows_mv = fill_missing_values(df=df, list_cols=list_cols, value=-200)
    assert np.sum(np.sum(df == -200)) == 0, "error in missing values"

    # remove rows with nan values.
    rows_with_nan = get_rows_nan_values(df=df)
    df_process = df.iloc[:9357, :]  # remove nan values...
    assert np.sum(np.sum(df_process.isnull())) == 0, "error in nan values"

    # remove incomplete half-days.
    df_process = df_process.iloc[6:-3, :]  # remove bouts of days.

    df_process['DateTime'] = df_process['Date'] + '-' + df_process['Time']
    df_process = df_process.set_index(keys='DateTime')
    df_process = df_process.drop(labels=["Date", "Time"], axis=1)

    # changing orders of columns:
    df_process = df_process[['PT08.S1(CO)',
                             'PT08.S2(NMHC)',
                             'PT08.S3(NOx)',
                             'PT08.S4(NO2)',
                             'PT08.S5(O3)',
                             'C6H6(GT)',
                             'T',
                             'RH',
                             'AH']]
    if save_path is not None:
        path = os.path.join(save_path, "air_quality")
        if not os.path.isdir(path):
            os.makedirs(path)
        df_process.to_csv(os.path.join(path, "raw_data.csv"))
        np.save(os.path.join(path, "raw_data.npy"), df_process.values)
    return df_process, df_process.values


def split_air_quality_dataset(data, TRAIN_SPLIT, VAL_SPLIT=0.5, save_path=None):
    # normalization
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std
    stats = (data_mean, data_std)

    data_in_seq = split_dataset_into_seq(data, start_index=0, end_index=None, history_size=13, step=1)
    # save_paths:
    if save_path is not None:
        train_path = os.path.join(save_path, "air_quality", "train")
        val_path = os.path.join(save_path, "air_quality", "val")
        test_path = os.path.join(save_path, "air_quality", "test")
        aq_path = os.path.join(save_path, "air_quality")

        for path in [train_path, val_path, test_path]:
            if not os.path.isdir(path):
                os.makedirs(path)
    # split between validation dataset and test set:
    train_data, val_data = train_test_split(data_in_seq, train_size=TRAIN_SPLIT, shuffle=True)
    val_data, test_data = train_test_split(val_data, train_size=VAL_SPLIT, shuffle=True)

    # save datasets:
    if save_path is not None:
        print("saving datasets into .npy files...")
        np.save(os.path.join(train_path, "air_quality.npy"), train_data)
        np.save(os.path.join(val_path, "air_quality.npy"), val_data)
        np.save(os.path.join(test_path, "air_quality.npy"), test_data)
        np.save(os.path.join(aq_path, "means.npy"), data_mean)
        np.save(os.path.join(aq_path, "stds.npy"), data_std)

    return (train_data, val_data, test_data), data_in_seq, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv_path", type=str, default="data/air_quality/AirQualityUCI.csv")
    parser.add_argument("-data_path", type=str, default="data",
                        help="data folder to upload and save weather dataset.")
    parser.add_argument("-TRAIN_SPLIT", type=float, default=0.7,
                        help="train split for spliting between train and validation sets.")
    parser.add_argument("-history", type=int, default=13, help="history of past observations.")
    args = parser.parse_args()

    df, array = preprocess_dataframe(args.csv_path, args.data_path)
    (train_data, val_data, test_data), data_in_seq, (mean, std) = split_air_quality_dataset(data=array, TRAIN_SPLIT=args.TRAIN_SPLIT, save_path=args.data_path)
    print("train data shape", train_data.shape)
    print("val data shape", val_data.shape)
    print("test data shape", test_data.shape)
    print("stats - mean:", mean)
    print("stats - std:", std)


