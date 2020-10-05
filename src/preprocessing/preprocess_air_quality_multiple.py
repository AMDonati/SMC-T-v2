import argparse

from src.preprocessing.utils import *


def preprocess_dataframe(csv_path, save_path=None):
    df_from_each_file = (pd.read_csv(os.path.join(csv_path, f)) for f in os.listdir(csv_path))
    frame = pd.concat(df_from_each_file, ignore_index=True)
    frame['Date'] = pd.to_datetime(frame[['year', 'month', 'day', "hour"]])
    frame.drop(["year", "month", "day", "hour", "No"], inplace=True, axis=1)
    frame.fillna(method="backfill", inplace=True)
    frame = pd.get_dummies(frame, columns=['wd'])
    frame.set_index(["station", "Date"], inplace=True)
    frame = frame.apply(pd.to_numeric, errors='coerce')
    return frame


def split_air_quality_dataset(data, TRAIN_SPLIT, VAL_SPLIT=0.5, save_path=None):
    # normalization
    # columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    # data_mean = data[[columns]].mean(axis=0)
    # data_std = data[[columns]].std(axis=0)
    # data[[columns]] = (data[[columns]] - data_mean) / data_std
    # stats = (data_mean, data_std)

    data_in_seq = []
    for station in data.index.levels[0]:
        station_df = data.loc[[station]]
        seq = split_dataset_into_seq(station_df.values, start_index=0, end_index=None, history_size=13, step=1)
        data_in_seq.extend(seq)
    data_in_seq = np.array(data_in_seq)

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
        np.save(os.path.join(train_path, "air_quality2.npy"), train_data)
        np.save(os.path.join(val_path, "air_quality2.npy"), val_data)
        np.save(os.path.join(test_path, "air_quality2.npy"), test_data)
        #np.save(os.path.join(aq_path, "means2.npy"), data_mean)
        #np.save(os.path.join(aq_path, "stds2.npy"), data_std)

    return (train_data, val_data, test_data), data_in_seq, (None,None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv_path", type=str, default="data/air_quality/PRSA_Data_20130301-20170228")
    parser.add_argument("-data_path", type=str, default="data",
                        help="data folder to upload and save weather dataset.")
    parser.add_argument("-TRAIN_SPLIT", type=float, default=0.7,
                        help="train split for spliting between train and validation sets.")
    parser.add_argument("-history", type=int, default=13, help="history of past observations.")
    args = parser.parse_args()

    df = preprocess_dataframe(args.csv_path, args.data_path)
    (train_data, val_data, test_data), data_in_seq, (mean, std) = split_air_quality_dataset(data=df,
                                                                                            TRAIN_SPLIT=args.TRAIN_SPLIT,
                                                                                            save_path=args.data_path)
    print("train data shape", train_data.shape)
    print("val data shape", val_data.shape)
    print("test data shape", test_data.shape)
    print("stats - mean:", mean)
    print("stats - std:", std)
