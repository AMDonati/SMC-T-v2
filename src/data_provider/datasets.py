import os
import numpy as np
import tensorflow as tf
from src.preprocessing.utils import split_synthetic_dataset, split_covid_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, data_path, BUFFER_SIZE, BATCH_SIZE):
        self.data_path = data_path
        self.data_arr = self.get_data_from_folder(self.data_path)
        self.train_path = os.path.join(data_path, "train")
        self.val_path = os.path.join(data_path, "val")
        self.test_path = os.path.join(data_path, "test")
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE

    def split_fn(self, chunk):
        input_text = chunk[:, :-1, :]
        target_text = chunk[:, 1:, :]
        return input_text, target_text

    def get_data_from_folder(self, folder_path, extension=".npy"):
        files = []
        for file in os.listdir(folder_path):
            if file.endswith(extension):
                files.append(file)
        file_path = os.path.join(folder_path, files[0])
        data = np.load(file_path)
        return data

    def get_data_sample_from_index(self, index):
        pass

    def get_datasets(self):
        train_data = self.get_data_from_folder(self.train_path)
        val_data = self.get_data_from_folder(self.val_path)
        test_data = self.get_data_from_folder(self.test_path)
        return train_data, val_data, test_data

    def data_to_dataset(self, train_data, val_data, test_data, target_feature=None, num_dim=4, with_lengths=False):
        '''
        :param train_data: input data for training > shape (N_train, S+1, F) ; N_train = number of samples in training dataset.
        :param val_data: input data used for validation set > shape (N_val, S+1, F)
        :param: cv: boolean; True if multiple train datasets / val datasets for cross-validation; False otherwise.
        :param target_feature: used to select the target feature to be predicted. Case of multivariate ts as input data > prediction of a univariate ts.
        :return:
        2 tf.data.Dataset, one for the training set, and one for the validation set, with:
        input data:  batches of train data > shape (B, S+1, F) > S+1 because the data is split in the SMC_Transformer.Py script.
        target data: shape (B,S,1) > univariate ts to be predicted (shifted from one timestep compared to the input data).
        '''
        x_train, y_train = self.split_fn(train_data)
        x_val, y_val = self.split_fn(val_data)
        x_test, y_test = self.split_fn(test_data)
        if with_lengths:
            lengths_train = x_train.shape[1] * np.ones(
                shape=x_train.shape[0])  # tensor with value seq_len of size batch_size.
            lengths_val = x_val.shape[1] * np.ones(shape=x_val.shape[0])
            lengths_test = x_test.shape[1] * np.ones(x_test.shape[0])

        if target_feature is not None:
            y_train = y_train[:, :, target_feature]
            y_train = np.reshape(y_train, newshape=(y_train.shape[0], y_train.shape[1], 1))
            y_val = y_val[:, :, target_feature]
            y_val = np.reshape(y_val, newshape=(y_val.shape[0], y_val.shape[1], 1))
            y_test = y_test[:, :, target_feature]
            y_test = np.reshape(y_test, newshape=(y_test.shape[0], y_test.shape[1], 1))
        if num_dim == 4:
            # adding the particle dim:
            x_train = x_train[:, np.newaxis, :, :]  # (B,P,S,F)
            y_train = y_train[:, np.newaxis, :, :]
            x_val = x_val[:, np.newaxis, :, :]
            y_val = y_val[:, np.newaxis, :, :]
            x_test = x_test[:, np.newaxis, :, :]
            y_test = y_test[:, np.newaxis, :, :]

        train_tuple = (x_train, y_train) if not with_lengths else (x_train, y_train, lengths_train)
        val_tuple = (x_val, y_val) if not with_lengths else (x_val, y_val, lengths_val)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_tuple)
        train_dataset = train_dataset.cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_tuple)
        val_dataset = val_dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        BATCH_SIZE_test = test_data.shape[0] if not with_lengths else self.BATCH_SIZE
        test_tuple = (x_test, y_test) if not with_lengths else (x_test, y_test, lengths_test)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_tuple) #TODO: could use from tensor instead.
        test_dataset = test_dataset.batch(BATCH_SIZE_test)

        return train_dataset, val_dataset, test_dataset

    def check_dataset(self, dataset):
        for (inp, tar) in dataset.take(1):
            if inp.shape == 4:
                assert inp[:,:,1:,:] == tar[:,:,:-1,:], "error in inputs/targets of dataset"
            elif inp.shape == 3:
                assert inp[:, 1:, :] == tar[:, :-1, :], "error in inputs/targets of dataset"

    def get_datasets_for_crossvalidation(self, TRAIN_SPLIT=0.8, VAL_SPLIT_cv=0.9, num_dim=4, target_feature=None):
        list_train_data, list_val_data, test_data = split_synthetic_dataset(x_data=self.data_arr, TRAIN_SPLIT=TRAIN_SPLIT, VAL_SPLIT_cv=VAL_SPLIT_cv, cv=True)
        list_test_data = [test_data] * len(list_train_data)
        train_datasets, val_datasets, test_datasets = [], [], []
        for train_data, val_data, test_data in zip(list_train_data, list_val_data, list_test_data):
            train_dataset, val_dataset, test_dataset = self.data_to_dataset(train_data=train_data, val_data=val_data, test_data=test_data, target_feature=target_feature, num_dim=num_dim)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        return train_datasets, val_datasets, test_datasets[0]

    def prepare_dataset_for_FIVO(self, train_data, val_data, test_data, split="train"):
        train_mean = np.mean(train_data)
        train_mean = tf.constant([train_mean], dtype=tf.float32)
        #val_mean = np.mean(val_data)
        #val_mean = tf.constant([val_mean], dtype=tf.float32)
        #test_mean = np.mean(test_data)
        #test_mean = tf.constant([test_mean], dtype=tf.float32)
        train_dataset, val_dataset, test_dataset = self.data_to_dataset(train_data=train_data, val_data=val_data, test_data=test_data, num_dim=3, with_lengths=True)
        if split == "train":
            dataset = train_dataset
            #mean = train_mean
        elif split == "val":
            dataset = val_dataset
            #mean = val_mean
        elif split == "test":
            dataset = test_dataset
            #mean = test_mean
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        inputs, targets, lengths = iterator.get_next()
        inputs = tf.transpose(inputs, perm=[1,0,2])
        targets = tf.transpose(targets, perm=[1,0,2])
        lengths = tf.cast(lengths, dtype=tf.int32)
        return inputs, targets, lengths, train_mean

class CovidDataset(Dataset):
    def __init__(self, data_path, BATCH_SIZE, BUFFER_SIZE=50):
        super(CovidDataset, self).__init__(data_path=data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)
        #TODO: have a stats npy file as well.

    def rescale_covid_data(self, data_sample, stats, index):
        data_mean, data_std = stats
        mean, std = data_mean[index], data_std[index]
        data_sample = std * data_sample + mean
        data_sample = data_sample.astype(np.int32)
        return data_sample

    def get_data_sample_from_index(self, index, past_len, num_dim=4):
        _, _, test_data = self.get_datasets()
        test_sample = test_data[index]
        print('test_sample', test_sample)
        test_sample = tf.expand_dims(tf.convert_to_tensor(test_sample), axis=0)
        inputs, targets = self.split_fn(test_sample[:, :past_len+1, :])
        if num_dim == 4: # adding the particle dimension.
            inputs = tf.expand_dims(inputs, axis=1)
            targets = tf.expand_dims(targets, axis=1)
            test_sample = tf.expand_dims(test_sample, axis=1)
        return inputs, targets, test_sample

if __name__ == '__main__':
    synthetic_dataset = Dataset(data_path='../../data/synthetic_model_1', BUFFER_SIZE=50, BATCH_SIZE=64)
    train_data, val_data, test_data = synthetic_dataset.get_datasets()
    print('train data shape', train_data.shape)
    print('val data shape', val_data.shape)
    print('test data shape', test_data.shape)

    # ----------------------------------------------- test of data_to_dataset function ----------------------------------
    train_dataset, val_dataset, test_dataset = synthetic_dataset.data_to_dataset(train_data=train_data,
                                                                                 val_data=val_data, test_data=test_data)
    for (inp, tar) in train_dataset.take(1):
        print('input example shape', inp.shape)
        print('input example', inp[0])
        print('target example shape', tar.shape)
        print('target example', tar[0])

    print("3D dataset........")

    train_dataset, val_dataset, test_dataset = synthetic_dataset.data_to_dataset(train_data=train_data,
                                                                                 val_data=val_data, test_data=test_data,
                                                                                 num_dim=3)
    for (inp, tar) in train_dataset.take(1):
        print('input example shape', inp.shape)
        print('input example', inp[0])
        print('target example shape', tar.shape)
        print('target example', tar[0])

    # ------------------------------------------------test prepare_dataset_for_FIVO--------------------------------------
    inputs, targets, lengths, train_mean = synthetic_dataset.prepare_dataset_for_FIVO(train_data=train_data, val_data=val_data, test_data=test_data)
    print("inputs shape", inputs.shape)
    print("targets shape", targets.shape)
    print("lengths shape", lengths.shape)
    print("lenghts", lengths)
    print("train_mean", train_mean)


