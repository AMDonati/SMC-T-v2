import os
import numpy as np
import tensorflow as tf
from src.preprocessing.utils import split_synthetic_dataset
from sklearn.preprocessing import StandardScaler
import h5py

class Dataset:
    def __init__(self, data_path, BATCH_SIZE=32, name="synthetic", model=None, BUFFER_SIZE=500, max_size_test=3000, max_samples=None):
        self.data_path = data_path
        self.data_arr = np.load(os.path.join(data_path, "raw_data.npy")) if os.path.exists(os.path.join(data_path, "raw_data.npy")) else None
        self.train_path = os.path.join(data_path, "train")
        self.val_path = os.path.join(data_path, "val")
        self.test_path = os.path.join(data_path, "test")
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.name = name
        self.max_samples = max_samples
        self.max_size_test = max_size_test
        self.model = model

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
        type = np.int32
        train_data = self.get_data_from_folder(self.train_path)
        train_data = train_data.astype(type)
        val_data = self.get_data_from_folder(self.val_path)
        val_data = val_data.astype(type)
        test_data = self.get_data_from_folder(self.test_path)
        if self.max_samples is not None:
            if train_data.shape[0] > self.max_samples:
                train_data = train_data[:self.max_samples] # to avoid memory issues at test time.
                print("reducing train dataset size to {} samples...".format(self.max_samples))
        if test_data.shape[0] > self.max_size_test:
            test_data = test_data[:self.max_size_test] # to avoid memory issues at test time.
            print("reducing test dataset size to {} samples...".format(self.max_size_test))
        test_data = test_data.astype(type)
        return train_data, val_data, test_data

    def get_features_labels(self, train_data, val_data, test_data):
        x_train, y_train = self.split_fn(train_data)
        x_val, y_val = self.split_fn(val_data)
        x_test, y_test = self.split_fn(test_data)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def data_to_dataset(self, train_data, val_data, test_data, num_dim=4):
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

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.get_features_labels(train_data=train_data, val_data=val_data, test_data=test_data)

        if num_dim == 4:
            # adding the particle dim:
            x_train = x_train[:, np.newaxis, :, :]  # (B,P,S,F)
            y_train = y_train[:, np.newaxis, :, :]
            x_val = x_val[:, np.newaxis, :, :]
            y_val = y_val[:, np.newaxis, :, :]
            x_test = x_test[:, np.newaxis, :, :]
            y_test = y_test[:, np.newaxis, :, :]

        train_tuple = (x_train, y_train)
        val_tuple = (x_val, y_val)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_tuple)
        train_dataset = train_dataset.cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_tuple)
        val_dataset = val_dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        BATCH_SIZE_test = self.BATCH_SIZE
        test_tuple = (x_test, y_test)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_tuple) #TODO: could use from tensor instead.
        test_dataset = test_dataset.batch(BATCH_SIZE_test, drop_remainder=True)

        return train_dataset, val_dataset, test_dataset

    def check_dataset(self, dataset):
        for (inp, tar) in dataset.take(1):
            if inp.shape == 4:
                assert inp[:,:,1:,:] == tar[:,:,:-1,:], "error in inputs/targets of dataset"
            elif inp.shape == 3:
                assert inp[:, 1:, :] == tar[:, :-1, :], "error in inputs/targets of dataset"

    def get_data_splits_for_crossvalidation(self, TRAIN_SPLIT=0.8, VAL_SPLIT_cv=0.9):
        list_train_data, list_val_data, test_data = split_synthetic_dataset(x_data=self.data_arr,
                                                                            TRAIN_SPLIT=TRAIN_SPLIT,
                                                                            VAL_SPLIT_cv=VAL_SPLIT_cv, cv=True)
        list_test_data = [test_data] * len(list_train_data)
        return list_train_data, list_val_data, list_test_data

    def get_datasets_for_crossvalidation(self, TRAIN_SPLIT=0.8, VAL_SPLIT_cv=0.9, num_dim=4):
        list_train_data, list_val_data, list_test_data = self.get_data_splits_for_crossvalidation(TRAIN_SPLIT=TRAIN_SPLIT, VAL_SPLIT_cv=VAL_SPLIT_cv)
        train_datasets, val_datasets, test_datasets = [], [], []
        for train_data, val_data, test_data in zip(list_train_data, list_val_data, list_test_data):
            train_dataset, val_dataset, test_dataset = self.data_to_dataset(train_data=train_data, val_data=val_data, test_data=test_data, num_dim=num_dim)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        return train_datasets, val_datasets, test_datasets[0]

if __name__ == '__main__':

    synthetic_dataset = Dataset(data_path='../../data/dummy_model_nlp', BUFFER_SIZE=50, BATCH_SIZE=64)
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
    synthetic_dataset.check_dataset(train_dataset)
    for (inp, tar) in train_dataset.take(1):
        print('input example shape', inp.shape)
        print('input example', inp[0])
        print('target example shape', tar.shape)
        print('target example', tar[0])





