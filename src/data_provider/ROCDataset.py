'''
Create a questions Dataset to train the language model.
Inspired from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''
import json
import numpy as np
from src.data_provider.CLEVR_tokenizer import Tokenizer
import tensorflow as tf
import os
import pandas as pd


# TODO: add a max samples here: select 350,000 questions.
class ROCDataset:
    def __init__(self, data_path, batch_size=32, max_samples=None):
        self.data_path = data_path
        self.vocab_path = os.path.join(data_path, "vocab.json")
        self.batch_size = batch_size
        self.vocab = self.get_vocab()
        self.output_size = len(self.vocab)
        self.tokenizer = Tokenizer(self.vocab)
        self.name = "roc"
        self.max_samples = max_samples

    def get_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab

    def get_dataset_elements(self, dataset_path):
        dataset = pd.read_pickle(dataset_path)
        input_sentence = np.array([seq for seq in dataset.input_sentence.values])
        target_sentence = np.array([seq for seq in dataset.target_sentence.values])
        attention_mask = np.array([seq for seq in dataset.attention_mask.values])
        if self.max_samples is not None:
            input_sentence = input_sentence[:self.max_samples]
            target_sentence = target_sentence[:self.max_samples]
        return input_sentence, target_sentence, None

    def get_test_dataset_elements(self, dataset_path):
        dataset = pd.read_pickle(dataset_path)
        input_sentence = dataset.sentence1
        target_sentence = dataset.sentence2
        if self.max_samples is not None:
            input_sentence = input_sentence[:self.max_samples]
            target_sentence = target_sentence[:self.max_samples]
        return input_sentence, target_sentence

    def _reshape(self, array, num_dim=4):
        if num_dim == 4:
            array = array[:, np.newaxis, :, np.newaxis]
        elif num_dim == 3:
            array = array[:, :, np.newaxis]
        return array

    def _get_test_shape(self, element, num_dim=4):
        if num_dim == 4:
            shape = (1, 1, len(element), 1)
        elif num_dim == 3:
            shape = (1, len(element), 1)
        elif num_dim == 2:
            shape = (1, len(element))
        return shape

        # words to remove.
        # "$": 4509, "%": 7129, "&": 534, "'": 823, "''": 9236,

    def get_datasets(self):
        train_data = self.get_dataset_elements(os.path.join(self.data_path, "train_set.pkl"))
        val_data = self.get_dataset_elements(os.path.join(self.data_path, "val_set.pkl"))
        test_data = self.get_test_dataset_elements(os.path.join(self.data_path, "test_set.pkl"))
        return train_data, val_data, test_data

    def get_dataloader(self, data, batch_size, num_dim=4, seq_len=False):
        inputs, targets, attn_mask = data
        if seq_len:
            self.seq_len = inputs.shape[1]
        inputs, targets = self._reshape(inputs, num_dim=num_dim), self._reshape(targets,
                                                                                           num_dim=num_dim)
        tfdataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets, attn_mask))
        tfdataloader = tfdataset.batch(batch_size, drop_remainder=True)
        return tfdataloader

    def get_test_dataloader(self, data, num_dim=4):
        inputs, targets = data
        inputs = inputs.to_list()
        targets = targets.to_list()
        inputs = [tf.constant(inp, dtype=tf.int32, shape=self._get_test_shape(inp, num_dim)) for inp in inputs]
        targets = [tf.constant(tar, dtype=tf.int32, shape=self._get_test_shape(tar, num_dim)) for tar in targets]
        return (inputs, targets)

    def data_to_dataset(self, train_data, val_data, test_data, num_dim=4):
        train_dataset = self.get_dataloader(train_data, self.batch_size, num_dim, seq_len=True)
        val_dataset = self.get_dataloader(val_data, self.batch_size, num_dim)
        test_dataset = self.get_test_dataloader(test_data, num_dim)
        return train_dataset, val_dataset, test_dataset

    def check_dataset(self, dataset):
        for (inp, tar, _) in dataset.take(5):
            if len(inp.shape) == len(tar.shape):
                if inp.shape == 4:
                    assert inp[:, :, 1:, :] == tar[:, :, :-1, :], "error in inputs/targets of dataset"
                elif inp.shape == 3:
                    assert inp[:, 1:, :] == tar[:, :-1, :], "error in inputs/targets of dataset"


if __name__ == '__main__':
    dataset = ROCDataset(data_path='data/ROC')
    train_data, val_data, test_data = dataset.get_datasets()
    train_dataset, val_dataset, test_dataset = dataset.data_to_dataset(train_data, val_data, test_data)
    print("done")
