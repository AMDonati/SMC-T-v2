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
    def __init__(self, data_path, batch_size=32):
        self.data_path = data_path
        self.vocab_path = os.path.join(data_path, "vocab.json")
        self.batch_size = batch_size
        self.vocab = self.get_vocab()
        self.output_size = len(self.vocab)
        self.tokenizer = Tokenizer(self.vocab)

    def get_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab

    def get_dataset_elements(self, dataset_path):
        dataset = pd.read_pickle(dataset_path)
        input_sentence = np.array([seq for seq in dataset.input_sentence.values])
        target_sentence = np.array([seq for seq in dataset.target_sentence.values])
        attention_mask = np.array([seq for seq in dataset.attention_mask.values])
        return input_sentence, target_sentence, attention_mask

    def _reshape(self, array, num_dim=4):
        if num_dim == 4:
            array = array[:, np.newaxis, :, np.newaxis]
        elif num_dim == 3:
            array = array[:, :, np.newaxis]
        return array

        # words to remove.
        # "$": 4509, "%": 7129, "&": 534, "'": 823, "''": 9236,

    def get_datasets(self):
        train_data = self.get_dataset_elements(os.path.join(self.data_path, "train_set.pkl"))
        val_data = self.get_dataset_elements(os.path.join(self.data_path, "val_set.pkl"))
        test_data = self.get_dataset_elements(os.path.join(self.data_path, "test_set.pkl"))
        return train_data, val_data, test_data

    def get_dataloader(self, data, batch_size, num_dim=4):
        inputs, targets, attn_mask = data
        self.seq_len = inputs.shape[1]
        inputs, targets, attn_mask = self._reshape(inputs, num_dim=num_dim), self._reshape(targets,
                                                                                           num_dim=num_dim), self._reshape(
            attn_mask, num_dim=num_dim)
        tfdataset = tf.data.Dataset.from_tensor_slices(
            (inputs, targets, attn_mask))
        tfdataloader = tfdataset.batch(batch_size, drop_remainder=True)
        return tfdataloader

    def data_to_dataset(self, train_data, val_data, test_data, num_dim=4):
        train_dataset = self.get_dataloader(train_data, self.batch_size, num_dim)
        val_dataset = self.get_dataloader(val_data, self.batch_size, num_dim)
        test_dataset = self.get_dataloader(test_data, 1, num_dim)
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
