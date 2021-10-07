'''
Create a questions Dataset to train the language model.
Inspired from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''
import json
import h5py
import numpy as np
from src.data_provider.text_functions import decode
from src.data_provider.CLEVR_tokenizer import  Tokenizer
import tensorflow as tf
import os


# TODO: add a max samples here: select 350,000 questions.
class QuestionsDataset():
    def __init__(self, data_path, batch_size=32, max_samples=None, max_seq_len=30):
        self.data_path = data_path
        self.vocab_path = os.path.join(data_path, "vocab.json")
        self.max_samples = max_samples
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_questions = self.get_vocab()
        self.idx_to_token = self.get_idx_to_token()
        self.output_size = len(self.vocab_questions)
        self.tokenizer = Tokenizer(self.vocab_questions)
        self.seq_len = max_seq_len

    def get_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)['question_token_to_idx']
        return vocab

    def get_idx_to_token(self):
        idx_to_token = dict(zip(list(self.vocab_questions.values()), list(self.vocab_questions.keys())))
        return idx_to_token

    def idx2word(self, seq_idx, delim=' ', stop_at_end=False):
        tokens = decode(seq_idx=seq_idx, idx_to_token=self.idx_to_token, stop_at_end=stop_at_end, delim=delim)
        return tokens

    def get_questions(self, dataset_path):
        hf = h5py.File(dataset_path, 'r')
        input_questions = hf.get('input_questions')
        input_questions = np.array(input_questions, dtype=np.int32)
        target_questions = hf.get('target_questions')
        target_questions = np.array(target_questions, dtype=np.int32)
        input_questions = tf.constant(input_questions, dtype=tf.int32)  # shape (num_samples, seq_len)
        target_questions = tf.constant(target_questions, dtype=tf.int32)
        return (input_questions, target_questions) # dim (B,S)

    def _reshape_tensor(self, tensor, num_dim=4):
        if num_dim == 4:
            tensor = tf.reshape(tensor, shape=(tensor.shape[0], 1, tensor.shape[1], 1))
        elif num_dim == 3:
            tensor = tf.reshape(tensor, shape=(tensor.shape[0], tensor.shape[1], 1))
        return tensor

    def _reshape(self, array, num_dim=4):
        if num_dim == 4:
            array = array[:, np.newaxis, :, np.newaxis]
        elif num_dim == 3:
            array = array[:, :, np.newaxis]
        return array

    def get_sequence_lengths(self, questions):
        lengths = tf.math.count_nonzero(questions, axis=1)
        return lengths

    def filter_max_length(self, questions):
        lengths = self.get_sequence_lengths(questions)
        mask = lengths <= self.max_seq_len
        filtered_questions = tf.boolean_mask(questions, mask, axis=0)
        filtered_questions = filtered_questions[:self.max_samples, :self.max_seq_len]
        return filtered_questions

    def get_tf_dataset(self, questions, batch_size, num_dim=4):
        input_questions, target_questions = questions
        input_questions = self.filter_max_length(input_questions)
        target_questions = self.filter_max_length(target_questions)
        input_questions = self._reshape_tensor(input_questions, num_dim=num_dim)
        target_questions = self._reshape_tensor(target_questions, num_dim=num_dim)
        tfdataset = tf.data.Dataset.from_tensor_slices(
            (input_questions, target_questions, None))
        tfdataloader = tfdataset.batch(batch_size, drop_remainder=True)
        print(next(iter(tfdataset)))
        return tfdataloader

    def get_datasets(self):
        train_data = self.get_questions(os.path.join(self.data_path, "train_questions.h5"))
        val_data = self.get_questions(os.path.join(self.data_path, "val_questions.h5"))
        test_data = self.get_questions(os.path.join(self.data_path, "test_questions.h5"))
        return train_data, val_data, test_data

    def data_to_dataset(self, train_data, val_data, test_data, num_dim=4, num_dim_targets=None):
        train_dataset = self.get_tf_dataset(train_data, self.batch_size, num_dim)
        val_dataset = self.get_tf_dataset(val_data, self.batch_size, num_dim)
        test_dataset = self.get_tf_dataset(test_data, 1,  num_dim)
        return train_dataset, val_dataset, test_dataset

    def check_dataset(self, dataset):
        for (inp, tar, _) in dataset.take(5):
            if len(inp.shape) == len(tar.shape):
                if inp.shape == 4:
                    assert inp[:,:,1:,:] == tar[:,:,:-1,:], "error in inputs/targets of dataset"
                elif inp.shape == 3:
                    assert inp[:, 1:, :] == tar[:, :-1, :], "error in inputs/targets of dataset"



if __name__ == '__main__':
    train_dataset = QuestionsDataset(data_path="data/clevr", max_samples=100)
    train_data, val_data, test_data = train_dataset.get_datasets()
    train_dataset, val_dataset, test_dataset = train_dataset.data_to_dataset(train_data, val_data, test_data)
    print("done")