from datasets import load_from_disk
import matplotlib.pyplot as plt
from pprint import pprint
from transformers import GPT2Tokenizer
from src.data_provider.sst_tokenizer import SSTTokenizer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import os
import tensorflow as tf
import numpy as np


class SSTDataset():
    def __init__(self, tokenizer, batch_size=32, max_seq_len=51, max_samples=None):
        self.tokenizer = tokenizer
        self.output_size = self.get_len_vocab()
        self.PAD_IDX = self.get_PAD_IDX()
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.max_seq_len = max_seq_len

    def get_len_vocab(self):
        if self.tokenizer.__class__ == GPT2Tokenizer:
            len_vocab = len(self.tokenizer.decoder)
        elif self.tokenizer.__class__ == SSTTokenizer:
            len_vocab = len(self.tokenizer.vocab)
        return len_vocab

    def get_PAD_IDX(self):
        if self.tokenizer.__class__ == GPT2Tokenizer:
            PAD_IDX = 50256
        elif self.tokenizer.__class__ == SSTTokenizer:
            PAD_IDX = 0
        return PAD_IDX

    def visualize_labels(self, dataset, out_path="src/statistics", split="train"):
        plt.hist(dataset['label'], 30, density=True, facecolor='g', alpha=0.75)
        plt.savefig(os.path.join(out_path, "sst_{}_dataset_label_distribution.png").format(split))

    def get_frequency_tokens(self, dataset):
        token = word_tokenize(" ".join(dataset["sentence"]))
        fdist = FreqDist(token)
        return fdist

    def plot_most_frequent_words(self, dataset, num_words=50, out_path="src/statistics", split="train"):
        fdist = self.get_frequency_tokens(dataset)
        fdist1 = fdist.most_common(num_words)
        fdist1_dict = {key: value for key, value in fdist1}
        plt.figure(figsize=(num_words, 10))
        plt.title("Most common words")
        plt.bar(fdist1_dict.keys(), fdist1_dict.values())
        plt.tick_params(labelsize=24)
        plt.savefig(os.path.join(out_path, "sst_{}_dataset_most_common_words.png").format(split))

    def get_len_reviews(self, dataset):
        def len_review(example):
            example["len_sentence"] = len(word_tokenize(example["sentence"]))
            return example
        dataset_with_len = dataset.map(len_review)
        return dataset_with_len

    def plot_len_reviews(self, dataset, out_path="src/statistics", split="train"):
        dataset = self.get_len_reviews(dataset)
        plt.figure(figsize=(20, 10))
        plt.title("reviews size distribution")
        plt.hist(dataset["len_sentence"], bins=30)
        plt.savefig(os.path.join(out_path, "sst_{}_dataset_review_size_distribution.png".format(split)))

    def get_datasets(self):
        train_set = load_from_disk("data/sst/train")
        val_set = load_from_disk("data/sst/val")
        test_set = load_from_disk("data/sst/test")
        train_set = self.preprocess_dataset(train_set)
        val_set = self.preprocess_dataset(val_set)
        test_set = self.preprocess_dataset(test_set)
        return train_set, val_set, test_set

    def tokenize(self, dataset):
        def tokenize_example(example):
            example["input_ids"] = self.tokenizer.encode(example['sentence'])
            example["attention_mask"] = [1] * len(example["input_ids"])
            return example
        if self.tokenizer.__class__ == GPT2Tokenizer:
            encoded_dataset = dataset.map(lambda example: self.tokenizer(example['sentence']), batched=True)
        elif self.tokenizer.__class__ == SSTTokenizer:
            encoded_dataset = dataset.map(tokenize_example)
        print("encoded_dataset[0]")
        pprint(encoded_dataset[0], compact=True)
        return encoded_dataset

    def get_input_target_sequences(self, dataset):
        def split_input_target(example):
            example['target_ids'] = example['input_ids'][1:] #TODO: add SOS Token ?
            example['input_ids'] = example['input_ids'][:-1] #TODO: add EOS Token?
            return example
        processed_dataset = dataset.map(split_input_target)
        return processed_dataset

    def get_tf_dataset(self, dataset, batch_size, num_dim=4, num_dim_targets=None):
        features = {x: tf.keras.preprocessing.sequence.pad_sequences(dataset[x], padding="post",
                                                                     truncating="post", maxlen=self.max_seq_len, value=self.PAD_IDX) for x in ['input_ids', 'target_ids']}

        if num_dim == 4:
            features_ = {k:v[:, np.newaxis, :, np.newaxis] for k,v in features.items()}
            self.seq_len = features_["input_ids"].shape[-2]
        elif num_dim == 3:
            features_ = {k: v[:, :, np.newaxis] for k, v in features.items()}
            self.seq_len = features_["input_ids"].shape[-2]
        elif num_dim == 2:
            features_ = features
            self.seq_len = features_["input_ids"].shape[1]
        if num_dim_targets is not None:
            if num_dim_targets == 4:
                features_["target_ids"] = features["target_ids"][:, np.newaxis, :, np.newaxis]
            elif num_dim_targets == 3:
                features_["target_ids"] = features["target_ids"][:, :, np.newaxis]
            elif num_dim_targets == 2:
                features_["target_ids"] = features["target_ids"]
        if self.max_samples is not None:
            features_ = {k: v[:self.max_samples] for k, v in features_.items()}
            print("reducing train dataset to {} samples".format(self.max_samples))

        if self.tokenizer.__class__ == GPT2Tokenizer:
            tfdataset = tf.data.Dataset.from_tensor_slices((features_["input_ids"], features_["target_ids"], features_["attention_mask"]))
        elif self.tokenizer.__class__ == SSTTokenizer:
            tfdataset = tf.data.Dataset.from_tensor_slices(
                (features_["input_ids"], features_["target_ids"], None))
        tfdataloader = tfdataset.batch(batch_size=batch_size, drop_remainder=True)
        next(iter(tfdataset))
        return tfdataset, tfdataloader

    def data_to_dataset(self, train_data, val_data, test_data, num_dim=4, num_dim_targets=None):
        _, train_dataset = self.get_tf_dataset(dataset=train_data, batch_size=self.batch_size, num_dim=num_dim, num_dim_targets=num_dim_targets)
        _, val_dataset = self.get_tf_dataset(dataset=val_data, batch_size=self.batch_size,  num_dim=num_dim, num_dim_targets=num_dim_targets)
        _, test_dataset = self.get_tf_dataset(dataset=test_data, batch_size=1, num_dim=num_dim, num_dim_targets=num_dim_targets)
        return train_dataset, val_dataset, test_dataset

    def preprocess_dataset(self, dataset):
        dataset = self.tokenize(dataset)
        dataset = self.get_input_target_sequences(dataset)
        return dataset

    def check_number_unk_tokens(self, dataset):
        inputs_ids = dataset["input_ids"]
        num_unk = 0
        num_tokens = 0
        for inp in inputs_ids:
            len_ = len(inp)
            num_tokens += len_
            mask = inp == 1
            unk = mask.sum().item()
            num_unk += unk
        return num_unk / num_tokens, num_unk

    def check_dataset(self, dataset):
        for (inp, tar, _) in dataset.take(1):
            if len(inp.shape) == len(tar.shape):
                if inp.shape == 4:
                    assert inp[:,:,1:,:] == tar[:,:,:-1,:], "error in inputs/targets of dataset"
                elif inp.shape == 3:
                    assert inp[:, 1:, :] == tar[:, :-1, :], "error in inputs/targets of dataset"



if __name__ == '__main__':
    print("-----------------------------------------------------------------------------")
    print("SST dataset with SST tokenizer")
    dataset = load_from_disk("data/sst/all_data")
    sst_tokenizer = SSTTokenizer(dataset=dataset)
    sst_dataset = SSTDataset(tokenizer=sst_tokenizer)
    train_set, val_set, test_set = sst_dataset.get_datasets()
    train_dataset, val_dataset, test_dataset = sst_dataset.data_to_dataset(train_set, val_set, test_set)
    sst_dataset.check_dataset(train_dataset)
    print("-----------------------------------------------------------------------------")
    print("SST dataset with GPT2 tokenizer")
    tokzer = GPT2Tokenizer.from_pretrained("cache/gpt2")
    dset = SSTDataset(tokenizer=tokzer)
    train_set, val_set, test_set = dset.get_datasets()
    train_dataset, val_dataset, test_dataset = dset.data_to_dataset(train_set, val_set, test_set, num_dim=2, num_dim_targets=4)
    for (inp, tar, attn_mask) in train_dataset.take(1):
        print('inputs', inp.shape)
        print('targets', tar.shape)
        print("attention_mask", attn_mask)
    dset.check_dataset(train_dataset)
    print("done")