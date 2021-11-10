'''
Create a questions Dataset to train the language model.
Inspired from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''
import json
import numpy as np
from nltk import word_tokenize
import pandas as pd
import re


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def get_sentences(df, max_samples=20000):
    df["sentence_1_2"] = df.sentence1 + " " + df.sentence2
    sentences = df.sentence_1_2[:max_samples]
    return sentences, df["sentence1"][:max_samples], df["sentence2"][:max_samples]


def split_train_test(sentences, sentences_1_and_2, train_size=10000, val_size=5000, test_size=5000):
    train_sentences = sentences[:train_size]
    val_sentences = sentences[train_size:train_size + val_size]
    test_sentences = sentences_1_and_2[train_size + val_size:train_size + val_size + test_size]
    paths = ["data/ROC/train_set.pkl", "data/ROC/val_set.pkl", "data/ROC/test_set.pkl"]
    for df, path in zip([train_sentences, val_sentences, test_sentences], paths):
        df.to_pickle(path)
    print("saving dataset splits in pkl files...")
    # train_data = split_dataset_elements(train_sentences)
    # val_data = split_dataset_elements(val_sentences)
    # test_data = split_dataset_elements(test_sentences)
    return train_sentences, val_sentences, test_sentences

def tokenize_test(sentences, vocab):
    tokenize_func = lambda t: word_tokenize(t)
    tok_to_id_func = lambda t: [vocab[w] for w in t if w in vocab.keys()]
    tokenized_sentences = sentences.apply(tokenize_func)
    tokens_id = tokenized_sentences.apply(tok_to_id_func)
    len_sentences = tokens_id.apply(len)
    return tokens_id, len_sentences

def tokenize(sentences, vocab):
    tokenize_func = lambda t: word_tokenize(t)
    tok_to_id_func = lambda t: [vocab[w] for w in t if w in vocab.keys()]
    tokenized_sentences = sentences.apply(tokenize_func)
    tokens_id = tokenized_sentences.apply(tok_to_id_func)
    df = pd.DataFrame()
    df["input_sentence"] = tokens_id.apply(lambda t: t[:-1])
    df["target_sentence"] = tokens_id.apply(lambda t: t[1:])
    len_sentences = tokens_id.apply(len)
    max_len = len_sentences.max() - 1
    pad_func = lambda t: t + [0] * (max_len - len(t))
    df["input_sentence"] = df.input_sentence.apply(pad_func)
    df["target_sentence"] = df.target_sentence.apply(pad_func)
    df["attention_mask"] = len_sentences.apply(lambda t: [1] * (t - 1) + [0] * (max_len - (t - 1)))
    print("max len", max_len)
    return df, len_sentences


def clean_text(sentences):
    clean_func1 = lambda t: ' '.join(t.split("-"))
    clean_func2 = lambda t: ' '.join(re.split(r"([0-9]+)([a-z]+)", t, re.I))
    clean_func3 = lambda t: ' '.join(re.split(r"([a-z]+)([0-9]+)", t, re.I))
    clean_func4 = lambda t: t.lower().replace("&", "and")
    sentences = sentences.apply(clean_func1)
    sentences = sentences.apply(clean_func2)
    sentences = sentences.apply(clean_func3)
    sentences = sentences.apply(clean_func4)
    return sentences


def get_vocab(sentences, tokens_to_remove=["$", "%", "'", "''"]):
    print("Building vocab....")
    tokenize_func = lambda t: word_tokenize(t.lower())
    # tokens = word_tokenize(' '.join(sentences))
    tokenized_sentences = sentences.apply(tokenize_func)
    tokenized_sentences = tokenized_sentences.values
    tokens = [w for s in tokenized_sentences for w in s]
    unique_tokens = list(set(tokens))
    for token in tokens_to_remove:
        unique_tokens.remove(token)
    unique_tokens.sort()
    vocab = {v: k for k, v in enumerate(unique_tokens)}
    # TODO: remove indesirable tokens.
    print("vocab length:", len(vocab))
    print("saving vocab...")
    with open("data/ROC/vocab.json", "w") as f:
        json.dump(vocab, f)
    return tokens, vocab

    # words to remove.
    # "$": 4509, "%": 7129, "&": 534, "'": 823, "''": 9236,


def preprocess_data(data_path):
    df = load_data(data_path)
    sentences, sentences_1, sentences_2 = get_sentences(df)
    sentences, sentences_1, sentences_2 = clean_text(sentences), clean_text(sentences_1), clean_text(sentences_2)
    tokens, vocab = get_vocab(sentences)
    padded_sentences, len_sentences = tokenize(sentences, vocab)
    sentences_1, len_sentences_1 = tokenize_test(sentences_1, vocab)
    sentences_2, len_sentences_2 = tokenize_test(sentences_2, vocab)
    sentences_1_and_2 = pd.concat([sentences_1, sentences_2], axis=1)
    train_sentences, val_sentences, test_sentences = split_train_test(padded_sentences, sentences_1_and_2)
    return train_sentences, val_sentences, test_sentences


if __name__ == '__main__':
    data_path = "data/ROC/ROCStories_winter2017.csv"
    train_sentences, val_sentences, test_sentences = preprocess_data(data_path)
    print("done")
