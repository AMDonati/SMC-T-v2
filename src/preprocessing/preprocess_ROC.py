'''
Create a questions Dataset to train the language model.
Inspired from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''
import json
import numpy as np
from nltk import word_tokenize
import pandas as pd
import re
from transformers import GPT2Tokenizer

gpt_tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")


def get_PAD_IDX(tokenizer):
    if tokenizer.__class__ == GPT2Tokenizer:
        PAD_IDX = 50256
    else:
        PAD_IDX = 0
    return PAD_IDX


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def get_sentences(df, max_samples=20000):
    df["sentence_1_2"] = df.sentence1 + " " + df.sentence2
    sentences = df.sentence_1_2[:max_samples]
    return sentences, df["sentence1"][:max_samples], df["sentence2"][:max_samples]


def split_train_test(sentences, sentences_1_and_2, train_size=10000, val_size=3000, test_size=5000, gpt_tok=False):
    train_sentences = sentences[:train_size]
    val_sentences = sentences[train_size:train_size + val_size]
    test_sentences = sentences_1_and_2[train_size + val_size:train_size + val_size + test_size]
    if gpt_tok:
        paths = ["data/ROC/gpt2_tok/train_set.pkl", "data/ROC/gpt2_tok/val_set.pkl", "data/ROC/gpt2_tok/test_set.pkl"]
    else:
        paths = ["data/ROC/train_set.pkl", "data/ROC/val_set.pkl", "data/ROC/test_set.pkl"]
    for df, path in zip([train_sentences, val_sentences, test_sentences], paths):
        df.to_pickle(path)
    print("saving dataset splits in pkl files...")
    return train_sentences, val_sentences, test_sentences

def _tokenize(sentences, vocab):
    tokenize_func = lambda t: word_tokenize(t)
    tok_to_id_func = lambda t: [vocab[w] for w in t if w in vocab.keys()]
    tokenized_sentences = sentences.apply(tokenize_func)
    tokens_id = tokenized_sentences.apply(tok_to_id_func)
    return tokens_id

def _tokenize_gpt(sentences):
    tokenize_func = lambda t: gpt_tokenizer(t)["input_ids"]
    tokens_id = sentences.apply(tokenize_func)
    return tokens_id

def tokenize_test(sentences, vocab, gpt_tok=False):
    if gpt_tok:
        tokens_id = _tokenize_gpt(sentences)
    else:
        tokens_id = _tokenize(sentences, vocab)
    len_sentences = tokens_id.apply(len)
    return tokens_id, len_sentences

def tokenize(sentences, vocab, max_len=20, gpt_tok=False):
    if gpt_tok:
        tokens_id = _tokenize_gpt(sentences)
    else:
        tokens_id = _tokenize(sentences, vocab)
    df = pd.DataFrame()
    df["input_sentence"] = tokens_id.apply(lambda t: t[:-1])
    df["target_sentence"] = tokens_id.apply(lambda t: t[1:])
    len_sentences = tokens_id.apply(len)
    # filtering sentence of length max_len max.
    df = df[len_sentences <= (max_len + 1)]
    len_sentences_ = len_sentences[len_sentences <= (max_len + 1)]
    tokenizer = gpt_tokenizer if gpt_tok else None
    PAD_IDX = get_PAD_IDX(tokenizer)
    pad_func = lambda t: t + [PAD_IDX] * (max_len - len(t))
    df["input_sentence"] = df.input_sentence.apply(pad_func)
    df["target_sentence"] = df.target_sentence.apply(pad_func)
    df["attention_mask"] = len_sentences_.apply(lambda t: [1] * (t - 1) + [0] * (max_len - (t - 1)))
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

def preprocess_data_gpt2tok(data_path):
    df = load_data(data_path)
    sentences, sentences_1, sentences_2 = get_sentences(df)
    padded_sentences, len_sentences = tokenize(sentences, vocab=None, gpt_tok=True)
    sentences_1, len_sentences_1 = tokenize_test(sentences_1, vocab=None, gpt_tok=True)
    sentences_2, len_sentences_2 = tokenize_test(sentences_2, vocab=None, gpt_tok=True)
    sentences_1_and_2 = pd.concat([sentences_1, sentences_2], axis=1)
    train_sentences, val_sentences, test_sentences = split_train_test(padded_sentences, sentences_1_and_2, gpt_tok=True)
    return train_sentences, val_sentences, test_sentences


if __name__ == '__main__':
    data_path = "data/ROC/ROCStories_winter2017.csv"
    train_sentences, val_sentences, test_sentences = preprocess_data_gpt2tok(data_path)
    print("done")
