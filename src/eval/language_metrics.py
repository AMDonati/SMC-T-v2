from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import tensorflow as tf

gpt2_config = GPT2Config(vocab_size=50257)
gpt2_model = TFGPT2LMHeadModel(gpt2_config).from_pretrained("cache/gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")
gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_tokenizer.pad_token_id = [50256]

def get_weights_bleu_score(n_gram=4):
    if n_gram == 2:
        weights = [0.5, 0.5]
    elif n_gram == 3:
        weights = [1 / 3, 1 / 3, 1 / 3]
    elif n_gram == 4:
        weights = [0.25, 0.25, 0.25, 0.25]
    return weights

def gpt2_perplexity(sentence):
    inputs = gpt2_tokenizer(sentence, return_tensors="tf")
    outputs = gpt2_model(input_ids=inputs["input_ids"])
    logits = outputs["logits"]
    preds = logits[:, :-1, :]
    targets = inputs["input_ids"][:, 1:]
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    cross_entropy = tf.reduce_mean(ce(y_true=targets, y_pred=preds)) # (B,1,S)
    ppl = tf.math.exp(cross_entropy)
    return round(ppl.numpy(),2)

def gpt2_perplexity_batch(sentences, tokenizer=None, reduction=True):
    if tokenizer is not None:
        P = sentences.shape[0]
        S = sentences.shape[1]
        sentences = [tokenizer.decode(sentences[i].numpy()) for i in range(sentences.shape[0])]
    inputs = gpt2_tokenizer(sentences, padding=True, truncation=True, return_tensors="tf")
    labels = tf.identity(inputs["input_ids"])
    labels_ = tf.where(inputs["attention_mask"] == 0, x=tf.constant(-100, shape=labels.shape), y=labels)
    outputs = gpt2_model(**inputs, labels=labels_)
    loss = outputs["loss"]
    if not reduction and tokenizer is not None:
        loss = tf.reshape(loss, shape=(P,S-1))
        ppl = tf.exp(tf.reduce_mean(loss, axis=-1))
    else:
        ppl = tf.math.exp(tf.reduce_mean(loss))
        ppl = round(ppl.numpy(),2)
    return ppl

def BLEU_score(true_sentence, generated_sentence, split_str=False):
    weights = get_weights_bleu_score(4)
    if split_str:
        true_sentence = true_sentence.split(sep=' ')
        generated_sentence = [generated_sentence.split(sep=' ')]
    score = sentence_bleu(references=generated_sentence, hypothesis=true_sentence, smoothing_function=SmoothingFunction().method2, weights=weights)
    return score

def SELFBLEU_score(sentences):
    scores = []
    for i, sentence in enumerate(sentences):
        ref_sentences = np.delete(sentences, i)
        score = BLEU_score(true_sentence=sentence, generated_sentence=ref_sentences)
        scores.append(score)
    return np.mean(score)


if __name__ == '__main__':
    true_sentence = "My name is Alice."
    generated_sentence = true_sentence
    score1 = BLEU_score(true_sentence, generated_sentence, split_str=True)
    print(score1)
    generated_sentence2 = "My name is Bob"
    score2 = BLEU_score(true_sentence, generated_sentence2, split_str=True)
    print(score2)
    generated_sentence3= "I love my name"
    score3 = BLEU_score(true_sentence, generated_sentence3, split_str=True)
    print(score3)
    generated_sentence4 = "I like it!"
    score4 = BLEU_score(true_sentence, generated_sentence4, split_str=True)
    print(score4)

    sentences = [true_sentence, generated_sentence4]
    gpt2_ppl = gpt2_perplexity_batch(sentences)

    print("checking perplexity formula....")
    print(gpt2_perplexity(true_sentence))
    print(gpt2_perplexity_batch(true_sentence))
    print("done")