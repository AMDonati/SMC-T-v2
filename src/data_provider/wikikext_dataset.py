from datasets import list_datasets, list_metrics, load_dataset, load_metric
from pprint import pprint
import torch
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np

def load_wikitext_dataset():
  dataset = load_dataset('wikitext', 'wikitext-2-v1')
  train_set, val_set, test_set = dataset["train"], dataset["validation"], dataset["test"]
  return train_set, val_set, test_set

if __name__ == '__main__':
  import tensorflow as tf
  from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

  # add the EOS token as PAD token to avoid warnings
  model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

  # encode context the generation is conditioned on
  input_ids = tokenizer.encode('I enjoy going running in the morning.', return_tensors='tf')

  # generate text until the output length (which includes the context length) reaches 50
  greedy_output = model.generate(input_ids, max_length=50)

  print("Output:\n" + 100 * '-')
  print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

  print("BEAM SEARCH ------------------------------------------------------------")

  beam_output = model.generate(
    input_ids,
    max_length=50,
    num_beams=10,
    early_stopping=True
  )

  print("Output:\n" + 100 * '-')
  print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

  print("SAMPLNG------------------------------------------------------------")

  # set seed to reproduce results. Feel free to change the seed though to get different results
  tf.random.set_seed(0)

  # activate sampling and deactivate top_k by setting top_k sampling to 0
  sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=0
  )

  print("Output:\n" + 100 * '-')
  print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

  print(" TOP-K SAMPLING ----------------------------------------------------------------------")

  # set seed to reproduce results. Feel free to change the seed though to get different results
  tf.random.set_seed(0)

  # set top_k to 50
  sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=50
  )

  print("Output:\n" + 100 * '-')
  print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

  print("NUCLEUS SAMPLING --------------------------------------------------------------------------------")

  # set seed to reproduce results. Feel free to change the seed though to get different results
  tf.random.set_seed(0)

  # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
  sample_outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
  )

  print("Output:\n" + 100 * '-')
  for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

