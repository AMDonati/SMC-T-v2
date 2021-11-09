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
  train_set, val_set, test_set = load_wikitext_dataset()
  print("done")
