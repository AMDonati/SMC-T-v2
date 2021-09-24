from src.preprocessing.utils import split_synthetic_dataset
import argparse
import numpy as np
import os
import tensorflow as tf

def generate_onesample(seq_len=30, vocab_size=50, num_samples=1000):
  X_obs = np.random.randint(0,vocab_size, size=(num_samples, seq_len))
  return X_obs[:,:,np.newaxis]



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-data_path", type=str, default="data", help="data folder to save synthetic dataset.")
  parser.add_argument("-TRAIN_SPLIT", type=float, default=0.8, help="train split for splitting between train and validation sets.")
  parser.add_argument("-VAL_SPLIT", type=float, default=0.5, help="split between validation and test sets.")
  parser.add_argument("-VAL_SPLIT_cv", type=float, default=0.9, help="split between train/val sets and test set when doing cv.")
  args = parser.parse_args()

  X_data = generate_onesample()

  out_path = os.path.join(args.data_path, "dummy_model_nlp")
  if not os.path.isdir(out_path):
      os.makedirs(out_path)
  out_file = os.path.join(out_path, "raw_data.npy")

  print("saving synthetic dataset into a .npy file...")
  np.save(out_file, X_data)
  folder_path = os.path.dirname(out_file)
  train_data_synt, val_data_synt, test_data_synt = split_synthetic_dataset(x_data=X_data,
                                                                           save_path=folder_path,
                                                                           TRAIN_SPLIT=args.TRAIN_SPLIT,
                                                                           VAL_SPLIT=args.VAL_SPLIT,
                                                                           VAL_SPLIT_cv=args.VAL_SPLIT_cv,
                                                                           cv=False)






