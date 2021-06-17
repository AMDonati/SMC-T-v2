from src.preprocessing.utils import split_synthetic_dataset
import argparse
import numpy as np
import os
import statsmodels.api as sm

def generate_arima_model(num_samples, seq_len):
    np.random.seed(12345)
    arparams = np.array([0.9, 0.85, 0.7, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    maparams = np.array([0.65, -0.35, 0.01, -0.025, 0.8, -0.5])
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    y = sm.tsa.arma_generate_sample(ar, ma, (num_samples,seq_len), axis=1)
    return y

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-data_path", type=str, default="data", help="data folder to save synthetic dataset.")
  parser.add_argument('-seq_len', type=int, default=24, help="number of timesteps in the time-series dataset.")
  parser.add_argument('-num_samples', type=int, default=1000, help="number of samples in the generated synthetic dataset.")
  parser.add_argument("-TRAIN_SPLIT", type=float, default=0.8, help="train split for splitting between train and validation sets.")
  parser.add_argument("-VAL_SPLIT", type=float, default=0.5, help="split between validation and test sets.")
  parser.add_argument("-VAL_SPLIT_cv", type=float, default=0.9, help="split between train/val sets and test set when doing cv.")
  args = parser.parse_args()
  X_data = generate_arima_model(num_samples=args.num_samples, seq_len=args.seq_len)
  out_path = os.path.join(args.data_path, "arima_model")
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






