from src.preprocessing.utils import split_synthetic_dataset, split_dataset_into_seq
import argparse
import numpy as np
import os
import statsmodels.api as sm

def generate_arima_model(num_samples, seq_len):
    np.random.seed(12345)
    arparams = np.array([.75, -.25, 0.095, -0.07, 0.05, -0.015, 0.01, 0.0075])
    maparams = np.array([.65, .35, -0.1, 0.08])
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    data = sm.tsa.arma_generate_sample(ar, ma, num_samples)
    # normalization
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data_norm = (data - data_mean) / data_std
    data_in_seq = split_dataset_into_seq(data_norm, start_index=0, end_index=None, history_size=seq_len, step=1)
    return data_in_seq

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-data_path", type=str, default="../../data", help="data folder to save synthetic dataset.")
  parser.add_argument('-seq_len', type=int, default=25, help="number of timesteps in the time-series dataset.")
  parser.add_argument('-num_samples', type=int, default=1000, help="number of samples in the generated synthetic dataset.")
  parser.add_argument("-TRAIN_SPLIT", type=float, default=0.8, help="train split for splitting between train and validation sets.")
  parser.add_argument("-VAL_SPLIT", type=float, default=0.5, help="split between validation and test sets.")
  parser.add_argument("-VAL_SPLIT_cv", type=float, default=0.9, help="split between train/val sets and test set when doing cv.")
  args = parser.parse_args()
  X_data = generate_arima_model(num_samples=args.num_samples, seq_len=args.seq_len)
  X_data = X_data[:,:,np.newaxis]
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
print("done")





