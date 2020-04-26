import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np


def split_synthetic_dataset(x_data, TRAIN_SPLIT, VAL_SPLIT=0.5, VAL_SPLIT_cv=0.9, cv=False):
  if not cv:
    train_data, val_test_data = train_test_split(x_data, train_size=TRAIN_SPLIT, shuffle=True)
    val_data, test_data = train_test_split(val_test_data, train_size=VAL_SPLIT, shuffle=True)

    return train_data, val_data, test_data

  else:
    train_val_data, test_data = train_test_split(x_data, train_size=VAL_SPLIT_cv)
    kf = KFold(n_splits=5)
    list_train_data, list_val_data = [], []
    for train_index, val_index in kf.split(train_val_data):
      train_data = train_val_data[train_index, :, :]
      val_data = train_val_data[val_index, :, :]
      list_train_data.append(train_data)
      list_val_data.append(val_data)

    return list_train_data, list_val_data, test_data

def split_input_target_uni_step(chunk):
  input_text = chunk[:,:-1,:]
  target_text = chunk[:,1:,:]
  return input_text, target_text


def data_to_dataset_uni_step(train_data, val_data, test_data, split_fn, BUFFER_SIZE, BATCH_SIZE, cv, target_feature=None):
  '''
  :param train_data: input data for training > shape (N_train, S+1, F) ; N_train = number of samples in training dataset.
  :param val_data: input data used for validation set > shape (N_val, S+1, F)
  :param split_fn: used to split between input data and target.
  :param BUFFER_SIZE: to shuffle the dataset.
  :param BATCH_SIZE:
  :param: cv: boolean; True if multiple train datasets / val datasets for cross-validation; False otherwise.
  :param target_feature: used to select the target feature to be predicted. Case of multivariate ts as input data > prediction of a univariate ts.
  :return:
  2 tf.data.Dataset, one for the training set, and one for the validation set, with:
  input data:  batches of train data > shape (B, S+1, F) > S+1 because the data is split in the SMC_Transformer.Py script.
  target data: shape (B,S,1) > univariate ts to be predicted (shifted from one timestep compared to the input data).
  '''
  if not cv:
    list_train_data = [train_data]
    list_val_data = [val_data]
  else:
    list_train_data = train_data
    list_val_data = val_data

  list_train_dataset, list_val_dataset = [], []

  for (train_data, val_data) in zip(list_train_data, list_val_data):
    x_train, y_train = split_fn(train_data)
    x_val, y_val = split_fn(val_data)

    if target_feature is not None:
      y_train = y_train[:, :, target_feature]
      y_train = np.reshape(y_train, newshape=(y_train.shape[0], y_train.shape[1], 1))
      y_val = y_val[:, :, target_feature]
      y_val = np.reshape(y_val, newshape=(y_val.shape[0], y_val.shape[1], 1))

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    list_train_dataset.append(train_dataset)
    list_val_dataset.append(val_dataset)

  x_test, y_test = split_fn(test_data)
  if target_feature is not None:
    y_test = y_test[:, :, target_feature]
    y_test = np.reshape(y_test, newshape=(y_test.shape[0], y_test.shape[1], 1))

  BATCH_SIZE_test = test_data.shape[0]
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_dataset = test_dataset.batch(BATCH_SIZE_test)

  if not cv:
    train_dataset, val_dataset = list_train_dataset[0], list_val_dataset[0]
    return train_dataset, val_dataset, test_dataset
  else:
    return list_train_dataset, list_val_dataset, test_dataset


if __name__ == "__main__":
  cv = False
  TRAIN_SPLIT = 0.7
  BUFFER_SIZE = 500
  BATCH_SIZE = 128

  file_path="../../../data/synthetic_dataset_1_feat.npy"
  X_data = np.load(file_path)
  train_data_synt, val_data_synt, test_data_synt = split_synthetic_dataset(x_data=X_data, TRAIN_SPLIT=0.7, cv=cv)

  if not cv:
    print('train data', train_data_synt.shape)
    print('val_data', val_data_synt.shape)
    print('test data', test_data_synt.shape)

  train_dataset_synt, val_dataset_synt, test_dataset_synt, _, _, _ = data_to_dataset_uni_step(train_data=train_data_synt,
                                                      val_data=val_data_synt,
                                                      test_data=test_data_synt,
                                                      split_fn=split_input_target_uni_step,
                                                      BUFFER_SIZE=BUFFER_SIZE,
                                                      BATCH_SIZE=BATCH_SIZE,
                                                      target_feature=None,
                                                      cv=cv)

  print('train synthetic dataset', train_dataset_synt)
  print('val dataset synthetic', val_dataset_synt)
  print('test dataset synthetic', test_dataset_synt)



