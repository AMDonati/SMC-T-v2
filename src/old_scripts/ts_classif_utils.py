import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def univariate_data_with_targets(dataset, start_index, end_index, history_size, target_size, targets=None):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i - history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i + target_size])
    if targets is None:
      labels.append(dataset[i + target_size])
    else:
      print('categorical binned targets')
      indices_tar = range(i - target_size, i)
      labels.append(np.reshape(targets[indices_tar], (target_size, 1)))  # to reshape like for the data.
  return np.array(data), np.array(labels)


def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i - history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i + target_size])
  return np.array(data), np.array(labels)

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def create_bins(lower_bound, width, quantity):
  """ create_bins returns an equal-width (distance) partitioning.
      It returns an ascending list of tuples, representing the intervals.
      A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0
      and i < quantity, satisfies the following conditions:
          (1) bins[i][0] + width == bins[i][1]
          (2) bins[i-1][0] + width == bins[i][0] and
              bins[i-1][1] + width == bins[i][1]
  """
  bins = []
  for low in range(lower_bound,
                   lower_bound + quantity * width + 1, width):
    bins.append((low, low + width))
  bins2 = pd.IntervalIndex.from_tuples(bins, closed='left')
  return bins2

def create_time_steps(length):
  return list(range(-length, 0))

def baseline(history):
  return np.mean(history)

def get_key(dict, val):
  for key, value in dict.items():
    if val == value:
      return key

def map_uni_data_classes(continuous_data, list_interval, dict_temp):
  for j,I in enumerate(list_interval):
    bool_index=continuous_data.map(lambda x: x in I) # pd.series with boolean.
    bool_index=bool_index.map(lambda x: get_key(dict_temp, I) if x is True else np.nan) # pd.series with index of I in dict and np.nan values
    # drop the nan values
    sample_class_I=bool_index.dropna() # pd.series
    # append the successive sample_class_I
    if j==0:
      final_series=sample_class_I
    else:
      final_series=pd.concat([final_series, sample_class_I], ignore_index=True)
  # assert that uni_data and the series with the classes have the same len
  #assert len(final_series)==len(continuous_data)
  return final_series

def create_categorical_dataset_from_bins(df, min_value, bin_interval, num_bins):
  bins_binary = create_bins(min_value, bin_interval, num_bins)

  bins2 = pd.IntervalIndex.from_tuples(bins_binary, closed='left')
  temp_bins = pd.cut(df, bins2)

  bins_temp = list(set(list(temp_bins.values)))
  dict_temp_range = OrderedDict(zip(range(num_bins), bins_temp))

  list_temp_classes = [get_key(dict_temp_range, val) for val in list(temp_bins.values)]

  temp_classes = pd.Series(list_temp_classes)
  classes_array = temp_classes.values

  return classes_array

