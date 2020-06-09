import tensorflow as tf
import numpy as np


def generate_onesample_model_1(A, std_matrix, seq_len, num_features):
  X = tf.random.normal(shape=(1, num_features))
  list_X=[X]
  for s in range(seq_len):
    X = tf.matmul(X,A) + tf.random.normal(stddev=std_matrix, shape=(1, num_features))
    list_X.append(X)
  X_obs = tf.stack(list_X, axis=1)
  return X_obs

def generate_onesample_model_2(A, std_matrix, seq_len, num_features):
   X = tf.random.normal(shape=(1, num_features))
   list_X=[X]
   for s in range(seq_len):
     U = np.random.uniform()
     if U<0.7:
         X = tf.matmul(X,A) + tf.random.normal(stddev=std_matrix, shape=(1, num_features))
     else:
         X = 0.6*tf.matmul(X,A) + tf.random.normal(stddev=std_matrix, shape=(1, num_features))
     list_X.append(X)
   X_obs = tf.stack(list_X, axis=1)
   return X_obs

if __name__ == "__main__":
  seq_len = 24
  num_samples = 10000
  num_features = 1

  std_matrix = tf.sqrt(tf.constant(0.5, shape=(1,1), dtype=tf.float32))
  A = tf.constant([0.8], shape=(1,1), dtype=tf.float32)
  list_samples = []

  for N in range(num_samples):
    X_seq = generate_onesample_model_1(A=A, std_matrix=std_matrix, seq_len=seq_len, num_features=num_features)
    list_samples.append(X_seq)

  X_data = tf.stack(list_samples, axis=0)
  X_data = tf.squeeze(X_data, axis=1)

  data_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data'
  file_path = data_path + '/synthetic_dataset_{}_feat.npy'.format(num_features)
  np.save(file_path, X_data)
  print('X data', X_data.shape)
