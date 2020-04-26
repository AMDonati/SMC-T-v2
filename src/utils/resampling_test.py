import tensorflow as tf
from utils.transformer_utils import resample
from neural_toolbox.SMC_layers import DecoderLayer_SMC


def sample_and_keep_indices(dec_timestep, num_particles, prev_sampling_weights, ind_matrix):  # add a mask argument?
  '''samples the set of N indices for doing the weights resampling
  adds this set of indices to the matrix of ancestor indices
  Args:
  -prev_sampling_weights: w(k-1) > dim (B, P)
  -indice matrix: I0:k-1 > dim (B, P, S)
  Returns:
  -The current set of indices to do a forward pass on the Decoder Layer > dim (batch_size, num_particles)
  -The updated ancestor indices matrix > dim (batch_size, NUM_PARTICLES, seq_length)'''

  # FUNCTION THAT NEEDS TO BE TESTED... ok test done.

  # Sample current set of indices with proba proportional to prev_sampling_weights
  if len(tf.shape(prev_sampling_weights)) == 3:
    prev_sampling_weights = tf.squeeze(prev_sampling_weights, axis=-1)
  indices = tf.random.categorical(prev_sampling_weights, num_particles)  # shape (..., num_particles)
  # indices=tf.math.add(indices, tf.constant(1, dtype=indices.dtype))

  # Add this set of indices to the indices matrix tensor:
  indices = tf.cast(indices, tf.int32)
  indices = tf.expand_dims(indices, axis=-1)
  updated_ind_matrix = tf.concat(
    [ind_matrix[:, :, :dec_timestep + 1], indices,
     ind_matrix[:, :, dec_timestep + 2:]], axis=-1) # check if dec_timestep+1 makes sense or not.

  return indices, updated_ind_matrix

if __name__ == "__main__":

  # --test of function resample----

  params_array=[[[[0.6, 0.3],[0.4, 0.5]],[[0.1, 0.1], [0.2, 0.3]], [[1, 2], [0.5, 0.6]]],
                [[[1,1], [2,2]],[[3,3], [4,3]], [[4,5],[0.1, 2]]]]
  params=tf.constant(params_array, dtype=tf.float32)
  print(params.shape)

  indices_array=[[[0,2], [2,2], [1,2]], [[1,1], [2,1], [0,1]]]
  indices=tf.constant(indices_array, dtype=tf.int32)
  print(indices.shape)

  #params_resampl=resample(indices, params)

  # ok works.

  # -----test of function sample_and_keep_indices of the SMC_layer----
  P=5
  t=2
  w=tf.random.uniform(maxval=1,shape=(1,5))
  ind_matrix=tf.ones(shape=(1,5,3), dtype=tf.int32)
  indices, upd_ind_matrix=sample_and_keep_indices(0,P,w,ind_matrix)
  print('indices',indices)
  print('indices matrix', upd_ind_matrix)

  # seems to work. Chech dec_timstep+1 though.

