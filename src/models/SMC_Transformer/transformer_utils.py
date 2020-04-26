import tensorflow as tf
import numpy as np

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def positional_encoding_SMC(position, d_model, num_particles):
  '''
  tiling operation added compared to the classical positional encoding.
  :param position:
  :param d_model:
  :param num_particles:
  :return:
  preprocessed word input of shape (B,P,S,D).
  '''
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]
  pos_encoding = pos_encoding[:, np.newaxis, :, :]

  pos_encoding = tf.tile(pos_encoding, [1, num_particles, 1, 1])

  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq, num_particles):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  temp = seq[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
  return tf.tile(temp, multiples=[1, num_particles, 1, 1, 1])  # (batch_size, num_particles 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_look_ahead_mask_v2(size):
  '''not used for now'''
  mask=1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return 1-mask

def resample_old(params, indices):
  '''GOOD RESAMPLING FUNCTION!!!'''
  seq_length=tf.shape(params)[2]
  params_list=[]
  for t in range(seq_length):
    params_t=params[:,:,t,:]
    indices_t=indices[:,:,t]
    params_resampl_t=tf.gather(params_t, indices_t, batch_dims=1)
    params_resampl_t=tf.expand_dims(params_resampl_t, axis=2)
    params_list.append(params_resampl_t)
  params_resampl=tf.stack(params_list, axis=2)
  params_resampl=tf.squeeze(params_resampl, axis=-2)
  return params_resampl

def resample(params, i_t, t):
  """
  :param params: attention parameters tensor to be reshaped (K or V) > shape (B,P,S,D)
  :param i_t: current set of indices at time t > shape (B,P)
  :param t: decoding timestep (int from 0 until seq_len-1)
  :return:
  the trajectories of the attention parameters resampling according to i_t.
  """
  #TODO use tf.scatter_nd instead to avoid the for loop on the number of particles?
  num_particles = tf.shape(params)[1]
  #i_t=ind_matrix[:,:,t]# shape (B,P)
  past_params = params[:,:,:t+1,:] # (B,P,t,D)
  future_params = params[:,:,t+1:,:] #(B,P,S-t,D)
  rows_new_params = []
  for m in range(num_particles):
    i_t_m = i_t[:,m] # shape B
    # reshaping to (B,1)
    i_t_m = tf.expand_dims(i_t_m, axis=-1)
    row_m_new_params = tf.gather(past_params, i_t_m, axis=1, batch_dims=1) # shape (B,1,t-1,D)
    # squeezing on 2nd dim:
    row_m_new_params = tf.squeeze(row_m_new_params, axis=1)
    rows_new_params.append(row_m_new_params)
  # stacking the new rows in the a new tensor
  new_params = tf.stack(rows_new_params, axis=1) # add a tf.expand_dims? # (B,P,t-1,D)
  new_params = tf.concat([new_params, future_params], axis=2) # concatenating new_params (until t-1) and old params (from t)

  return new_params


def resample_z(z, curr_ind):
  #curr_ind = indices[:,:,dec_timestep]
  z_resampl = tf.gather(z, curr_ind, batch_dims=1)
  return z_resampl


def compute_direct_update_cov_matrix(self):
    '''THIS FUNCTION WONT BE USED....'''
    '''
    for each layer, update the sigma of the reparametrized noise with the optimal sigma of the current training step.
    '''
    # function that will be probably not needed.

    #TODO: change the implementation of the update operation following Florian's advice:
    #TODO https://stackoverflow.com/questions/48260394/whats-the-differences-between-tf-graphkeys-trainable-variables-and-tf-graphkeys

    list_stddev = self.decoder.list_stddev
    list_upd_sigmas = []
    # num_layers=len(list_stddev)
    seq_length = tf.shape(list_stddev[0])[2]
    for l, stddev in enumerate(list_stddev):
      sigma_optimal_l = []
      for t in range(seq_length):
        stddev_t = stddev[:, :, t, :]
        # permute P and D dims:
        stddev_t = tf.transpose(stddev_t, perm=[0, 2, 1])  # dim (B,D,P)
        # do double dot product to have on the particle and batch dims to have at the end a tensor of dim (D,D)
        sigma_optimal_l_t = tf.tensordot(stddev_t, tf.linalg.matrix_transpose(stddev_t),
                                         axes=[[0, 2], [0, 1]])  # dim (D,D)
        sigma_optimal_l.append(sigma_optimal_l_t)
      sigma_optimal_l = tf.stack(sigma_optimal_l, axis=0)  # dim (S,D,D)
      # sum over all the layers
      sigma_optimal_l = tf.reduce_sum(sigma_optimal_l, axis=0)  # dim (D,D)
      # multiply by 1/seq_len
      # sigma_optimal_l=tf.math.scalar_mul(1/seq_length, sigma_optimal_l) # to debug.
      # sigma_optimal_l=1/seq_length*sigma_optimal_l
      sigma_optimal_l = tf.cast(sigma_optimal_l, dtype=tf.float32)
      list_upd_sigmas.append(sigma_optimal_l)
      self.decoder.dec_layers[l].mha1.sigma = sigma_optimal_l  # to check with Florian and to change with new

    return list_upd_sigmas

if __name__ == "__main__":
  test_resample_z = True
  test_resample = True
  test_sample_and_keep_indices = False

  # --- test of positional encoding -------------------------------------------------------------------------------------------------------------------
  b=8
  S=20
  pe_target=10
  d_model=64
  inputs=tf.random.uniform(shape=(b, S, d_model))
  pos_enc=positional_encoding(position=pe_target, d_model=d_model)
  #inputs+=pos_enc[:,:,:]
  #print('inputs', input.shape)
  #---- resampling z test--------------------------------------------------------------------------------------------------------------------------------

  if test_resample_z:
    z = tf.constant([[[1,1],[2,2],[3,3],[4,4],[5,5]],[[6,6],[7,7],[8,8],[9,9],[10,10]]])
    indices = tf.constant([[[0],[0],[1],[2],[1]],[[1],[1],[4],[4],[1]]])
    z_resampl = resample_z(z, indices, 0)
    print('z resampled', z_resampl.shape)

  #---------- test of corrected resample function-----------------------------------------------------------------------------------------------------------

  if test_resample:
    B=2
    S=3
    P=4
    D=1

    ind_matrix= tf.constant([[[1, 1, 2, 2], [0, 0, 0, 0], [1, 1, 1, 0]],
                             [[0, 1, 3, 2], [3, 3, 2, 0], [1, 2, 3, 1]]], shape=(B, S, P))
    ind_matrix = tf.transpose(ind_matrix, perm=[0,2,1])
    #ind_matrix = tf.tile(tf.expand_dims(ind_matrix, axis=0), multiples=[B, 1, 1])  # (B,P,S)

    print('indices_matrices', ind_matrix[0,:,:].numpy())

    K=tf.constant([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                   [[13,14,15,16],[17,18,19,20],[21,22,23,24]]], shape=(B,S,P))
    K=tf.transpose(K, perm=[0,2,1])
    K=tf.expand_dims(K, axis=-1) # (B,P,S,D=1)
    print('init K', K[0,:,:,0])

    truth_t0_b1 = tf.constant([[2, 5, 9], [2, 6, 10], [3, 7, 11], [3, 8, 12]], shape=(P, S))
    truth_t1_b1 = tf.constant([[2, 5, 9], [2, 5, 10], [2, 5, 11], [2, 5, 12]], shape=(P, S))
    truth_t2_b1 = tf.constant([[2, 5, 10], [2, 5, 10], [2, 5, 10], [2, 5, 9]], shape=(P, S))
    truth_t0_b2 = tf.constant([[13, 17, 21], [14, 18, 22], [16, 19, 23], [15, 20, 24]], shape=(P, S))
    truth_t1_b2 = tf.constant([[15, 20, 21], [15, 20, 22], [16, 19, 23], [13, 17, 24]], shape=(P, S))
    truth_t2_b2 = tf.constant([[15, 20, 22], [16, 19, 23], [13, 17, 24], [15, 20, 22]], shape=(P, S))

    truth_t0 = tf.stack([truth_t0_b1, truth_t0_b2], axis=0)
    truth_t1 = tf.stack([truth_t1_b1, truth_t1_b2], axis=0)
    truth_t2 = tf.stack([truth_t2_b1, truth_t2_b2], axis=0)

    new_K=K
    for t in range(S):
      i_t=ind_matrix[:,:,t]
      new_K=resample(params=new_K, i_t=i_t, t=t)
      print('new K at time_step for batch 0 {}: {}'.format(t,new_K[0,:,:,0]))
      print('new K at time_step for batch 1 {}: {}'.format(t,new_K[1,:,:,0]))

    # ok, test passed.

  #---- test of sample_and_keep_indices function---------------------------------------------------------------------

  if test_sample_and_keep_indices:
    B = 2
    P = 3
    S = 4
    t=2
    prev_sampling_weights = tf.random.uniform(shape=(B, P), maxval=1)
    ind_matrix_T = tf.constant([0 for _ in range(P)], shape=(1, P, 1), dtype=tf.int32)
    # S+1 is a trick to be able to update the last decoding timestep
    ind_matrix_T = tf.tile(ind_matrix_T, [B, 1, S+1])

    # ind_matrix=tf.constant([[1,2,3],[2,2,2]], shape=(1,P,2), dtype=tf.int32)
    # ind_matrix=tf.tile(ind_matrix, multiples=[B,1,1])
    # ind_matrix_right = tf.tile(ind_matrix_right, [B, 1, S-2])
    # ind_matrix_T=tf.concat([ind_matrix, ind_matrix_right],axis=-1)

    # FOR TESTING - to remove
    indices_t1 = tf.constant([2, 2, 2], shape=(1, P, 1), dtype=tf.int32)
    indices_t1 = tf.tile(indices_t1, multiples=[B, 1, 1])

    indices_t2 = tf.constant([1, 2, 2], shape=(1, P, 1), dtype=tf.int32)
    indices_t2= tf.tile(indices_t2, multiples=[B, 1, 1])

    indices_t3 = tf.constant([1, 1, 3], shape=(1, P, 1), dtype=tf.int32)
    indices_t3 = tf.tile(indices_t3, multiples=[B, 1, 1])

    curr_indices, matrix_updated_t1 = sample_and_keep_indices(prev_sampling_weights=prev_sampling_weights,
                                                              ind_matrix=ind_matrix_T,
                                                              num_particles=P,
                                                              dec_timestep=1,
                                                              indices=indices_t1)

    curr_indices, matrix_updated_t2 = sample_and_keep_indices(prev_sampling_weights=prev_sampling_weights,
                                                    ind_matrix=matrix_updated_t1,
                                                    num_particles=P,
                                                    dec_timestep=2,
                                                    indices=indices_t2)

    #TODO: solve the bug happening at the last_timestep.
    # does not work for the last time_step
    curr_indices, matrix_updated_t3 = sample_and_keep_indices(prev_sampling_weights=prev_sampling_weights,
                                                           ind_matrix=matrix_updated_t2,
                                                          num_particles=P,
                                                          dec_timestep=3,
                                                          indices=indices_t3)


    print('indices matrix at time t1', matrix_updated_t1[0,:,:])
    print('indices matrix at time t2', matrix_updated_t2[0, :, :])
    #print('indices matrix at time t3', matrix_updated_t3[0, :, :])