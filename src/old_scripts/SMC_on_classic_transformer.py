import tensorflow as tf
from src.models.Baselines.Transformer_without_enc import Transformer

def SMC_on_Transformer(transformer, dict_sigmas, sigma_obs, list_inputs, list_targets):
  '''
  :param transformer:
  :param dict_sigmas:
  :param sigma_obs:
  :param list_inputs:
  :param list_targets:
  :return:
  '''
  num_particles = tf.shape(list_inputs[0])[1]
  mha = transformer.decoder.dec_layers[-1].mha

  # adding smc parameters to the self-attention part of the transformer.
  mha.add_SMC_parameters(dict_sigmas=dict_sigmas)

  # putting positional encoding to None:
  transformer.maximum_position_encoding = None
  transformer.decoder.maximum_position_encoding = None

  for t, (input, target) in enumerate(zip(list_inputs, list_targets)):
    mha.dec_timestep = t
    pred_t, _ = transformer(inputs=input, training=False, mask=None)
    if t == 0:
      predictions = pred_t
    else:
      predictions = tf.concat([predictions, pred_t], axis=-2)

    # compute w_t, and i_t:
    w_t = compute_w_regression(prediction=pred_t, target=target, sigma_obs=sigma_obs) # should be of shape (B,P)
    i_t = tf.random.categorical(w_t, num_particles)
    # resample K,V,predictions
    K = resample(params=mha.K, i_t=i_t)
    V = resample(params=mha.V, i_t=i_t)
    predictions = resample(params=predictions, i_t=i_t)
    mha.K, mha.V = K, V

  return predictions, K, V

def compute_w_regression(prediction, target, sigma_obs):
  mu = target - prediction # (B,P,F)
  log_w = tf.matmul(mu, mu, transpose_b=True) # try a transpose a instead? # (B,P,P)
  log_w = tf.linalg.diag_part(log_w) # (B,P,1)
  log_w = tf.scalar_mul(-1/(2* (sigma_obs)**2), log_w) # (B,P,1)
  log_w_min = tf.reduce_min(log_w, axis=1, keepdims=True) # (B,P,1)
  log_w = log_w - log_w_min # (B,P,1)
  w = tf.math.exp(log_w) # (B,P,1)
  w = w / tf.reduce_sum(w, axis=1, keepdims=True)

  w = tf.squeeze(w, axis=-1)
  assert len(tf.shape(w)) == 2

  return w

def resample(params, i_t):
  """
  :param params: attention parameters tensor to be reshaped (K or V) > shape (B,P,S,D)
  :param i_t: current set of indices at time t > shape (B,P)
  :return:
  the trajectories of the attention parameters resampling according to i_t.
  """
  num_particles = tf.shape(params)[1]
  rows_new_params = []
  for m in range(num_particles):
    i_t_m = i_t[:,m] # shape B
    i_t_m = tf.expand_dims(i_t_m, axis=-1) # reshaping to (B,1)
    row_m_new_params = tf.gather(params, i_t_m, axis=1, batch_dims=1) # shape (B,1,t,D)
    # squeezing on 2nd dim:
    row_m_new_params = tf.squeeze(row_m_new_params, axis=1)
    rows_new_params.append(row_m_new_params)
  # stacking the new rows in the a new tensor
  new_params = tf.stack(rows_new_params, axis=1) # (B,P,t,D)

  return new_params

if __name__ == "__main__":
  B = 8
  P = 5
  S = 10
  d_model = 6
  dff = 24
  F = 3
  num_heads = 1
  maximum_position_encoding = 50
  data_type = 'time_series_multi'
  C = F
  rate = 0

  sigma_obs = 0.5
  dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [0.1, 0.1, 0.1, 0.1]))

  list_inputs = [tf.random.uniform(shape=(B,P,1,F)) for _ in range(S)]
  list_targets = [tf.random.uniform(shape=(B,P,1,F)) for _ in range(S)]

  sample_transformer = Transformer(num_layers=1, d_model=d_model, num_heads=1, dff=dff, target_vocab_size=C,
                                   maximum_position_encoding=maximum_position_encoding, rate=rate, full_model=False)

  predictions, K, V = SMC_on_Transformer(transformer=sample_transformer,
                                         dict_sigmas=dict_sigmas,
                                         sigma_obs=sigma_obs,
                                         list_inputs=list_inputs,
                                         list_targets=list_targets)

  #  ----------------- test of resample function ------------------------------------------------------------------------------------------------
  B = 2
  S = 3
  P = 4
  D = 1
  ind_matrix = tf.constant([[[1, 1, 2, 2], [0, 0, 0, 0], [1, 1, 1, 0]],
                            [[0, 1, 3, 2], [3, 3, 2, 0], [1, 2, 3, 1]]], shape=(B, S, P))
  ind_matrix = tf.transpose(ind_matrix, perm=[0, 2, 1])

  K = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], shape=(B, S, P))
  K = tf.transpose(K, perm=[0, 2, 1])
  K = tf.expand_dims(K, axis=-1)  # (B,P,S,D=1)

  new_K = K
  for t in range(S):
    i_t = ind_matrix[:, :, t]
    new_K = resample(params=new_K, i_t=i_t)
    print('new K at time_step for batch 0 {}: {}'.format(t, new_K[0, :, :, 0]))
    print('new K at time_step for batch 1 {}: {}'.format(t, new_K[1, :, :, 0]))
