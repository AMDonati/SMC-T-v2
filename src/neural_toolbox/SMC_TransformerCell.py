import tensorflow as tf
import collections
# additional imports
from models.SMC_Transformer.self_attention_SMC import Self_Attention_SMC
from models.SMC_Transformer.transformer_utils import resample
from neural_toolbox.classic_layers import point_wise_feed_forward_network

NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])

class SMC_Transf_Cell(tf.keras.layers.Layer):
  def __init__(self, d_model, output_size, seq_len, full_model, dff, **kwargs):
    '''
    :param full_model:
    :param dff:
    '''
    # store the decoding timestep
    self.dec_timestep = 0
    self.cell_count = 0
    self.attention_smc = Self_Attention_SMC(d_model=d_model)
    self.d_model = d_model
    self.output_size = output_size
    self.seq_len = seq_len
    self.full_model = full_model

    if self.full_model:
      self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm1')
      self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm2')
      self.ffn = point_wise_feed_forward_network(d_model, dff)

    # initializing smc parameters for training
    self.num_particles = 1
    self.noise = False

    # output layer for computing the weights
    self.output_layer = tf.keras.layers.Dense(output_size, name='output_layer')

    # internal states: K,V,R. size without batch_dim.
    self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  R=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]))
    self.output_size = (tf.TensorShape([self.num_particles, 1, self.d_model]),
                        tf.TensorShape([self.num_particles, 1, self.seq_len])) # r, attention_weights

    super(SMC_Transf_Cell, self).__init__(**kwargs)

  def add_SMC_parameters(self, dict_sigmas, sigma_obs, num_particles):
    self.noise = True
    self.attention_smc.add_SMC_parameters(dict_sigmas=dict_sigmas)
    self.num_particles = num_particles
    if sigma_obs is not None:
      self.Sigma_obs = sigma_obs
    else:
      self.Sigma_obs = tf.Variable(0.5, shape=(), name='Sigma_obs')
      self.Sigma_obs.assign(tf.square(self.Sigma_obs))
      print('learning sigma_obs...')
    self.list_weights, self.list_indices = [], []

  def compute_w_regression(self, predictions, y):
    '''
    # FORMULA
    # logw = -0.5 * mu_t ^ T * mu_t / omega
    # logw = logw - max(logw)
    # w = exp(logw)
    :param predictions: output of final layer: (B,P,1,F_y)
    :param y: current target element > shape (B,P,1,F_y).
    :return:
    resampling weights of shape (B,P).
    '''
    assert len(tf.shape(self.Sigma_obs)) == 0
    mu_t = y - predictions # (B,P,1,F_y)
    mu_t = tf.squeeze(mu_t, axis=-2) # removing sequence dim. # (B,P,F_y).
    log_w = (-1 / (2 * self.Sigma_obs)) * tf.matmul(mu_t, mu_t, transpose_b=True)  # (B,P,P)
    log_w = tf.linalg.diag_part(log_w)  # take the diagonal. # (B,P).
    log_w_max = tf.reduce_max(log_w, axis=1, keepdims=True)
    #log_w_scaled = log_w - log_w_max
    w = tf.math.exp(log_w)
    #w = w / tf.reduce_sum(w, axis=1, keepdims=True) # normalization.
    w = tf.nn.softmax(w)
    # check if w contains a nan number
    bool_tens = tf.math.is_nan(w)
    has_nan = tf.math.reduce_any(bool_tens).numpy()
    # if has_nan:
    #   log_w_min = tf.reduce_min(log_w, axis=1, keepdims=True)
    #   log_w_scaled = log_w - log_w_min
    #   w = tf.math.exp(log_w_scaled)
    #   w = w / tf.reduce_sum(w, axis=1, keepdims=True)  # normalization.
    #   # recheck if w contains a nan number
    #   bool_tens = tf.math.is_nan(w)
    #   has_nan = tf.math.reduce_any(bool_tens).numpy()
    assert has_nan == False

    assert len(tf.shape(w)) == 2

    return w

  def call(self, inputs, states):
    '''
    :param inputs:
    :param states:
    :return:
    '''
    x, y = tf.nest.flatten(inputs) # unnesting inputs x: shape (B,P,D), y = shape(B,P,D) with P=1 during training.
    x, y = tf.expand_dims(x, axis=-2), tf.expand_dims(y, axis=-2) # adding sequence dim.
    K, V, R = states # getting states

    # self attention:
    (z, K, V), attn_weights = self.attention_smc(inputs=x, timestep=self.dec_timestep, K=K, V=V)

    if self.full_model:
      out = self.layernorm1(z + x)
      r = self.ffn(out)
      r = self.layernorm2(r + out)
    else:
      r = z

    predictions = self.output_layer(r)  # (B,P,1,F_y)
    # storing r in R:
    R_past = R[:,:,:self.dec_timestep,:]
    R_future = R[:,:,self.dec_timestep+1:,:]
    R = tf.concat([R_past, r, R_future], axis=-2)

    # -------- SMC Algo ---------------------------------------------------------------------------------------------------------
    if self.noise:
      w = self.compute_w_regression(predictions=predictions, y=y)
      i_t = tf.random.categorical(w, self.num_particles)  # (B,P,1)
      w, i_t = tf.stop_gradient(w), tf.stop_gradient(i_t)
      self.list_weights.append(w.numpy())
      self.list_indices.append(i_t.numpy())
      # resample K, V, and R
      K = resample(params=K, i_t=i_t, t=self.dec_timestep)
      V = resample(params=V, i_t=i_t, t=self.dec_timestep)
      R = resample(params=R, i_t=i_t, t=self.dec_timestep)
      # Getting internal noises for computing the loss.
      internal_noises = [self.attention_smc.noise_q, self.attention_smc.noise_z]
      output = [r, attn_weights, internal_noises] # attn_weights > shape (B,P,1,S). noises: (B,P,1,D).
    else:
      output = [r, attn_weights]

    new_states = NestedState(K=K, V=V, R=R)
    self.cell_count += 1
    if self.cell_count > 1:
      self.dec_timestep += 1

    return output, new_states

if __name__ == "__main__":
  batch_size = 8
  d_model = 12
  output_size = 1
  seq_len = 4
  task_type = 'regression'

  # ---- test of compute w_regression ------------------------------------
  temp_cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=False, dff=0)
  temp_cell.add_SMC_parameters(dict_sigmas=None, sigma_obs=0.5, num_particles=10)

  temp_pred = tf.random.uniform(shape=(batch_size, 10, 1, output_size))
  temp_y = tf.random.uniform(shape=(batch_size, 10, 1, output_size))

  temp_w = temp_cell.compute_w_regression(predictions=temp_pred, y=temp_y)
  print('w', temp_w.shape)

  # output_size = 3
  # temp_pred = tf.random.uniform(shape=(batch_size, 10, 1, output_size))
  # temp_y = tf.random.uniform(shape=(batch_size, 10, 1, output_size))
  #
  # diag = tf.linalg.diag(tf.random.uniform(shape=(output_size,), dtype=tf.float32))
  # SMC_Transf_Cell.sigma_obs = tf.matmul(diag, diag, transpose_b=True)
  # temp_w = temp_cell.compute_w_regression(predictions=temp_pred, y=temp_y)
  # print('w', temp_w.shape)

  # ------------------------------------- code draft -------------------------------------------------------------------------------------
  # def compute_w_regression(self, predictions, y):
  #   '''
  #   # FORMULA
  #   # logw = -0.5 * mu_t ^ T * mu_t / omega
  #   # logw = logw - max(logw)
  #   # w = exp(logw)
  #   :param predictions: output of final layer: (B,P,1,F_y)
  #   :param y: current target element > shape (B,P,1,F_y).
  #   :return:
  #   resampling weights of shape (B,P).
  #   '''
  #   mu_t = y - predictions # (B,P,1,F_y)
  #   mu_t = tf.squeeze(mu_t, axis=-2) # removing sequence dim. # (B,P,F_y).
  #
  #   if len(tf.shape(self.sigma_obs)) == 0: # self.sigma_obs is a scalar std.
  #     temp = tf.scalar_mul((self.sigma_obs)**-2, mu_t)
  #   else: # self.sigma_obs is a std matrix of shape (F_y, F_y)
  #     Sigma_obs_inv = tf.matmul(tf.linalg.inv(self.sigma_obs), tf.linalg.inv(self.sigma_obs), transpose_b=True) # (F_y, F_y)
  #     temp = tf.einsum('bij,jj->bij', mu_t, Sigma_obs_inv) # (B,P,F_y)
  #
  #   log_w = (-1 / 2) * tf.matmul(temp, mu_t, transpose_b=True)  # (B,P,P)
  #   log_w = tf.linalg.diag_part(log_w)  # take the diagonal. # (B,P).
  #   log_w_max = tf.reduce_max(log_w, axis=1, keepdims=True)
  #   log_w = log_w - log_w_max
  #   w = tf.math.exp(log_w)
  #   w = w / tf.reduce_sum(w, axis=1, keepdims=True) # normalization.
  #
  #   assert len(tf.shape(w)) == 2
  #   # check if w contains a nan number
  #   bool_tens = tf.math.is_nan(w)
  #   has_nan = tf.math.reduce_any(bool_tens).numpy()
  #   assert has_nan == False
  #
  #   return w
