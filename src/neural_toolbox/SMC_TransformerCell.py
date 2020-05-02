import tensorflow as tf
import collections
# additional imports
from models.SMC_Transformer.self_attention_SMC import Self_Attention_SMC
from models.SMC_Transformer.transformer_utils import resample

NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])

class SMC_Transf_Cell(tf.keras.layers.Layer):
  def __init__(self, d_model, output_size, seq_len, **kwargs):
    '''
    '''
    # store the decoding timestep
    self.dec_timestep = 0 # decoding timestep starts at 1 because we have the init step. Cell is called S times.
    self.attention_smc = Self_Attention_SMC(d_model=d_model)
    self.d_model = d_model
    self.output_size = output_size
    self.seq_len = seq_len

    # initializing smc parameters for training
    self.num_particles = 1
    self.noise = False

    # output layer for computing the weights
    self.output_layer = tf.keras.layers.Dense(output_size, name='output_layer')

    # internal states: K,V,R. size without batch_dim.
    self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  R=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]))
    self.output_size = (tf.TensorShape([self.num_particles, 1, self.seq_len]))  # attention_weights

    super(SMC_Transf_Cell, self).__init__(**kwargs)

  def add_SMC_parameters(self, dict_sigmas, sigma_obs, num_particles):

    self.attention_smc.add_SMC_parameters(dict_sigmas=dict_sigmas)
    self.num_particles = num_particles
    self.sigma_obs = sigma_obs
    self.noise = True
    self.list_weights, self.list_indices  = [], []

  def compute_w_regression(self, predictions, y):
    '''
    # FORMULA
    # logw = -0.5 * mu_t ^ T * mu_t / omega
    # logw = logw - min(logw)
    # w = exp(logw)
    :param predictions: output of final layer (logits.) > shape (B,P,F) or (B,P,1,F) (regression case)
    :param y: current sequence element (x_t) > shape (B,F,1); F > 1 for multivariate case.
    :return:
    '''
    mu_t = y - predictions # (B,P,F)
    log_w = tf.matmul(mu_t, mu_t, transpose_b=True)  # (B,P,P)
    log_w = tf.linalg.diag_part(log_w)  # take the diagonal.
    log_w = tf.scalar_mul(-1 / (2 * (self.sigma_obs) ** 2), log_w)# omega here is the stddev.
    log_w = tf.squeeze(log_w, axis=-1)
    log_w_min = tf.reduce_min(log_w, axis=1, keepdims=True)
    log_w = log_w - log_w_min
    w = tf.math.exp(log_w)
    # normalization
    w = w / tf.reduce_sum(w, axis=1, keepdims=True)

    assert len(tf.shape(w)) == 2
    # check if w contains a nan number
    bool_tens = tf.math.is_nan(w)
    has_nan = tf.math.reduce_any(bool_tens).numpy()
    assert has_nan == False

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

    #TODO: here remove the first timestep element of K,V,R when self.dec_timestep == 1 (because the cell computes twice
    #TODO: the first element of the sequence).

    # multi-head attention:
    (z, K, V), attn_weights = self.attention_smc(inputs=x, timestep=self.dec_timestep, K=K, V=V)
    predictions = self.output_layer(z)  # (B,P,1,F_y)

    # storing z in R:
    R_past = R[:,:,:self.dec_timestep,:]
    R_future = R[:,:,self.dec_timestep+1:,:]
    R = tf.concat([R_past, z, R_future], axis=-2)

    # -------- SMC Algo at inference .... ---------------------------------------------------------------------------------------------------------
    if self.noise:
      # computing resampling weights
      w_squeezed = self.compute_w_regression(predictions=predictions, y=y)
      i_t = tf.random.categorical(w_squeezed, self.num_particles)  # (B,P,1)
      # resample K, V, and R
      K = resample(params=K, i_t=i_t, t=self.dec_timestep)
      V = resample(params=V, i_t=i_t, t=self.dec_timestep)
      R = resample(params=R, i_t=i_t, t=self.dec_timestep)

      self.list_weights.append(w_squeezed.numpy())
      self.list_indices.append(i_t.numpy())

    output = attn_weights # attn_weights > shape (B,P,1,S).
    new_states = NestedState(K=K, V=V, R=R)
    self.dec_timestep += 1

    return output, new_states

if __name__ == "__main__":
  batch_size = 8
  d_model = 12
  target_vocab_size = 1
  seq_len = 4
  task_type = 'regression'


