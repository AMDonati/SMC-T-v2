import tensorflow as tf
import collections
# additional imports
from models.SMC_Transformer.self_attention_SMC import Self_Attention_SMC
from models.SMC_Transformer.transformer_utils import resample

NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])

class SMC_Transf_Cell(tf.keras.layers.Layer):
  def __init__(self, d_model, output_size, num_particles, seq_len,
               sigmas_att, noise, task_type, sigma_obs, training,
               resampling=True, test=False, **kwargs):
    '''
    '''
    # store the decoding timestep
    self.dec_timestep = 0 # decoding timestep starts at 1 because we have the init step. Cell is called S times.
    sigma_k = sigmas_att['k']
    sigma_q = sigmas_att['q']
    sigma_v = sigmas_att['v']
    sigma_z = sigmas_att['z']
    self.attention_smc = Self_Attention_SMC(d_model=d_model,
                                            num_particles=num_particles,
                                            sigma_k=sigma_k,
                                            sigma_v=sigma_v,
                                            sigma_q=sigma_q,
                                            sigma_z=sigma_z,
                                            noise=noise)
    self.num_particles = num_particles
    self.d_model = d_model
    self.output_size = output_size
    self.seq_len = seq_len
    self.noise = noise
    self.sigma_obs = sigma_obs
    self.training = training
    self.resampling = resampling
    if self.num_particles > 1:
      assert self.resampling == True
      assert self.training == False
    if self.training:
      assert self.noise == False
    self.task_type = task_type
    if self.task_type == 'classification':
      assert self.output_size > 1
    # for unit tests of SMC_Transformer_cell & SMC_transformer
    self.test = test
    # output layer for computing the weights
    self.output_layer = tf.keras.layers.Dense(output_size, name='output_layer')
    # to store weights and indices in case of SMC algo.
    self.list_weights, self.list_indices = [], []
    #------------- state_size and output_size of the SMC Cell (without the batch dimension)-----------------------------------------
    # internal states: K,V,R.
    self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  R=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]))
    self.output_size = (tf.TensorShape([self.num_particles, 1, self.seq_len]))  # attention_weights

    super(SMC_Transf_Cell, self).__init__(**kwargs)

  def compute_w_classification(self, predictions, y):
    '''
    :param predictions: output of final layer (logits.)
    :param x: current sequence element (x_t) >
    :return:
    '''
    log_probas = tf.nn.softmax(predictions, axis=-1)  # shape (B,P,1,V)
    w = tf.gather(log_probas, y, axis=-1, batch_dims=1)
    w = tf.squeeze(w, axis=-1)  # shape (B,P,1)
    w_squeezed = tf.squeeze(w, axis=-1)  # shape (B,P)
    return w_squeezed  # shape (B,P)

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
    # TODO: replace a the tf.cast by an assert (input data should be of dtype=tf.float32 for the regression case).
    y = tf.cast(y, dtype=tf.float32)  # y of shape (B,F) for classif case / shape (B,F) for time_series case.
    # expanding and tiling x over the particle dimensions to have the right shape
    y = tf.expand_dims(y, axis=1) # (B,1,F)
    y = tf.tile(y, multiples=[1, self.num_particles, 1])

    mu_t = y - predictions # (B,P,F)
    log_w = tf.matmul(mu_t, mu_t, transpose_b=True)  # (B,P,P)
    log_w = tf.linalg.diag_part(log_w)  # take the diagonal.
    log_w = tf.scalar_mul(-1 / (2 * (self.sigma_obs) ** 2), log_w) # omega here is the stddev.
    log_w_min = tf.reduce_min(log_w, axis=-1, keepdims=True)
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

    x, y = tf.nest.flatten(inputs)  # unnesting inputs
    K, V, R = states # getting states

    # multi-head attention:
    (z, K, V), attn_weights = self.attention_smc(inputs=x, timestep=self.dec_timestep, K=K, V=V)
    predictions = self.output_layer(z)  # (B,P,1,V)

    # storing z in R:
    R_past = R[:,:,:self.dec_timestep,:]
    R_future = R[:,:,self.dec_timestep+1:,:]
    R = tf.concat([R_past, z, R_future], axis=-2)

    # -------- SMC Algo at inference .... ---------------------------------------------------------------------------------------------------------
    if self.noise:
      # computing resampling weights
      if self.task_type == 'classification':
        w_squeezed = self.compute_w_classification(predictions=predictions, x=y)
      elif self.task_type == 'regression':
        w_squeezed = self.compute_w_regression(predictions=predictions, y=y)
      i_t = tf.random.categorical(w_squeezed, self.num_particles)  # (B,P,1)
      # resample K, V, and R
      if self.resampling:
        K = resample(params=K, i_t=i_t, t=self.dec_timestep)
        V = resample(params=V, i_t=i_t, t=self.dec_timestep)
        R = resample(params=R, i_t=i_t, t=self.dec_timestep)

      w = w_squeezed.numpy()
      i_t = i_t.numpy()
      self.list_weights.append(w)
      self.list_indices.append(i_t)

    output = [attn_weights] # attn_weights > shape (B,P,1,D). Others (B,P).
    new_states = NestedState(K=K, V=V, R=R)
    self.dec_timestep += 1

    if self.test:
      print('end of decoding timestep')
      print('<---------------------------------------------------------------------------------------------------------------------------------------------->')

    return output, new_states

if __name__ == "__main__":
  batch_size = 8
  d_model = 12
  num_heads = 1
  target_vocab_size = 1
  num_particles = 5
  seq_len = 4
  sigma = 1
  noise = False
  data_type = 'time_series_multi'
  task_type = 'regression'


