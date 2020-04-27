import tensorflow as tf

# ----- scaled_dot_product_attention_function & mha function ------------

class Self_Attention_SMC(tf.keras.layers.Layer):

  def __init__(self, d_model):
    super(Self_Attention_SMC, self).__init__()

    self.d_model = d_model
    self.wq = tf.keras.layers.Dense(d_model, name='dense_projection_q')
    self.wk = tf.keras.layers.Dense(d_model, name='dense_projection_k')
    self.wv = tf.keras.layers.Dense(d_model, name='dense_projection_v')
    self.dense = tf.keras.layers.Dense(d_model, name='dense_projection_z')
    self.noise = False

  def add_SMC_parameters(self, dict_sigmas):
    # noise parameters.
    self.sigma_k = dict_sigmas['k']
    self.sigma_q = dict_sigmas['q']
    self.sigma_v = dict_sigmas['v']
    self.sigma_z = dict_sigmas['z']
    self.noise = True

  def call(self, inputs, timestep, K, V):
    '''
    :param inputs: X_t (B,P,1,D) with P = 1 during training.
    :param timestep:
    :param K: (B,P,S,D) with P=1 during training.
    :param V: (B,P,S,D) with P= 1 during training.
    :return:
    '''
    assert len(tf.shape(inputs)) == 4 # (B,P,1,D)

    # computing current k,q,v from inputs
    k = self.wk(inputs) # (B,P,1,D)
    q = self.wq(inputs) # (B,P,1,D)
    v = self.wv(inputs) # (B,P,1,D)

    if self.noise:
      k = k + tf.random.normal(stddev=self.sigma_k, shape=tf.shape(k))
      q = q + tf.random.normal(stddev=self.sigma_q, shape=tf.shape(q))
      v = v + tf.random.normal(stddev=self.sigma_v, shape=tf.shape(v))

    K_past = K[:, :, :timestep, :]
    K_future = K[:, :, timestep + 1:, :]
    K = tf.concat([K_past, k, K_future], axis=2) # (B,P,S,D)
    V_past = V[:, :, :timestep, :]
    V_future = V[:, :, timestep + 1:, :]
    V = tf.concat([V_past, v, V_future], axis=2) # (B,P,S,D)

    # Computation of z from K,V,q.
    matmul_qk = tf.matmul(q, K, transpose_b=True)  # (B, P, 1, S)
    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (B,P,1,S)
    # softmax to get pi:
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B, P, 1, S)
    z = tf.matmul(attention_weights, V)  # (B,P,1,S)
    z = self.dense(z)

    if self.noise:
      z = z + tf.random.normal(stddev=self.sigma_z, shape=tf.shape(z))

    return (z, K, V), attention_weights


if __name__ == "__main__":
  B = 8
  S = 20
  d_model = 512
  dec_timestep = 1

  x = tf.ones(shape=(B, 1, 1, d_model))
  K = tf.random.uniform(shape=(B, 1, S, d_model))
  V = tf.random.uniform(shape=(B, 1, S, d_model))

  temp_attention = Self_Attention_SMC(d_model)
  (temp_z, temp_K, temp_V), attn_weights = temp_attention(x, dec_timestep, K, V)
  print('temp_out', temp_z.shape)
  print('temp_K', temp_K.shape)
  print('temp_V', temp_V.shape)
  print('attention_weights', attn_weights.shape)

  # test with noise and more than one particule
  num_particles = 10
  sigma = 0.1
  x = tf.ones(shape=(B, num_particles, 1, d_model))
  K = tf.random.uniform(shape=(B, num_particles, S, d_model))
  V = tf.random.uniform(shape=(B, num_particles, S, d_model))
  dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigma for _ in range(4)]))
  temp_attention.add_SMC_parameters(dict_sigmas=dict_sigmas)
  (temp_z, temp_K, temp_V), attn_weights = temp_attention(x, dec_timestep, K, V)
  print('temp_out', temp_z.shape)
  print('temp_K', temp_K.shape)
  print('temp_V', temp_V.shape)
  print('attention_weights', attn_weights.shape)



