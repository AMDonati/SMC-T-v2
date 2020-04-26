import tensorflow as tf

# ----- scaled_dot_product_attention_function & mha function ------------

class Self_Attention_SMC(tf.keras.layers.Layer):

  def __init__(self, d_model, num_particles, sigma_k, sigma_q, sigma_v, sigma_z, noise):
    super(Self_Attention_SMC, self).__init__()

    self.d_model = d_model
    self.wq = tf.keras.layers.Dense(d_model, name='dense_projection_q')
    self.wk = tf.keras.layers.Dense(d_model, name='dense_projection_k')
    self.wv = tf.keras.layers.Dense(d_model, name='dense_projection_v')
    self.dense = tf.keras.layers.Dense(d_model, name='dense_projection_z')
    self.num_particles = num_particles #TODO: to remove?
    # noise parameters.
    self.sigma_k = sigma_k
    self.sigma_q = sigma_q
    self.sigma_v = sigma_v
    self.sigma_z = sigma_z
    self.noise = noise

  def call(self, inputs, timestep, K, V):
    '''
    :param inputs: X_t (B,P,1,D)
    :param timestep:
    :param K: (B,P,S,D)
    :param V: (B,P,S,D)
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
    z = tf.matmul(attention_weights, V)  # (B,P,1,D)
    z = self.dense(z)

    if self.noise:
      z = z + tf.random.normal(stddev=self.sigma_z, shape=tf.shape(z))

    return (z, K, V), attention_weights


if __name__ == "__main__":
  #TODO: test this function.
  B = 64
  num_particles = 10
  num_heads = 8
  S = 20
  d_model = 512
  dec_timestep = 20
  sigma = 'learned'
  noise = True

  x = tf.ones(shape=(B, num_particles, num_heads, 1, int(d_model/num_heads)))
  K = tf.random.uniform(shape=(B, num_particles, num_heads, S, int(d_model/num_heads)))
  V = tf.random.uniform(shape=(B, num_particles, num_heads, S, int(d_model/num_heads)))

  (temp_out, temp_K, temp_V), attn_weights = Self_Attention_SMC(x, dec_timestep, K, V)
  print('temp_out', temp_out.shape)
  print('temp_K', temp_K.shape)
  print('temp_V', temp_V.shape)


