import tensorflow as tf

# ----- scaled_dot_product_attention FUNCTION--------------------------------------------------------------------------------------

def self_attention_classic(Q, K, V, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.
  Args:
    q: query shape == (B,P,H,S,D)
    k: key shape == (B,P,H,S,D)
    v: value shape == (B,P,S,H,D)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (B,P,S,S) # to check.

  # scale matmul_qk
  dk = tf.cast(tf.shape(K)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B,P,H,S,S)

  output = tf.matmul(attention_weights, V)  # (B,P,H,S,D)

  return output, attention_weights

## ------ Multi-head attention CLASS------------------------------------------------------------------------------------------------

class MultiHeadAttention_classic(tf.keras.layers.Layer):
  '''
  multi-head attention mechanism for each layer of the Transformer.
  -args:
    -d_model: depth model
    -num_heads: number of heads for the multi-head attention mechanism
    -num_particles: number of particles to generate
    -sigma: constant, 'learned' (for learned noise)
    -noise: boolean: True if noise injected in the attention context vector z, False if no noise injected.
  -returns: attention parameters (Z,K,V) of shape (B,P,S,D).
    '''

  def __init__(self, d_model, num_heads, num_particles, sigma, noise):  # 2 arguments added: dec_timestep, mode.
    super(MultiHeadAttention_classic, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

    # additionnal parameters for SMC algorithm.
    self.num_particles = num_particles
    self.sigma_scalar=sigma
    self.noise=noise


  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    (batch_size, num_particle, seq_length, d_model) => (batch_size, num_particle, seq_length, num_heads, depth=d_model/num_heads)
    """
    x = tf.reshape(x, (batch_size, self.num_particles, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 1, 3, 2, 4])

  def concat_heads(self, x):
    '''concat attention parameters over all heads (and permute dimensions)
    -returns a tensor of shape (B, P, S, D)'''
    scaled_attention = tf.transpose(x, perm=[0, 1, 3, 2, 4])  # (batch_size, NUM_PARTICLES, seq_len_q, num_heads, depth)

    return tf.reshape(scaled_attention,
                      (tf.shape(scaled_attention)[0],
                       tf.shape(scaled_attention)[1],
                       -1,
                       self.d_model))  # (batch_size, NUM_PARTICLES, seq_len_q, d_model)

  def call(self, inputs, mask, seed=123):
    '''
    -Args:
      -inputs[q,k,v]: v(k), k(k), q(k): attention parameters (over all heads) @ current decoding timestep. > shape (B,P,D)
      -mask: look_ahead mask.
      -K,V,Z: KO:k, V0:k, Z0:k: total length attention parameters (until decoding timestep) > shape (B, P, S, D)
    -Returns:
      -K:0:k+1, V0:k+1, Z0:k+1
      -attention_weights
    '''
    q=inputs[0] # (B,P,H,S,D/H)
    k=inputs[1]
    v=inputs[2]

    batch_size = tf.shape(v)[0]

    # computing the Q,K,V from the v,k,q parameters.
    #q=tf.cast(q, dtype=tf.int32)
    Q = self.wq(q)  # (B,P,S,D)
    K = self.wk(k)  # (B,P,S,D)
    V = self.wv(v)  # (B,P,S,D)

    # splitting heads to do multi_head attention.
    Q = self.split_heads(Q, batch_size)  # (B,P,H,S,D/H)
    K = self.split_heads(K, batch_size)  # (B,P,H,S,D/H)
    V = self.split_heads(V, batch_size)  # (B,P,H,S,D/H)

    #TODO: add a mask for the time-window considered.
    scaled_attention, attention_weights= self_attention_classic(Q, K, V, mask) # (B,P,H,S,D/H)

    # concat attention, K, V over all the heads
    concat_attention = self.concat_heads(scaled_attention) # shape (B,P,S,D)
    K=self.concat_heads(K) # shape (B,P,S,D)
    V=self.concat_heads(V) # shape (B,P,S,D)

    #------------------Add the noise using the reparametrization trick---------------------------------------------------------
    d_model = self.d_model

    # initialize sigma as a 'positive' diagonal matrix as a start
    if self.sigma_scalar=='learned':
      self.sigma=tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(d_model,), dtype=tf.float32)),
                             dtype=tf.float32) # shape (D,D)
      # apply tf.stop_gradient on sigma to avoid backprop for this set of parameters
      #TODO: At the end, remove the tf.stop_gradient and use a simple SGD algo to update this parameter.
      self.sigma=tf.stop_gradient(self.sigma)
      self.sigma = tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(d_model,))), dtype=tf.float32)
    else:
      sigma_tensor=tf.constant(self.sigma_scalar, shape=(d_model,), dtype=tf.float32)
      self.sigma = tf.Variable(tf.linalg.diag(sigma_tensor), dtype=tf.float32) # (D,D)

    #TODO: add an assert to check that self.sigma is inversible.

    # compute the $\epsilon$ of the reparametrized noise.
    if self.noise:
      self.stddev = tf.random.normal(shape=tf.shape(concat_attention), seed=seed, name='stddev') # (B,P,S,D)
    else:
      # self.stddev is null if no noise
      self.stddev = tf.zeros(shape=tf.shape(concat_attention), dtype=tf.float32)

    # tensordot multiplication for sigma and epsilon (fixed gaussian noise)
    stddev = tf.tensordot(self.sigma, self.stddev, axes=[0, 3])  # shape (D,B,1,D)
    # (if self.stddev is a zero tensor, then stddev is also a zero tensor. test done).
    stddev = tf.transpose(stddev, perm=[1, 2, 3, 0]) # (B,P,S,D)

    Z = self.dense(concat_attention) + stddev

    return (Z, K, V), attention_weights  # attention_weights

if __name__ == "__main__":
  B=64
  P=10
  H=2
  D=12
  S=20
  noise=False

  #----------------------test of self_attention_classic----------------------------------------------------

  X = tf.ones(shape=(B, P, H, S, D), dtype=tf.float32)
  K = tf.random.uniform(shape=(B, P, H, S, D))
  V = tf.random.uniform(shape=(B, P, H, S, D))

  output, attention_weights = self_attention_classic(X, X, X, mask=None)
  print('temp_out', output.shape)

  # ----------------------test of MultiHeadAttention classic----------------------------------------------------

  temp_mha = MultiHeadAttention_classic(d_model=D, num_heads=H, num_particles=10, sigma=1, noise=noise)
  X_mha = tf.ones(shape=(B, P, S, D), dtype=tf.float32)
  inputs_mha=[X_mha for _ in range(3)]
  (Z, K, V), attention_weights = temp_mha(inputs=inputs_mha, mask=None)
  print('Z', Z.shape)
  print('K', K.shape)
  print('V', V.shape)
  print('attention_weights', attention_weights.shape)

