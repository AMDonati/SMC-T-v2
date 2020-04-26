import tensorflow as tf

# additional imports
from models.SMC_Transformer.self_attention_classic import MultiHeadAttention_classic
from neural_toolbox.classic_layers import point_wise_feed_forward_network
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask

# original DecoderLayer from TF 2.0 tutorial on Tranformer
class DecoderLayer(tf.keras.layers.Layer):
  '''adaptated version of the original Decoder Layer of the Transformer.
  The only difference are the shapes of the input tensor (B, P, S, D) instead of (B, S, D)
  and the eventual injection of a (reparametrized) gaussian noise in the attention vector z.
  -args:
    -d_model: model depth
    -num_heads: number of heads in the multi-head attention mechanism
    -dff: output dimension of the feed forward network
    -num_particles: number of simulated particles for the latent state space of the Transformer
    -sigma: 'learned' or 'constant': value of the covariance matrix when injecting noise in the multi-head attention equations.
    -noise: Boolean. True if noise is injected in the computation of the attention vector z, False if not.
    -rate: dropout rate for output layers
  '''

  def __init__(self, d_model, num_heads, dff, num_particles, sigma, noise, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.num_particles = num_particles

    self.mha1 = MultiHeadAttention_classic(d_model=d_model,
                                           num_heads=num_heads,
                                           num_particles=num_particles,
                                           sigma=sigma,
                                           noise=noise)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

    self.noise=noise

  def call(self, inputs, training, look_ahead_mask):
    '''
    -args:
        -inputs: input work or output of the previous layer > shape (B,P,S,D)
        -training: boolean to distinct between training and evaluation phase.
        -look_ahead_mask: for masking future decoding timestep
        -padding_mask: for fixed-size words sequence.
    -returns
        -r0:T output of the Decoder layer > dim (B, P, S, D)
        -reparametrized gaussian noise for the current layer (to compute the loss)
    '''
    # preparing inputs_mha[x,x,x (x float] for mha class.
    inputs_float = tf.cast(inputs, dtype=tf.float32)
    inputs_mha = [inputs_float for _ in range(3)]
    # computing multi-head attention.
    (Z, K, V), attention_weights = self.mha1(inputs=inputs_mha, mask=look_ahead_mask)  # shape (B,P,S,D).
    attn1 = self.dropout1(Z, training=training)
    out1 = self.layernorm1(attn1 + inputs_float)  # TODO: Bug with multivariate

    ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

    return out3, self.mha1.stddev, attention_weights  # attn_weights_block1 # shapes (B,P,S,D), (B,P,S,D).


if __name__ == "__main__":
  d_model = 512
  dff = 2048
  num_heads = 8
  num_particles = 10
  noise=False
  sigma=1

  sample_decoder_layer = DecoderLayer(d_model=d_model,
                                      dff=dff,
                                      num_heads=num_heads,
                                      num_particles=num_particles,
                                      sigma=sigma,
                                      noise=noise)

  inputs_layer = tf.ones((64, 10, 50, 512), dtype=tf.int32)
  seq_len = tf.shape(inputs_layer)[2]
  mask = create_look_ahead_mask(seq_len)
  sample_decoder_layer_output, stddev, attention_weights = sample_decoder_layer(inputs=inputs_layer, look_ahead_mask=mask, training=False)
  print('output of classic decoder layer', sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)


