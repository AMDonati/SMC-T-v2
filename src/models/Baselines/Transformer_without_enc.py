import tensorflow as tf
from models.SMC_Transformer.transformer_utils import positional_encoding
from neural_toolbox.classic_layers import point_wise_feed_forward_network
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from models.Baselines.Attention_Transformer import MultiHeadAttention
from models.Baselines.Attention_Transformer import OneHeadAttention


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate, full_model):
    super(DecoderLayer, self).__init__()
    self.full_model = full_model
    self.rate = rate

    if full_model:
      self.ffn = point_wise_feed_forward_network(d_model, dff)
      self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.dropout1 = tf.keras.layers.Dropout(rate)
      self.dropout3 = tf.keras.layers.Dropout(rate)

    if num_heads > 1:
      self.mha = MultiHeadAttention(d_model, num_heads)
    else:
      print("Transformer with One-Head Attention...")
      self.mha = OneHeadAttention(d_model)

  def call(self, inputs, training, look_ahead_mask):
    input = inputs
    inputs = [tf.cast(inputs, dtype=tf.float32) for _ in range(3)]

    attn1, attn_weights = self.mha(inputs=inputs, mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)

    if self.full_model:
      attn1 = self.dropout1(attn1, training=training) # (B,S,D)
      out1 = self.layernorm1(attn1 + input) # (B,S,D)
      ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
      ffn_output = self.dropout3(ffn_output, training=training)
      out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)
    else:
      out3 = attn1

    return out3, attn_weights

class Decoder(tf.keras.layers.Layer):
  '''Class Decoder with the Decoder architecture
  -args
    '''
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate,
               full_model):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.dff = dff
    self.num_layers = num_layers
    self.maximum_position_encoding = maximum_position_encoding
    self.rate = rate
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.input_dense_projection = tf.keras.layers.Dense(d_model) # for regression case (multivariate > to be able to have a d_model > F).
    if maximum_position_encoding is not None:
      self.pos_encoding = positional_encoding(position=maximum_position_encoding, d_model=d_model)
    self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate, full_model=full_model) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(self.rate)
    self.full_model = full_model

  def call(self, inputs, training, look_ahead_mask):
    seq_len = tf.shape(inputs)[1]
    attention_weights = {}
    inputs = self.input_dense_projection(inputs)
    inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) #TODO: add this on the SMC_Transformer as well?

    if self.maximum_position_encoding is not None:
      assert self.maximum_position_encoding >= seq_len
      inputs += self.pos_encoding[:, :seq_len, :]

    inputs = self.dropout(inputs, training=training)

    for i in range(self.num_layers):
      inputs, block = self.dec_layers[i](inputs=inputs,
                                         training=training,
                                         look_ahead_mask=look_ahead_mask)
      attention_weights['decoder_layer{}'.format(i + 1)] = block

    return inputs, attention_weights # (B,S,S)

"""## Create the Transformer
The transTransformer consists of the decoder and a final linear layer. 
The output of the decoder is the input to the linear layer and its output is returned.
"""

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate,
               full_model):
    super(Transformer, self).__init__()
    self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                           target_vocab_size=target_vocab_size, maximum_position_encoding=maximum_position_encoding,
                           rate=rate, full_model=full_model)
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def stop_SMC_algo(self):
    self.decoder.dec_layers[-1].mha.noise = False

  def call(self, inputs, training, mask):
    '''
    :param inputs: input data > shape (B,S,1) # CAUTION.... not the same shape as smc_transformer.
    :param training: Boolean.
    :param mask: look_ahead_mask to mask the future.
    :return:
    final_output (log probas of predictions > shape (B,S,C or V).
    attention_weights
    '''

    dec_output, attention_weights = self.decoder(inputs=inputs, training=training, look_ahead_mask=mask) # (B,S,D)
    final_output = self.final_layer(dec_output)  # (B, S, F_y)

    return final_output, attention_weights

if __name__ == "__main__":
  B = 8
  F = 3
  num_layers = 1
  d_model = 64
  num_heads = 1
  dff = 128
  maximum_position_encoding = None # needs to be None if we have an input_tensor of shape (B,P,S,D).
  data_type = 'time_series_multi'
  C = F
  S = 20
  rate = 0

  sample_transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                   target_vocab_size=C, maximum_position_encoding=maximum_position_encoding, rate=rate,
                                   full_model=False)

  temp_input = tf.random.uniform((B, 10, 1, F), dtype=tf.float32, minval=0, maxval=200)

  mask = create_look_ahead_mask(S)

  fn_out, attn_weights = sample_transformer(inputs=temp_input,
                                 training=True,
                                 mask=None) # mask need to be None for input of seq_len = 1.

  print('model output', fn_out.shape) # (B,S,C)