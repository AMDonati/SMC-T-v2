import tensorflow as tf

def softmax_layer(logits, labels, num_labels, mask):
  logits = tf.reshape(logits, [-1, num_labels])
  labels = tf.reshape(labels, [-1])
  mask = tf.cast(mask, dtype=tf.float32)
  one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
  loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
  loss *= tf.reshape(mask, [-1])
  loss = tf.reduce_sum(loss)
  total_size = tf.reduce_sum(mask)
  total_size += 1e-12  # to avoid division by 0 for all-0 weights
  loss /= total_size
  # predict not mask we could filtered it in the prediction part.
  probabilities = tf.math.softmax(logits, axis=-1)
  predict = tf.math.argmax(probabilities, axis=-1)
  return loss, predict

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', name='FFN1_after_mha_dff'),  # (B, S, dff)
      tf.keras.layers.Dense(d_model, name='FFN2_after_mha_dmodel')  # (B, P, S, d_model)
  ])

class Linear(tf.keras.layers.Layer):
    def __init__(self, name, units=32, use_bias=True, kernel_init=None, bias_init=None):
      super(Linear, self).__init__(name=name)
      self.units = units
      self.kernel_init = kernel_init
      self.bias_init = bias_init
      self.use_bias = use_bias

    def build(self, input_shape):
      self.w = self.add_weight(name="kernel",
        shape=(input_shape[-1], self.units),
        initializer="glorot_uniform",
        trainable=True,
      )
      if self.use_bias:
        self.b = self.add_weight(name="bias",
          shape=(1, self.units), initializer="zeros", trainable=True
        )
      if self.kernel_init is not None:
        if self.bias_init is not None:
          list_weights = [self.kernel_init, self.bias_init]
        else:
          list_weights = [self.kernel_init]
        self.set_weights(list_weights)

    def call(self, inputs):
      if self.use_bias:
        outputs = tf.tensordot(inputs, self.w, axes=[-1,0]) + self.b
      else:
        outputs = tf.tensordot(inputs, self.w, axes=[-1, 0])
      #outputs_ = tf.matmul(inputs, self.w) + self.b
      return outputs


class MLP(tf.keras.layers.Layer):
  def __init__(self, name, units_1, units_2, use_bias_1=True, use_bias_2=True, kernel_1_init=None, kernel_2_init=None, bias_init_1=None, bias_init_2=None):
    super(MLP, self).__init__(name=name)
    self.linear_1 = Linear(name=name+'/linear_1', units=units_1, use_bias=use_bias_1, kernel_init=kernel_1_init, bias_init=bias_init_1)
    self.linear_2 = Linear(name=name+'/linear_2', units=units_2, use_bias=use_bias_2, kernel_init=kernel_2_init, bias_init=bias_init_2)

  def call(self, inputs):
    x = self.linear_1(inputs)
    x = tf.nn.gelu(x)
    x = self.linear_2(x)
    return x

# class MyDenseLayer(tf.keras.layers.Layer):
#   def __init__(self, num_outputs, kernel_init=None):
#     super(MyDenseLayer, self).__init__()
#     self.num_outputs = num_outputs
#     self.kernel_init = kernel_init
#
#   def build(self, input_shape):
#     self.kernel = self.add_weight("kernel",
#                                   shape=[int(input_shape[-1]),
#                                          self.num_outputs])
#     if self.kernel_init is not None:
#       self.set_weights([self.kernel_init])
#
#   def call(self, inputs):
#     return tf.matmul(inputs, self.kernel)

# layer = MyDenseLayer(10)

if __name__ == '__main__':
  print("....test linear....")
  inputs = tf.random.uniform(shape=(6,1))
  layer = Linear(name="dense_1", units=1)
  outputs = layer(inputs)
  inputs_2 = tf.random.uniform(shape=(8,6,1))
  layer_2 = Linear(name="dense_2", units=1, kernel_init=tf.ones(shape=(1,1)).numpy(), bias_init=tf.zeros(shape=(1,1)).numpy())
  dense_layer = tf.keras.layers.Dense(1, kernel_initializer='ones',
    bias_initializer='zeros')
  outputs_2 = layer_2(inputs_2)
  outputs_dense = dense_layer(inputs_2)
  print(outputs_dense == outputs_2)

  print("...test MLP....")
  inputs = tf.random.uniform(shape=(8, 6, 768))
  mlp = MLP(name='mlp', units_1=3072, units_2=768)
  outputs = mlp(inputs)



