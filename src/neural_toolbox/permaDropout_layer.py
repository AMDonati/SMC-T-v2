# from uncertainties code.

# -*- coding: utf-8 -*-
import tensorflow as tf

class PermaDropout(tf.keras.layers.Layer):
  """Applies permanent Dropout to the input.
  PermaDropout consists in randomly setting
  a fraction `rate` of input units to 0 at each update.
  # Arguments
      rate: float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the
          binary dropout mask that will be multiplied with the input.
          For instance, if your inputs have shape
          `(batch_size, timesteps, features)` and
          you want the dropout mask to be the same for all timesteps,
          you can use `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.
  # References
      - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
         http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
  """
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(PermaDropout, self).__init__(**kwargs)
    self.rate = min(1., max(0., rate))
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    if self.noise_shape is None:
        return self.noise_shape

    symbolic_shape = tf.keras.backend.shape(inputs)
    noise_shape = [symbolic_shape[axis] if shape is None else shape
                   for axis, shape in enumerate(self.noise_shape)]
    return tuple(noise_shape)

  def call(self, inputs, training=True):
    if 0. < self.rate < 1.:
        noise_shape = self._get_noise_shape(inputs)

        def dropped_inputs():
            return K.dropout(inputs, self.rate, noise_shape,
                             seed=self.seed)
        return tf.keras.backend.in_train_phase(dropped_inputs, inputs,
                                training=True)
    return inputs

  def get_config(self):
    config = {'rate': self.rate,
              'noise_shape': self.noise_shape,
              'seed': self.seed}
    base_config = super(PermaDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    return input_shape