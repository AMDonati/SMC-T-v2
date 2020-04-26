import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils

# Implementation of a LSTMCell with particle filtering based on the source code of the Keras LSTM Cell:
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/recurrent_v2.py#L653-L739

class SMC_LSTMCell(tf.keras.layers.LSTMCell):
    """Cell class for the SMC LSTM layer.
    Arguments:
    num_particles: number of particles used to get a set of (c,h) pairs at each timestep of the LSTM
    num_words: number of output classes for the LSTM (Classification setting)
    targets: training targets: used to compute the set of weights at each timestep of the LSTM
    batch_size: size of the batch of training samples
    resampling_method: method used to re-sample (c,h) pairs at each timestep of the LSTM
    noise_initializer: initializer for the noise variable.
    noise_regularizer: regularizer for the noise weights matrix.
    noise_constraints: constraint function applied to the 'kernel' weights matrix.
    output_initializer: initializer for the  output kernel (used for computing the particle filtering weights).
    output_regularizer: regularizer for the output kernel.
    output_constraints: constraint function applied to the 'output kernel' weights.

    ---------other arguments: arguments of the LSTMCell in tf.keras-----------------
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Setting it to true will also force `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """

    def __init__(self,
                 units,
                 num_particles,
                 alpha,
                 num_words,
                 targets, # to remove and put it only in the SMC_LSTM_layer??
                 batch_size,
                 resampling_method,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 output_initializer='glorot_uniform',
                 noise_initializer='glorot_uniform',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 noise_regularizer=None,
                 bias_regularizer=None,
                 output_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 output_constraint=None,
                 noise_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(SMC_LSTMCell, self).__init__(units,
                                           activation='tanh',
                                           recurrent_activation='hard_sigmoid',
                                           use_bias=True,
                                           kernel_initializer='glorot_uniform',
                                           recurrent_initializer='orthogonal',
                                           bias_initializer='zeros',
                                           unit_forget_bias=True,
                                           kernel_regularizer=None,
                                           recurrent_regularizer=None,
                                           bias_regularizer=None,
                                           kernel_constraint=None,
                                           recurrent_constraint=None,
                                           bias_constraint=None,
                                           dropout=0.,
                                           recurrent_dropout=0.,
                                           implementation=1,
                                           **kwargs)

        self.particles=num_particles
        self.alpha=alpha
        self.classes=num_words
        self.targets=targets
        self.batch_size=batch_size
        self.resampling=resampling_method

        self.noise_initializer=noise_initializer
        self.noise_regularizer=noise_regularizer
        self.noise_constraint=noise_constraint
        self.output_initializer=output_initializer
        self.output_regularizer=output_regularizer
        self.output_constraint=output_constraint

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        # kernel shape 2D -> 3D: (Num_particles, input_dim, self.units*4)
        self.kernel = self.add_weight(
            shape=(self.particles, input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        # recurrent kernel keeps the same shape.
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

            # adding the kernel for the learned noise
            self.learned_noise = self.add_weight(
                shape=(self.batch_size, self.particles, self.units),
                name='noise',
                initializer=self.noise_initializer,
                regularizer=self.noise_regularizer,
                constraint=self.noise_constraint)

            # adding the kernel and bias for the output layer (to compute the set of weights)
            self.output_kernel = self.add_weight(
                shape=(self.num_units, self.num_words),
                name='output_kernel',
                initializer=self.output_initializer,
                regularizer=self.output_regularizer,
                constraint=self.output_constraint)
        self.output_bias = self.add_weight(
        shape=(self.num_words,),
            name='output_bias',
            initializer=bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)

        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1

        # one dimension added in the self.recurrent kernel matrix.
        # input gate
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :, :self.units]))
        # forget gate
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, :, self.units:self.units * 2]))
        # carry state (element-wise multiplication)
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, :, self.units * 2:self.units * 3]))
        # output gate
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, :, self.units * 3:]))
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, targets, training=None):
        h_tm1 = states[0]  # previous set of memory states: shape (batch_size, number of particles, number of units)
        c_tm1 = states[1]  # previous set of carry states: shape (batch_size, number of particles, number of units)
        # adding the set of weights
        w_tm1=states[2] # previous set of weights (for SMC): shape (batch_size, number of particles)

        # RESAMPLING STEP FOR LSTM with particle filter
        indices = tf.random.categorical(logits=w_tm1, num_samples=self.particles) # see if we need to use numpy functions instead
        indices = list(K.eval(indices))
        h_sampl = h_tm1[:, indices, :]
        c_sampl = c_tm1[:, indices, :]

        # TO DO: check these 2 functions to see if they need to be adjusted for PF LSTM Cell.
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            # split on last axis
            k_i, k_f, k_c, k_o = array_ops.split(
                self.kernel, num_or_size_splits=4, axis=-1)
            x_i = K.dot(inputs_i, k_i)
            x_f = K.dot(inputs_f, k_f)
            x_c = K.dot(inputs_c, k_c)
            x_o = K.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = array_ops.split(
                    self.bias, num_or_size_splits=4, axis=0)
                x_i = K.bias_add(x_i, b_i)
                x_f = K.bias_add(x_f, b_f)
                x_c = K.bias_add(x_c, b_c)
                x_o = K.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.:
                h_sampl_i = h_sampl * rec_dp_mask[0]
                h_sampl_f = h_sampl * rec_dp_mask[1]
                h_sampl_c = h_sampl * rec_dp_mask[2]
                h_sampl_o = h_sampl * rec_dp_mask[3]
            else:
                h_tm1_i = h_sampl
                h_tm1_f = h_sampl
                h_tm1_c = h_sampl
                h_tm1_o = h_sampl
            x = (x_i, x_f, x_c, x_o)
            h_sampl = (h_sampl_i, h_sampl_f, h_sampl_c, h_sampl_o)
            c, o = self._compute_carry_and_output(x, h_sampl, c_sampl)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_sampl, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z = array_ops.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_sampl)

        h = o * self.activation(c)


        # COMPUTATION OF THE NEW SET OF WEIGHTS
        w = K.softmax(K.dot(h, self.output_kernel) + self.bias_output) # shape (batch_size, num of particles, num of words)
        # w=keras.layers.Dense(self.num_words)(h)
        # w=keras.layers.softmax(w)
        w = w * targets # assuming that targets are one-hot encoded.
        w= K.sum(w, axis=-1)

        # add a linear combination with a uniform sampling (soft resampling: presumed trick from the 'Particle Filter Recurrent Networks' to make everything differentiable)
        # link to the article:
        w_final = self.alpha * w + (1 - self.alpha) * K.random_uniform(shape=[self.batch_size, self.particles],
                                                                       minval=0,
                                                                       maxval=1 / self.particles)

        w_final=w/w_final
        w_final=w_final/K.sum(w_final, axis=-1)

        def average_state(h_states, weights):
            # TO SIMPLIFY???!!!
            mean_weights = tf.expand_dims(weights, axis=-1)
            h_mean = tf.squeeze(tf.matmul(h_states, mean_weights, transpose_a=True))
            sum_weights = tf.tile(tf.expand_dims(tf.reduce_sum(weights, axis=-1), axis=-1), [1, h_mean.shape[-1]])
            h_mean = h_mean / sum_weights
            return h_mean

        # COMPUTATION OF HMEAN
        h_mean=average_state(h, w_final)

        return h_mean, [h, c, w_final]

    def get_config(self):
        # adding the config elements specific to the SMC_LSTM Cell
        config = {
        'units': self.units,
        'num_particles': self.particles,
        'alpha': self.alpha,
        'num_words': self.num_words,
        'resampling_method': self.resampling,
        'batch_size':self.batch_size,
        'noise_initializer':self.noise_initializer,
        'noise_regularizer':self.noise_regularizer,
        'noise_constraint':self.noise_constraint,
        'output_initializer':self.output_initializer,
        'output_regularizer':self.output_regularizer,
        'output_constraint': self.output_constraint,
        }
        base_config = super(SMC_LSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _generate_zero_filled_state_for_cell(self, cell, inputs, batch_size, num_particles, dtype):
        if inputs is not None:
            batch_size = array_ops.shape(inputs)[0]
            dtype = inputs.dtype
        elif batch_size is None or dtype is None:
            raise ValueError(
                'batch_size and dtype cannot be None while constructing initial state: '
                'batch_size={}, dtype={}'.format(batch_size, dtype))

            def _generate_zero_filled_state(batch_size_tensor, state_size, num_particles, dtype):
                ## should return a tuple (state, weights?)
                """Generate a zero filled tensor with shape [batch_size, state_size]."""
                if batch_size_tensor is None or dtype is None:
                    raise ValueError(
                        'batch_size and dtype cannot be None while constructing initial state: '
                        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

                def create_zeros(unnested_state_size, num_particles):
                    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
                    init_state_size = [batch_size_tensor] + [num_particles] + flat_dims
                    return array_ops.zeros(init_state_size, dtype=dtype)

                if nest.is_sequence(state_size):
                    return nest.map_structure(create_zeros, (state_size, num_particles))
                else:
                    return create_zeros(state_size, num_particles)

                # the key here is cell.state_size: see how to create a state as a list of 3 tensors of shape (batch_size, number of particles, state size)
                # look into the Layer Class: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/base_layer.py
            return _generate_zero_filled_state(batch_size, cell.state_size, cell.particles, dtype)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_c=self._generate_zero_filled_state_for_cell(
        self, inputs, batch_size, self.particles, dtype)
        initial_h = self._generate_zero_filled_state_for_cell(
            self, inputs, batch_size, self.particles, dtype)
        initial_w=tf.random_uniform(shape=[self.batch_size, self.paricles], dtype=tf.float32)
        return [initial_c, initial_h, initial_w]
