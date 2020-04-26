import tensorflow as tf
from models.SMC_LSTMCell import SMC_LSTMCell
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops

# Implementation of a LSTM layer with particle filtering using the SMC_LSTM Cell and based on the LSTM layer of keras:
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/recurrent_v2.py#L743-L987
# Might be actually smarter to start directly from the tf.keras.layers.RNN class, as a re-implementation of the 'call' function of this class is needed.
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/recurrent.py#L183-L889

class LSTM(tf.keras.layers.LSTM):
  """Long Short-Term Memory layer - Hochreiter 1997.
   Note that this cell is not optimized for performance on GPU. Please use
  `tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU.
  Arguments:
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
      used for the linear transformation of the inputs..
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
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
    return_sequences: Boolean. Whether to return the last output.
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               num_particles,
               alpha,
               num_words,
               targets,  # to remove??
               batch_size,
               resampling_method,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               output_initializer='glorot_uniform',
               noise_initializer='glorot_uniform',
               activity_regularizer=None,
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
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')

    # replacing the
    cell = SMC_LSTMCell(
        num_particles,
        alpha,
        num_words,
        targets,  # to remove??
        batch_size,
        resampling_method,
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation)

    super(LSTM, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, targets, mask=None, training=None, initial_state=None):
    # RECODE THE FUNCTION CALL "MANUALLY" (BY TAKING THE CODE OF THE RNN CALL FUNCTION)
    self.cell.reset_dropout_mask()
    self.cell.reset_recurrent_dropout_mask()
    return super(LSTM, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  #------------------------------

  def call(self,
           inputs,
           targets, #add targets + batch_size?
           mask=None,
           training=None,
           initial_state=None,
           constants=None):
      inputs, initial_state, constants = self._process_inputs(
          inputs, initial_state, constants)

      if mask is not None:
          # Time step masks must be the same for each input.
          # TODO(scottzhu): Should we accept multiple different masks?
          mask = nest.flatten(mask)[0]

      if nest.is_sequence(inputs):
          # In the case of nested input, use the first element for shape check.
          input_shape = K.int_shape(nest.flatten(inputs)[0])
      else:
          input_shape = K.int_shape(inputs)
      timesteps = input_shape[0] if self.time_major else input_shape[1]
      if self.unroll and timesteps is None:
          raise ValueError('Cannot unroll a RNN if the '
                           'time dimension is undefined. \n'
                           '- If using a Sequential model, '
                           'specify the time dimension by passing '
                           'an `input_shape` or `batch_input_shape` '
                           'argument to your first layer. If your '
                           'first layer is an Embedding, you can '
                           'also use the `input_length` argument.\n'
                           '- If using the functional API, specify '
                           'the time dimension by passing a `shape` '
                           'or `batch_shape` argument to your Input layer.')

      kwargs = {}
      if generic_utils.has_arg(self.cell.call, 'training'):
          kwargs['training'] = training

      # TF RNN cells expect single tensor as state instead of list wrapped tensor.
      is_tf_rnn_cell = getattr(self.cell, '_is_tf_rnn_cell', None) is not None
      if constants:
          if not generic_utils.has_arg(self.cell.call, 'constants'):
              raise ValueError('RNN cell does not support constants')

          def step(inputs, states):
              constants = states[-self._num_constants:]  # pylint: disable=invalid-unary-operand-type
              states = states[:-self._num_constants]  # pylint: disable=invalid-unary-operand-type

              states = states[0] if len(states) == 1 and is_tf_rnn_cell else states

              ### LIGN TO CHANGE!!!! adapt with the function call of the SMC_LSTMCell
              output, new_states = self.cell.call(
                  inputs, states, constants=constants, **kwargs)
              if not nest.is_sequence(new_states):
                  new_states = [new_states]
              return output, new_states
      else:

          def step(inputs, states):
              states = states[0] if len(states) == 1 and is_tf_rnn_cell else states
              output, new_states = self.cell.call(inputs, states, **kwargs)
              if not nest.is_sequence(new_states):
                  new_states = [new_states]
              return output, new_states


    # BACK TO TENSORFLOW code with K.rnn???
      last_output, outputs, states = K.rnn(
          step,
          inputs,
          initial_state,
          constants=constants,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      if self.stateful:
          updates = []
          for state_, state in zip(nest.flatten(self.states), nest.flatten(states)):
              updates.append(state_ops.assign(state_, state))
          self.add_update(updates, inputs)

      if self.return_sequences:
          output = outputs
      else:
          output = last_output

      if self.return_state:
          if not isinstance(states, (list, tuple)):
              states = [states]
          else:
              states = list(states)
          return generic_utils.to_list(output) + states
      else:
          return output


  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(LSTM, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)