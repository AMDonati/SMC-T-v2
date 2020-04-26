# imports
import tensorflow as tf
from neural_toolbox.SMC_TransformerCell import SMC_Transf_Cell
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
import collections

# use this instead: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN?version=stable
NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])

#------------------------CREATE THE SMC TRANSFORMER MODEL ------------------------------------------------------------

class SMC_Transformer(tf.keras.Model):

  def __init__(self, d_model,
               output_size, num_particles, seq_len, sigmas_att, sigma_obs, noise, data_type, task_type,
               training, resampling=True, test=False):
    super(SMC_Transformer, self).__init__()

    self.input_dense_projection = tf.keras.layers.Dense(d_model, name='projection_layer_ts')
    self.cell = SMC_Transf_Cell(d_model=d_model,
                                output_size=output_size,
                                num_particles=num_particles,
                                seq_len=seq_len,
                                sigmas_att=sigmas_att,
                                noise=noise,
                                sigma_obs=sigma_obs,
                                task_type=task_type,
                                resampling=resampling,
                                training=training,
                                test=test)

    # for pre_processing words in the one_layer case.
    if task_type == 'classification':
      self.embedding = tf.keras.layers.Embedding(output_size, d_model)
    self.final_layer = self.cell.output_layer
    self.output_size = output_size
    self.num_particles = num_particles
    self.d_model = d_model
    self.data_type = data_type
    self.task_type = task_type
    self.seq_len = seq_len
    self.sigmas_att = sigmas_att
    self.sigma_obs = sigma_obs
    self.noise = noise
    # to test the class SMC_Transformer.
    self.test = test


  def call(self, inputs, targets):
    '''

    '''
    # check dimensionality of inputs (B,S,F)
    assert len(tf.shape(inputs)) == 3
    batch_size = tf.shape(inputs)[0]
    seq_len = tf.shape(inputs)[1]
    assert self.seq_len == seq_len

    if self.data_type=='nlp':
      input_tensor_processed = self.embedding(inputs)  # (B,S,D)
      input_tensor_processed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # division by the root of the d_model
    elif self.data_type=='time_series_uni' or "time_series_multi":
      input_tensor_processed = self.input_dense_projection(inputs) # (B,S,D)
    else:
      raise ValueError('wrong data type: should be either "nlp", "time-series_uni", or "time_series_multi"')
    input_tensor_processed = tf.expand_dims(input_tensor_processed, axis=1)  # (B,1,S,D)
    input_tensor_processed = tf.tile(input_tensor_processed, multiples=[1, self.num_particles, 1, 1])  # dim (B,P,S,D)
    x = tf.transpose(input_tensor_processed, perm=[0, 2, 1, 3])  # shape (B,S,P,D) so that it can be processed by the RNN_cell & RNN_layer.

    # 'dummy' initialization of cell's internal state for memory efficiency.
    K0 = tf.zeros(shape=(batch_size, self.num_particles, seq_len, self.depth), dtype=tf.float32)
    V0 = tf.zeros(shape=(batch_size,self.num_particles, seq_len, self.depth), dtype=tf.float32)
    R0 = tf.zeros(shape=(batch_size,self.num_particles, seq_len, self.depth), dtype=tf.float32)
    initial_state = NestedState(K=K0,
                                V=V0,
                                R=R0)

    def step_function(inputs, states):
       return self.cell(inputs, states)

    inputs_for_rnn = NestedInput(x=x, y=targets) # y > (B,S,F,1), #x > (B,S,P,D)

    if self.test:
      print('inputs(x)', inputs)
      print('y', targets)

    last_output, outputs, new_states = tf.keras.backend.rnn(step_function=step_function,
                                                            inputs=inputs_for_rnn,
                                                            initial_states=initial_state)
    # reset decoding timestep of the cell to 1:
    self.cell.dec_timestep = 0

    # ------------------ EXTRACTING OUTPUTS OF THE RNN LAYER ------------------------------------------------------
    indices_matrix = outputs[0]
    w0_T = outputs[1] # (B,S,P,D)
    attn_weights_SMC_layer = outputs[3]  # shape (B,S,P,H,S)

    # states
    K, V, R = new_states[0], new_states[1], new_states[2] # (B,P,S+1,D)
    K = K[:,:,1:,:] # remove first timestep (dummy init.) # (B,P,S,D)
    V = V[:,:,1:,:] # (B,S,P,D)
    R = R[:,:,1:,:] # (B,P,S,D)

    Y0_T = self.final_layer(R) # (B,P,S,C) used to compute the categorical cross_entropy loss. # logits.

    attn_weights = tf.transpose(attn_weights_SMC_layer, perm=[0,2,3,1,4]) #TODO: caution, attention weights - one layer less.

    return (Y0_T, w0_T, indices_matrix, (K,V,R)), attn_weights

if __name__ == "__main__":
  num_particles = 10
  seq_len = 5
  b = 8
  F = 3 # multivariate case.
  num_layers = 1
  d_model = 12
  num_heads = 1
  dff = 128
  maximum_position_encoding = seq_len
  sigma = 'learned'
  omega = 0.25
  data_type = 'time_series_multi'
  task_type = 'regression'
  target_feature = 0
  C = F if target_feature is None else 1
  noise_encoder = False
  noise_SMC_layer = True
  rate = 0.1
  test = True

  ####---------test of Transformer class--------------------------------------------------------------------------------

  inputs = tf.constant([[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]], shape=(1, seq_len, F), dtype=tf.int32) # ok works with len(tf.shape(inputs)==3.
  inputs = tf.tile(inputs, multiples=[b,1,1])

  mask = create_look_ahead_mask(seq_len)

  (predictions, trajectories, weights, (K,V,U)), attn_weights = sample_transformer(inputs=inputs,
                                                                                    training=True,
                                                                                    mask=mask)
  print('final predictions - one sample', predictions[0,:,:,:])
  print('final K - one sample', K[0,:,:,0])
  print('w_T', weights)

  if num_layers > 1:
    print('attn weights first layer', attn_weights['encoder_layer1'].shape) # shape (B,P,H,S,S)


