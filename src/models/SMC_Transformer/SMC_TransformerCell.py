import tensorflow as tf
import collections
# additional imports
from src.models.SMC_Transformer.self_attention_SMC import Self_Attention_SMC
from src.models.SMC_Transformer.multi_head_attention_SMC import mha_SMC
from src.models.SMC_Transformer.transformer_utils import resample
from src.models.classic_layers import point_wise_feed_forward_network
from src.models.SMC_Transformer.transformer_utils import create_look_ahead_mask

NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])


class SMC_Transf_Cell(tf.keras.layers.Layer):
    def __init__(self, d_model, output_size, seq_len, full_model, dff, sampl_freq=1, num_heads=1, attn_window=None, **kwargs):
        '''
        :param attn_window:
        :param full_model:
        :param dff:
        '''
        # store the decoding timestep
        self.dec_timestep = 0
        self.cell_count = 0
        self.attention_smc = Self_Attention_SMC(d_model=d_model, attn_window=attn_window) if num_heads == 1 else mha_SMC(d_model=d_model, num_heads=num_heads, attn_window=attn_window)
        self.d_model = d_model
        self.output_size = output_size
        self.seq_len = seq_len
        self.full_model = full_model
        self.sampl_freq = sampl_freq

        if self.full_model:
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm1')
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm2')
            self.ffn = point_wise_feed_forward_network(d_model, dff)

        # initializing smc parameters for training
        self.num_particles = 1
        self.noise = False
        self.len_resampling = None

        # output layer for computing the weights
        self.output_layer = tf.keras.layers.Dense(output_size, use_bias=False, name='output_layer')

        # internal states: K,V,R. size without batch_dim.
        self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                      V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                      R=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]))
        self.output_size = (tf.TensorShape([self.num_particles, 1, self.d_model]),
                            tf.TensorShape([self.num_particles, 1, self.seq_len]))  # r, attention_weights
        self.look_ahead_mask = self.create_look_ahead_mask()

        super(SMC_Transf_Cell, self).__init__(**kwargs)

    def create_look_ahead_mask(self):
        mask = create_look_ahead_mask(self.seq_len)
        mask = mask[:self.sampl_freq]
        return mask

    def add_SMC_parameters(self, dict_sigmas, num_particles):
        self.noise = True
        self.attention_smc.add_SMC_parameters(dict_sigmas=dict_sigmas)
        self.num_particles = num_particles
        #self.list_weights, self.list_indices = [], []

    def compute_w_classification(self, predictions, y):
      # right now, the predictions corresponds to the logits. Adding a softmax layer to have the normalized log probas:
      probas = tf.nn.softmax(predictions, axis=-1)  # shape (B,P,F,V)
      w = tf.gather(probas, y, axis=-1, batch_dims=3) # shape (B,P,S)
      w_norm = tf.nn.softmax(tf.squeeze(w, axis=-1), axis=1) # shape (B,S) # how to combine the three weights ?
      w_norm = tf.reduce_prod(w_norm, axis=-1) #TODO: check reduce_logsumexp
      return w_norm  # shape (B,P)

    def call_inference(self, inputs, states, timestep):
        K, V = states
        # self attention:
        (z, K, V), attn_weights = self.attention_smc(inputs=inputs, timestep=timestep, K=K, V=V)

        if self.full_model:
            out = self.layernorm1(z + inputs)
            r = self.ffn(out)
            r = self.layernorm2(r + out)
        else:
            r = z
        predictions = self.output_layer(r)  # (B,P,1,F_y)

        return predictions, (K, V)

    def add_stop_resampling(self, len_resampling):
        assert self.noise
        self.len_resampling = len_resampling

    def call(self, inputs, states):
        '''
        :param inputs:
        :param states:
        :return:
        '''
        x, y = tf.nest.flatten(inputs)  # unnesting inputs x: shape (B,P,resample_freq, D), y = shape(B,P, resampl_freq, D) with P=1 during training.
        #x, y = tf.expand_dims(x, axis=-2), tf.expand_dims(y, axis=-2)  # adding sequence dim. #TODO: not needed here because already a sequence dim.
        K, V, R = states  # getting states

        # self attention:
        timestep = self.sampl_freq * self.dec_timestep # if seq_len = 12, and sampl_freq = 3: values = [0,0,3,6,9,12] (duplication of zeros due to tensorflow
        timestep_mask = self.sampl_freq * (self.dec_timestep+1) # for same use-case, values = [3,3,6,9,12]
        #look_ahead_mask = self.create_look_ahead_mask(timestep=timestep_mask) # Mask for masking the future in the self attention computation.
        (z, K, V), attn_weights = self.attention_smc(inputs=x, timestep=timestep, dec_timestep=self.dec_timestep, K=K, V=V, mask=self.look_ahead_mask)  #TODO: compute decoding timestep as sample_freq * self.dec_timestep

        if self.full_model:
            out = self.layernorm1(z + x)
            r = self.ffn(out)
            r = self.layernorm2(r + out)
        else:
            r = z # (B,P,F,D)

        predictions = self.output_layer(r)  # (B,P,F,V)
        # storing r in R:
        R = self.attention_smc.update_state(new_state=r, states=R, timestep=timestep, dec_timestep=self.dec_timestep)

        # -------- SMC Algo ---------------------------------------------------------------------------------------------------------
        if self.noise:
            w = self.compute_w_classification(predictions=predictions, y=y)
            i_t = tf.random.categorical(w, self.num_particles)  # (B,P,1)
            w, i_t = tf.stop_gradient(w), tf.stop_gradient(i_t)
            #self.list_weights.append(w.numpy())
            #self.list_indices.append(i_t.numpy())
            # resample K, V, and R
            if self.len_resampling is None or self.dec_timestep < self.len_resampling:
                KVR = tf.concat([K,V,R], axis=-1)
                KVR = resample(KVR, i_t)
                K, V, R = tf.split(KVR, num_or_size_splits=3, axis=-1)
            # Getting internal noises for computing the loss.
            internal_noises = [self.attention_smc.noise_q, self.attention_smc.noise_z]
            output = [r, attn_weights, internal_noises]  # attn_weights > shape (B,P,1,S). noises: (B,P,1,D).
        else:
            output = [r, attn_weights]

        new_states = NestedState(K=K, V=V, R=R)
        self.cell_count += 1
        if self.cell_count > 1:
            self.dec_timestep += 1

        return output, new_states


if __name__ == "__main__":
    batch_size = 8
    d_model = 12
    output_size = 1
    seq_len = 4

    num_particles = 20
    sigma = 0.1
    sigma_obs = 0.5
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigma for _ in range(4)]))

    print(".........................................Test of compute w regression ................................")
    temp_cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=False, dff=0)

    temp_cell.add_SMC_parameters(dict_sigmas=dict_sigmas, num_particles=num_particles)

    temp_pred = tf.random.uniform(shape=(batch_size, 10, 1, output_size))
    temp_y = tf.random.uniform(shape=(batch_size, 10, 1, output_size))

    temp_w = temp_cell.compute_w_regression(predictions=temp_pred, y=temp_y)
    print('w', temp_w.shape)

    print(".........................................Test of multi-head attention  ................................")
    temp_cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=True, dff=24, num_heads=3)
    temp_cell.add_SMC_parameters(dict_sigmas=dict_sigmas, num_particles=num_particles)



