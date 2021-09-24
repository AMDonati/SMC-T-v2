import tensorflow as tf
import collections
# additional imports
from src.models.SMC_Transformer.self_attention_SMC import Self_Attention_SMC
from src.models.SMC_Transformer.multi_head_attention_SMC import mha_SMC
from src.models.SMC_Transformer.transformer_utils import resample
from src.models.classic_layers import point_wise_feed_forward_network

NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])


class SMC_Transf_Cell(tf.keras.layers.Layer):
    def __init__(self, d_model, output_size, seq_len, full_model, dff, num_heads=1, attn_window=None, task="classification", **kwargs):
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

        self.task = task

        if self.full_model:
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm1')
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm2')
            self.ffn = point_wise_feed_forward_network(d_model, dff)

        # initializing smc parameters for training
        self.num_particles = 1
        self.noise = False
        self.len_resampling = None

        # output layer for computing the weights
        if self.task == "regression":
            self.output_layer = tf.keras.layers.Dense(output_size, name='output_layer')
        elif self.task == "classification":
            self.output_layer = tf.keras.layers.Dense(output_size, use_bias=False, name='output_layer')

        # internal states: K,V,R. size without batch_dim.
        self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                      V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                      R=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]))
        self.output_size = (tf.TensorShape([self.num_particles, 1, self.d_model]),
                            tf.TensorShape([self.num_particles, 1, self.seq_len]))  # r, attention_weights

        super(SMC_Transf_Cell, self).__init__(**kwargs)

    def add_SMC_parameters(self, dict_sigmas, sigma_obs, num_particles):
        self.noise = True
        self.attention_smc.add_SMC_parameters(dict_sigmas=dict_sigmas)
        self.num_particles = num_particles
        self.Sigma_obs = sigma_obs
        self.list_weights, self.list_indices = [], []

    def compute_w_regression(self, predictions, y):
        '''
        # FORMULA
        # logw = -0.5 * mu_t ^ T * mu_t / omega
        # logw = logw - max(logw)
        # w = exp(logw)
        :param predictions: output of final layer: (B,P,1,F_y)
        :param y: current target element > shape (B,P,1,F_y).
        :return:
        resampling weights of shape (B,P).
        '''
        assert len(tf.shape(self.Sigma_obs)) == 0
        mu_t = y - predictions  # (B,P,1,F_y)
        mu_t = tf.squeeze(mu_t, axis=-2)  # removing sequence dim. # (B,P,F_y).
        log_w = (-1 / (2 * self.Sigma_obs)) * tf.matmul(mu_t, mu_t, transpose_b=True)  # (B,P,P)
        log_w = tf.linalg.diag_part(log_w)  # take the diagonal. # (B,P).
        w = tf.nn.softmax(log_w)
        # check if w contains a nan number
        bool_tens = tf.math.is_nan(w)
        has_nan = tf.math.reduce_any(bool_tens).numpy()
        assert has_nan == False
        assert len(tf.shape(w)) == 2
        return w

    def compute_w_classification(self, predictions, y):
      # right now, the predictions corresponds to the logits. Adding a softmax layer to have the normalized log probas:
      probas = tf.nn.softmax(predictions, axis=-1)  # shape (B,P,1,V)
      w = tf.gather(tf.squeeze(probas, axis=-2), tf.squeeze(y, axis=[-1,-2]), axis=-1, batch_dims=2) # shape (B,P)
      w_norm = tf.nn.softmax(w, axis=-1) # shape (B,P)
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
        x, y = tf.nest.flatten(inputs)  # unnesting inputs x: shape (B,P,D), y = shape(B,P,D) with P=1 during training.
        x, y = tf.expand_dims(x, axis=-2), tf.expand_dims(y, axis=-2)  # adding sequence dim.
        K, V, R = states  # getting states

        # self attention:
        (z, K, V), attn_weights = self.attention_smc(inputs=x, timestep=self.dec_timestep, K=K, V=V)

        if self.full_model:
            out = self.layernorm1(z + x)
            r = self.ffn(out)
            r = self.layernorm2(r + out)
        else:
            r = z

        predictions = self.output_layer(r)  # (B,P,1,F_y)
        # storing r in R:
        R_past = R[:, :, :self.dec_timestep, :]
        R_future = R[:, :, self.dec_timestep + 1:, :]
        R = tf.concat([R_past, r, R_future], axis=-2)

        # -------- SMC Algo ---------------------------------------------------------------------------------------------------------
        if self.noise:
            if self.task == "regression":
                w = self.compute_w_regression(predictions=predictions, y=y)
            elif self.task == "classification":
                w = self.compute_w_classification(predictions=predictions, y=y)
            i_t = tf.random.categorical(w, self.num_particles)  # (B,P,1)
            w, i_t = tf.stop_gradient(w), tf.stop_gradient(i_t)
            self.list_weights.append(w.numpy())
            self.list_indices.append(i_t.numpy())
            # resample K, V, and R
            if self.len_resampling is None or self.dec_timestep < self.len_resampling:
                K = resample(params=K, i_t=i_t, t=self.dec_timestep)
                V = resample(params=V, i_t=i_t, t=self.dec_timestep)
                R = resample(params=R, i_t=i_t, t=self.dec_timestep)
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

    temp_cell.add_SMC_parameters(dict_sigmas=dict_sigmas, sigma_obs=sigma_obs, num_particles=num_particles)

    temp_pred = tf.random.uniform(shape=(batch_size, 10, 1, output_size))
    temp_y = tf.random.uniform(shape=(batch_size, 10, 1, output_size))

    temp_w = temp_cell.compute_w_regression(predictions=temp_pred, y=temp_y)
    print('w', temp_w.shape)

    print(".........................................Test of multi-head attention  ................................")
    temp_cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=True, dff=24, num_heads=3)
    temp_cell.add_SMC_parameters(dict_sigmas=dict_sigmas, sigma_obs=sigma_obs, num_particles=num_particles)



