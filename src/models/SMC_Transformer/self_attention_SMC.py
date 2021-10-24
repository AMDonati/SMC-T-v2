import tensorflow as tf
import numpy as np
from src.models.SMC_Transformer.transformer_utils import create_look_ahead_mask, create_look_ahead_mask_per_block

# ----- scaled_dot_product_attention_function & mha function ------------

class Self_Attention_SMC(tf.keras.layers.Layer):

    def __init__(self, d_model, attn_window=None):
        super(Self_Attention_SMC, self).__init__()
        self.d_model = d_model
        self.attn_window = attn_window
        self.wq = tf.keras.layers.Dense(d_model, name='dense_projection_q')
        self.wk = tf.keras.layers.Dense(d_model, name='dense_projection_k')
        self.wv = tf.keras.layers.Dense(d_model, name='dense_projection_v')
        self.dense = tf.keras.layers.Dense(d_model, name='dense_projection_z')
        self.noise = False

    def add_SMC_parameters(self, dict_sigmas):
        # noise parameters.
        self.logvar_k = tf.Variable(initial_value=tf.math.log(dict_sigmas['k']), name="logvar_k")
        self.logvar_q = tf.Variable(initial_value=tf.math.log(dict_sigmas['q']), name="logvar_q")
        self.logvar_v = tf.Variable(initial_value=tf.math.log(dict_sigmas['v']), name="logvar_v")
        self.logvar_z = tf.Variable(initial_value=tf.math.log(dict_sigmas['z']), name="logvar_z")
        self.noise = True

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def add_noise(self, params, logvar):
        '''
        :param params: K,q,V or z. shape (B,P,S,D) for K, V. or shape (B,P,1,D) for q, z.
        :param sigma: scalar or matrix of shape (D,D).
        :return:
        '''
        assert len(tf.shape(logvar)) == 0
        logvar_ = tf.stop_gradient(logvar)
        gaussian_noise = tf.random.normal(shape=tf.shape(params), dtype=params.dtype)
        noise = tf.exp(logvar_ * 0.5) * gaussian_noise
        return params + noise

    def create_masks(self, inputs, K, timestep):
        mask_time = np.zeros(shape=K.shape)
        mask_time[:, :, timestep:timestep + inputs.shape[-2], :] = 1
        mask_time = tf.constant(mask_time, dtype=tf.float32) #mask for the current input block (time window sampl freq).
        return mask_time

    def update_state(self, new_state, states, timestep, dec_timestep):
        """

        Args:
            new_state: K(t:t+sampl_freq): shape (B,P,sampl_freq,D) (here sampl_freq=3)
            states: K(1:T): shape (B,P,S,D) (here S=12)
            timestep: current timestep for the whole sequence (seq_len = 12, sampl_freq=3) = > values = 0,3,6,12
            dec_timestep: recurrent timestep for the recurrent cell (only 4 recurrences in this case) => 0,1,2,3
        Returns:

        """
        paddings = tf.constant([[0, 0], [0, 0], [timestep, states.shape[-2] - new_state.shape[-2]*(dec_timestep+1)], [0, 0]])
        padded_new_state = tf.pad(new_state, paddings=paddings)
        states = states + padded_new_state
        return states

    def call(self, inputs, timestep, dec_timestep, K, V, mask=None):
        '''
        :param inputs: X_t (B,P,1,D) with P = 1 during training.
        :param timestep:
        :param K: (B,P,S,D) with P=1 during training.
        :param V: (B,P,S,D) with P=1 during training.
        :return:
        '''
        assert len(tf.shape(inputs)) == 4  # (B,P,1,D)

        # computing current k,q,v from inputs
        k_ = self.wk(inputs)  # (B,P,F,D)
        q_ = self.wq(inputs)  # (B,P,F,D)
        v_ = self.wv(inputs)  # (B,P,F,D)

        if self.noise:
            k = self.add_noise(k_, self.logvar_k)
            q = self.add_noise(q_, self.logvar_q)
            v = self.add_noise(v_, self.logvar_v)
            self.noise_k = k - k_
            self.noise_q = q - q_
            self.noise_v = v - v_
        else:
            k, q, v = k_, q_, v_

        K = self.update_state(new_state=k, states=K, timestep=timestep, dec_timestep=dec_timestep)
        V = self.update_state(new_state=v, states=V, timestep=timestep, dec_timestep=dec_timestep)

        # Computation of z from K,V,q.
        matmul_qk = tf.matmul(q, K, transpose_b=True)  # (B, P, 1, S)
        # scale matmul_qk
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # (B,P,1,S)
        if mask is not None: # mask per timestep on the future: shape (sample_freq, seq_len)
            scaled_attention_logits += (mask * -1e9)
        # softmax to get pi:
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B, P, 1, S)
        z_ = tf.matmul(attention_weights, V)  # (B,P,1,S)
        z_ = self.dense(z_)

        if self.noise:
            z = self.add_noise(z_, self.logvar_z)
            self.noise_z = z - z_ # TODO: remove this one, because we need resampled noise.
        else:
            z = z_
        return (z, K, V), attention_weights


if __name__ == "__main__":
    B = 8
    S = 20
    d_model = 512
    P = 10
    dec_timestep = 12
    x = tf.ones(shape=(B, P, 1, d_model))
    K = tf.random.uniform(shape=(B, P, S, d_model))
    V = tf.random.uniform(shape=(B, P, S, d_model))

    # temp_attention_logits = tf.random.uniform(shape=(B, P, 1, S))
    # scaled_attention_logits_masked = tf.concat([temp_attention_logits[:,:,:,:dec_timestep+1], -1e9 * tf.ones(shape=(B,P,1,S))], axis=-1)

    temp_attention = Self_Attention_SMC(d_model, attn_window=4)
    (temp_z, temp_K, temp_V), attn_weights = temp_attention(x, dec_timestep, K, V)
    print('temp_out', temp_z.shape)
    print('temp_K', temp_K.shape)
    print('temp_V', temp_V.shape)
    print('attention_weights', attn_weights.shape)

    # test of add noise function.
    temp_params = tf.random.uniform(shape=(B, 10, S, d_model), dtype=tf.float32)
    sigma = tf.square(tf.Variable(0.5, shape=()))
    new_params = temp_attention.add_noise(temp_params, sigma)
    print('new params', new_params.shape)

    # test with noise and more than one particule
    num_particles = 10
    sigma = 0.1
    x = tf.ones(shape=(B, num_particles, 1, d_model))
    K = tf.random.uniform(shape=(B, num_particles, S, d_model))
    V = tf.random.uniform(shape=(B, num_particles, S, d_model))
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigma for _ in range(4)]))
    temp_attention.add_SMC_parameters(dict_sigmas=dict_sigmas)
    (temp_z, temp_K, temp_V), attn_weights = temp_attention(x, dec_timestep, K, V)
    print('temp_out', temp_z.shape)
    print('temp_K', temp_K.shape)
    print('temp_V', temp_V.shape)
    print('attention_weights', attn_weights.shape)

    # with learned noise
    temp_attention.add_SMC_parameters(dict_sigmas=dict_sigmas)
    (temp_z, temp_K, temp_V), attn_weights = temp_attention(x, dec_timestep, K, V)
    print('temp_out', temp_z.shape)
    print('temp_K', temp_K.shape)
    print('temp_V', temp_V.shape)
    print('attention_weights', attn_weights.shape)

    # -------------------------------------------- code draft -----------------------------------------------------------------------
    # matriciel case for add noise;

    # def add_noise(self, params, sigma):
    #   '''
    #   :param params: K,q,V or z. shape (B,P,S,D) for K, V. or shape (B,P,1,D) for q, z.
    #   :param sigma: scalar or matrix of shape (D,D).
    #   :return:
    #   '''
    #   gaussian_noise = tf.random.normal(shape=tf.shape(params), dtype=params.dtype)
    #   if len(tf.shape(sigma)) == 0: # sigma is a scalar
    #     noise = tf.scalar_mul(sigma, gaussian_noise)
    #   else: # sigma is the std matrix of shape (B,B)
    #     noise = tf.einsum('bijk,kk->bijk', params, sigma)
    #   return params + noise
