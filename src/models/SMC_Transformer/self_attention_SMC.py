import tensorflow as tf
import numpy as np
from src.models.classic_layers import Linear

# ----- scaled_dot_product_attention_function & mha function ------------

class Self_Attention_SMC(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads=1, attn_window=None, init_variables=None):
        super(Self_Attention_SMC, self).__init__()
        self.d_model = d_model
        self.attn_window = attn_window

        #initialize layers
        if init_variables is None:
            self.wq = tf.keras.layers.Dense(d_model, name='dense_projection_q')
            self.wk = tf.keras.layers.Dense(d_model, name='dense_projection_k')
            self.wv = tf.keras.layers.Dense(d_model, name='dense_projection_v')
            self.dense = tf.keras.layers.Dense(d_model, name='dense_projection_z')
        else: # GPT2decoder:
            w_q, w_k, w_v = tf.split(init_variables['attn/c_attn/weight:0'], 3, axis=-1)
            b_q, b_k, b_v = tf.split(init_variables['attn/c_attn/bias:0'], 3, axis=-1)
            self.wq = Linear(name='dense_projection_q', units=d_model, kernel_init=w_q.numpy(), bias_init=b_q.numpy())
            self.wk = Linear(name='dense_projection_k', units=d_model, kernel_init=w_k.numpy(), bias_init=b_k.numpy())
            self.wv = Linear(name='dense_projection_v', units=d_model, kernel_init=w_v.numpy(), bias_init=b_v.numpy())
            w_z, b_z = init_variables['attn/c_proj/weight:0'], init_variables['attn/c_proj/bias:0']
            self.dense = Linear(name='dense_projection_v', units=d_model, kernel_init=w_z.numpy(), bias_init=b_z.numpy())

        self.layers = [self.wq, self.wk, self.wv, self.dense]

        self.noise = False
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

    def diagonal_variance_matrix(self, sigmas):
        return tf.linalg.diag(
            tf.math.log(tf.constant(sigmas, dtype=tf.float32))
        )

    def add_SMC_parameters(self, dict_sigmas, EM=False):
        # noise parameters.
        if EM:
            self.logvar_k = tf.math.log(dict_sigmas['k'])
            self.logvar_q = tf.math.log(dict_sigmas['q'])
            self.logvar_v = tf.math.log(dict_sigmas['v'])
            self.logvar_z = tf.math.log(dict_sigmas['z'])
        else:
            if isinstance(dict_sigmas['k'], list):
                self.logvar_k = tf.Variable(initial_value=self.diagonal_variance_matrix(dict_sigmas['k']), name="logvar_k")
                self.logvar_q = tf.Variable(initial_value=self.diagonal_variance_matrix(dict_sigmas['q']), name="logvar_q")
                self.logvar_v = tf.Variable(initial_value=self.diagonal_variance_matrix(dict_sigmas['v']), name="logvar_v")
                self.logvar_z = tf.Variable(initial_value=self.diagonal_variance_matrix(dict_sigmas['z']), name="logvar_z")
            else:
                self.logvar_k = tf.Variable(initial_value=tf.math.log(dict_sigmas['k']), name="logvar_k")
                self.logvar_q = tf.Variable(initial_value=tf.math.log(dict_sigmas['q']), name="logvar_q")
                self.logvar_v = tf.Variable(initial_value=tf.math.log(dict_sigmas['v']), name="logvar_v")
                self.logvar_z = tf.Variable(initial_value=tf.math.log(dict_sigmas['z']), name="logvar_z")
        self.noise = True

    # def reparameterize(self, mean, logvar):
    #     eps = tf.random.normal(shape=mean.shape)
    #     return eps * tf.exp(logvar * .5) + mean
    def concat_heads(self, x):
        '''concat attention parameters over all heads (and permute dimensions)
        -returns a tensor of shape (B, P, S, D)'''
        scaled_attention = tf.transpose(x, perm=[0, 1, 3, 2,
                                                 4])  # (batch_size, NUM_PARTICLES, seq_len_q, num_heads, depth)
        return tf.reshape(scaled_attention,
                          (tf.shape(scaled_attention)[0], tf.shape(scaled_attention)[1], -1,
                           self.d_model))  # (batch_size, NUM_PARTICLES, seq_len, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        (batch_size, num_particle, seq_length, d_model) => (batch_size, num_particle, seq_length, num_heads, depth=d_model/num_heads)
        """
        x = tf.reshape(x, (batch_size, tf.shape(x)[1], -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

    def add_noise(self, params, logvar):
        '''
        :param params: K,q,V or z. shape (B,P,S,D) for K, V. or shape (B,P,1,D) for q, z.
        :param sigma: scalar or matrix of shape (D,D).
        :return:
        '''
        gaussian_noise = tf.random.normal(shape=tf.shape(params), dtype=params.dtype)
        logvar_ = tf.stop_gradient(logvar)
        if len(tf.shape(logvar)) == 0:
            noise = tf.exp(logvar_ * 0.5) * gaussian_noise
        else:
            diag_std = tf.linalg.diag_part(tf.exp(logvar_ * 0.5))
            std_tiled = tf.reshape(diag_std, (1,1,1,diag_std.shape[0]))
            std_tiled = tf.tile(std_tiled, [gaussian_noise.shape[0], gaussian_noise.shape[1], gaussian_noise.shape[2], 1])
            noise = tf.math.multiply(gaussian_noise, std_tiled)
        return params + noise

    def self_attention_SMC(self, k, q, v, K, V, timestep):
        K_past = K[:, :, :, :timestep, :]
        K_future = K[:, :, :, timestep + 1:, :]
        K = tf.concat([K_past, k, K_future], axis=3)  # (B,P,H,S,D')
        V_past = V[:, :, :, :timestep, :]
        V_future = V[:, :, :, timestep + 1:, :]
        V = tf.concat([V_past, v, V_future], axis=3)  # (B,P,H,S,D')

        # Computation of z from K,V,q.
        matmul_qk = tf.matmul(q, K, transpose_b=True)  # (B, P, H, 1, S)
        # scale matmul_qk
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (B,P,H,1,S)
        bs, P, S = tf.shape(K)[0], tf.shape(K)[1], tf.shape(K)[-2]
        if self.attn_window is not None and timestep > self.attn_window:
            scaled_attention_logits_masked = tf.concat(
                [-1e9 * tf.ones(shape=(bs, P, self.num_heads, 1, timestep - self.attn_window)),
                 scaled_attention_logits[:, :, :, :,
                 timestep - self.attn_window:timestep + 1],
                 -1e9 * tf.ones(shape=(bs, P, self.num_heads, 1, S - (timestep + 1)))], axis=-1)
        else:
            scaled_attention_logits_masked = tf.concat([scaled_attention_logits[:, :, :, :, :timestep + 1],
                                                        -1e9 * tf.ones(shape=(
                                                            bs, P, self.num_heads, 1, S - (timestep + 1)))], axis=-1)
        # softmax to get pi:
        attention_weights = tf.nn.softmax(scaled_attention_logits_masked, axis=-1)  # (B, P, H, 1, S)
        z = tf.matmul(attention_weights, V)  # (B,P,H,1,D)

        return (z, K, V), attention_weights

    def call(self, inputs, timestep, K, V):
        '''
        :param inputs: X_t (B,P,1,D) with P = 1 during training.
        :param timestep:
        :param K: (B,P,S,D) with P=1 during training.
        :param V: (B,P,S,D) with P=1 during training.
        :return:
        '''
        assert len(tf.shape(inputs)) == 4  # (B,P,1,D)

        # computing current k,q,v from inputs
        k_ = self.wk(inputs)  # (B,P,1,D)
        q_ = self.wq(inputs)  # (B,P,1,D)
        v_ = self.wv(inputs)  # (B,P,1,D)

        if self.noise:
            k = self.add_noise(k_, self.logvar_k)
            q = self.add_noise(q_, self.logvar_q)
            v = self.add_noise(v_, self.logvar_v)
            self.noise_k = k - k_
            self.noise_q = q - q_
            self.noise_v = v - v_
        else:
            k, q, v = k_, q_, v_

        # Split k,q,v, K, V over heads:
        bs = tf.shape(inputs)[0]
        k = self.split_heads(k, bs)  # (B,P,H,1,D')
        q = self.split_heads(q, bs)  # (B,P,H,1,D')
        v = self.split_heads(v, bs)  # (B,P,H,1,D')
        if K is not None:
            K = self.split_heads(K, bs)  # (B,P,H,S,D')
        if V is not None:
            V = self.split_heads(V, bs)  # (B,P,H,S,D')

        # compute self attention for each head in parallel.
        (z_, K, V), attention_weights = self.self_attention_SMC(k=k, q=q, v=v, K=K, V=V, timestep=timestep)

        # concatenate over heads (z, K, V)
        z_ = self.concat_heads(z_) # (B,P,1,D)
        K = self.concat_heads(K) # (B,P,S,D)
        V = self.concat_heads(V) # (B,P,S,D)

        z_ = self.dense(z_)
        if self.noise:
            z = self.add_noise(z_, self.logvar_z)
            self.noise_z = z - z_
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
