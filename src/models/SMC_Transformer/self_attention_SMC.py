import tensorflow as tf


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
        self.sigma_k = dict_sigmas['k']
        self.sigma_q = dict_sigmas['q']
        self.sigma_v = dict_sigmas['v']
        self.sigma_z = dict_sigmas['z']
        self.noise = True

    def add_noise(self, params, sigma):
        '''
        :param params: K,q,V or z. shape (B,P,S,D) for K, V. or shape (B,P,1,D) for q, z.
        :param sigma: scalar or matrix of shape (D,D).
        :return:
        '''
        assert len(tf.shape(sigma)) == 0
        gaussian_noise = tf.random.normal(shape=tf.shape(params), dtype=params.dtype)
        noise = (sigma) ** (1 / 2) * gaussian_noise
        return params + noise

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
            k = self.add_noise(k_, self.sigma_k)
            q = self.add_noise(q_, self.sigma_q)
            v = self.add_noise(v_, self.sigma_v)
            self.noise_k = k - k_  # TODO: remove this one because we need resampled noise.
            self.noise_q = q - q_
            self.noise_v = v - v_  # TODO: remove this one, because we need resampled noise.
        else:
            k, q, v = k_, q_, v_

        K_past = K[:, :, :timestep, :]
        K_future = K[:, :, timestep + 1:, :]
        K = tf.concat([K_past, k, K_future], axis=2)  # (B,P,S,D)
        V_past = V[:, :, :timestep, :]
        V_future = V[:, :, timestep + 1:, :]
        V = tf.concat([V_past, v, V_future], axis=2)  # (B,P,S,D)

        # Computation of z from K,V,q.
        matmul_qk = tf.matmul(q, K, transpose_b=True)  # (B, P, 1, S)
        # scale matmul_qk
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (B,P,1,S)
        bs, P, S = tf.shape(K)[0], tf.shape(K)[1], tf.shape(K)[2]
        if self.attn_window is not None and timestep > self.attn_window:
            scaled_attention_logits_masked = tf.concat([-1e9 * tf.ones(shape=(bs, P, 1, timestep - self.attn_window)),
                                                        scaled_attention_logits[:, :, :, timestep - self.attn_window:timestep + 1],
                                                        -1e9 * tf.ones(shape=(bs, P, 1, S - (timestep + 1)))], axis=-1)
        else:
            scaled_attention_logits_masked = tf.concat([scaled_attention_logits[:, :, :, :timestep + 1],
                                                    -1e9 * tf.ones(shape=(
                                                    bs, P, 1, S - (timestep + 1)))], axis=-1)
        # softmax to get pi:
        attention_weights = tf.nn.softmax(scaled_attention_logits_masked, axis=-1)  # (B, P, 1, S)
        z_ = tf.matmul(attention_weights, V)  # (B,P,1,S)
        z_ = self.dense(z_)

        if self.noise:
            z = self.add_noise(z_, self.sigma_z)
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
