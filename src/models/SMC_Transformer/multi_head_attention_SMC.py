import tensorflow as tf


# ----- scaled_dot_product_attention_function & mha function ------------

class mha_SMC(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, attn_window=None):
        super(mha_SMC, self).__init__()
        self.d_model = d_model
        self.attn_window = attn_window
        self.wq = tf.keras.layers.Dense(d_model, name='dense_projection_q')
        self.wk = tf.keras.layers.Dense(d_model, name='dense_projection_k')
        self.wv = tf.keras.layers.Dense(d_model, name='dense_projection_v')
        self.dense = tf.keras.layers.Dense(d_model, name='dense_projection_z')
        self.noise = False

        self.num_heads = num_heads
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

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
            k = self.add_noise(k_, self.sigma_k)
            q = self.add_noise(q_, self.sigma_q)
            v = self.add_noise(v_, self.sigma_v)
            self.noise_k = k - k_  # TODO: remove this one because we need resampled noise.
            self.noise_q = q - q_
            self.noise_v = v - v_  # TODO: remove this one, because we need resampled noise.
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
            z = self.add_noise(z_, self.sigma_z)
            self.noise_z = z - z_
        else:
            z = z_
        return (z, K, V), attention_weights


if __name__ == "__main__":
    B = 8
    S = 20
    d_model = 32
    num_heads = 2
    P = 10
    dec_timestep = 12
    x = tf.ones(shape=(B, P, 1, d_model))
    K = tf.random.uniform(shape=(B, P, S, d_model))
    V = tf.random.uniform(shape=(B, P, S, d_model))

    temp_attention = mha_SMC(d_model, num_heads=num_heads, attn_window=4)
    (temp_z, temp_K, temp_V), attn_weights = temp_attention(x, dec_timestep, K, V)
    print('temp_out', temp_z.shape)
    print('temp_K', temp_K.shape)
    print('temp_V', temp_V.shape)
    print('attention_weights', attn_weights.shape)

    print("................... TEST OF ADD NOISE FUNCTION................................")
    temp_params = tf.random.uniform(shape=(B, 10, S, d_model), dtype=tf.float32)
    sigma = tf.square(tf.Variable(0.5, shape=()))
    new_params = temp_attention.add_noise(temp_params, sigma)
    print('new params', new_params.shape)

    print("................... TEST OF ADD SMC PARAMETERS FUNCTION................................")
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
