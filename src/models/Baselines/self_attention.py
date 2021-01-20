import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """
    :param q:
    :param k:
    :param v:
    :param mask:
    :return:
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (B,P,S_q,S_k) or (B,S,S)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (B,P,S_q,S_k) or (B,S,S)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B,P,S_q, S_k) or (B,S,S)

    output = tf.matmul(attention_weights, v)  # (B,P,S,D) or (B,S,S)

    return output, attention_weights


class OneHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(OneHeadAttention, self).__init__()
        self.d_model = d_model

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        self.noise = False

    def call(self, inputs, mask):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]

        q = self.wq(q)  # (B, S, D)
        k = self.wk(k)  # (B, S, D)
        v = self.wv(v)  # (B, S, D)

        self.K, self.V = k, v

        scaled_attention, attention_weights = scaled_dot_product_attention(q=q, k=self.K, v=self.V,
                                                                           mask=mask)  # (B,P,S,D)
        z = self.dense(scaled_attention)  # (B,S,D)

        return z, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def concat_heads(self, x):
        '''concat attention parameters over all heads (and permute dimensions)
        -returns a tensor of shape (B, P, S, D)'''
        if len(x.shape) == 4:
            scaled_attention = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, S, H, D')
            scaled_attention = tf.reshape(scaled_attention,
                                          (tf.shape(scaled_attention)[0], -1, self.d_model))  # (B, S, D)
        elif len(x.shape) == 5:
            scaled_attention = tf.transpose(x, perm=[0, 1, 3, 2,
                                                     4])  # (B, P, S, H, D')
            scaled_attention = tf.reshape(scaled_attention,
                                          (tf.shape(scaled_attention)[0], tf.shape(scaled_attention)[1], -1,
                                           self.d_model))  # (B,P,S,D)
        return scaled_attention

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        if len(x.shape) == 3:
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            x = tf.transpose(x, perm=[0, 2, 1, 3])
        elif len(x.shape) == 4:
            x = tf.reshape(x, (batch_size, tf.shape(x)[1], -1, self.num_heads, self.depth))
            x = tf.transpose(x, perm=[0, 1, 3, 2, 4])
        return x

    def call(self, inputs, mask):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        concat_attention = self.concat_heads(scaled_attention)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
