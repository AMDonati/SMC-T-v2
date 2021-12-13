import tensorflow as tf
import collections
# additional imports
from src.models.SMC_Transformer.self_attention_SMC import Self_Attention_SMC
from src.models.SMC_Transformer.multi_head_attention_SMC import mha_SMC
from src.models.SMC_Transformer.transformer_utils import resample
from src.models.classic_layers import point_wise_feed_forward_network
from src.eval.language_metrics import gpt2_perplexity_batch, gpt2_perplexity_batch_2
from src.models.classic_layers import MLP, Linear

NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])


class SMC_Transf_Cell(tf.keras.layers.Layer):
    def __init__(self, d_model, output_size, seq_len, full_model, dff, num_heads=1, attn_window=None,
                 init_variables=None, rate=0., **kwargs):
        '''
        :param attn_window:
        :param full_model:
        :param dff:
        '''
        # store the decoding timestep
        self.dec_timestep = 0
        self.cell_count = 0
        self.attention_smc = Self_Attention_SMC(d_model=d_model, num_heads=num_heads, attn_window=attn_window, init_variables=init_variables)
        self.d_model = d_model
        self.output_size = output_size
        self.seq_len = seq_len
        self.full_model = full_model
        self.init_variables = init_variables

        # initialize layers:
        if self.init_variables is None:
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm1')
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm2')
            self.ffn = point_wise_feed_forward_network(d_model, dff)
            # output layer for computing the weights
            self.output_layer = tf.keras.layers.Dense(output_size, use_bias=False, name='output_layer')
        else:
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='layer_norm1')
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='layer_norm2')
            w_1, b_1 = init_variables['mlp/c_fc/weight:0'], init_variables['mlp/c_fc/bias:0']
            w_2, b_2 = init_variables['mlp/c_proj/weight:0'], init_variables['mlp/c_proj/bias:0']
            self.ffn = MLP(name='ffnn', units_1=dff, units_2=d_model, kernel_1_init=w_1.numpy(),
                           kernel_2_init=w_2.numpy(), bias_init_1=b_1.numpy(), bias_init_2=b_2.numpy())
            self.output_layer = Linear(name="output_layer", units=output_size, use_bias=False, kernel_init=init_variables['wte/weight:0'].numpy().T)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layers = [self.layernorm1, self.layernorm2, self.ffn]

        # initializing smc parameters for training
        self.num_particles = 1
        self.noise = False
        self.len_resampling = None
        self.training = False

        # internal states: K,V,R. size without batch_dim.
        self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                      V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                      R=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]))
        self.output_size = (tf.TensorShape([self.num_particles, 1, self.d_model]),
                            tf.TensorShape([self.num_particles, 1, self.seq_len]))  # r, attention_weights

        super(SMC_Transf_Cell, self).__init__(**kwargs)

    def init_inference_parameters(self, tokenizer):
        self.tokenizer=tokenizer

    def add_SMC_parameters(self, dict_sigmas, num_particles, EM=False):
        self.noise = True
        self.attention_smc.add_SMC_parameters(dict_sigmas=dict_sigmas, EM=EM)
        self.num_particles = num_particles
        # self.list_weights, self.list_indices = [], []

    def compute_w_classification(self, predictions, y):
        # right now, the predictions corresponds to the logits. Adding a softmax layer to have the normalized log probas:
        probas = tf.nn.softmax(predictions, axis=-1)  # shape (B,P,1,V)
        w = tf.gather(tf.squeeze(probas, axis=-2), tf.squeeze(y, axis=[-1, -2]), axis=-1, batch_dims=2)  # shape (B,P)
        # w_norm = tf.nn.softmax(w, axis=-1) # shape (B,P)
        w_norm = w
        return w_norm  # shape (B,P)

    def compute_w_inference(self, predictions, inputs):
        probs = tf.squeeze(tf.nn.softmax(predictions), axis=-2)
        values, indices = tf.math.top_k(probs, k=10)
        indices = tf.expand_dims(indices, axis=-1)  # shape (B,P,10,1)
        list_sentences = [tf.squeeze(tf.concat([inputs, tf.expand_dims(indices[:, :, i], axis=-2)], axis=-2)) for i in
                          range(indices.shape[-2])]  # list of tensor of shape (P,S).
        list_gpt2_ppl = [gpt2_perplexity_batch_2(sentence, tokenizer=self.tokenizer) for sentence in
                         list_sentences]
        # list_gpt2_ppl = [gpt2_perplexity_batch(sentence, tokenizer=self.tokenizer, reduction=False) for sentence in list_sentences]
        gpt2_ppls = tf.stack(list_gpt2_ppl, axis=-1)  # shape (B,10)
        weighted_gpt2_ppls = tf.math.multiply(gpt2_ppls, values)
        w = tf.reduce_mean(weighted_gpt2_ppls, axis=-1)
        return w  # shape P.

    def call_inference(self, inputs, encoded_inputs, states, timestep):
        K, V = states
        input = tf.expand_dims(encoded_inputs[:, :, -1, :],
                               axis=-2)  # caution, we need here the encoded input and not the raw one....
        # self attention:
        (z, K, V), attn_weights = self.attention_smc(inputs=input, timestep=timestep, K=K, V=V)

        if self.full_model:
            out = self.layernorm1(z + input)
            r = self.ffn(out)
            r = self.layernorm2(r + out)  # TODO: put this one before the attention
        else:
            r = z
        predictions = self.output_layer(r)  # (B,P,1,F_y)

        # resampling K,V:
        if self.noise:
            w = self.compute_w_inference(predictions=predictions, inputs=inputs)
            i_t = tf.random.categorical(w, self.num_particles)  # (B,P,1)
            w, i_t = tf.stop_gradient(w), tf.stop_gradient(i_t)
            KV = tf.concat([K, V], axis=-1)
            KV = resample(KV, i_t)
            K, V = tf.split(KV, num_or_size_splits=2, axis=-1)
        return tf.squeeze(predictions, axis=-2), (K, V)

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

        # for GPT2 - first layer norm before self-attention block
        x = self.layernorm1(x)

        # self attention:
        (z, K, V), attn_weights = self.attention_smc(inputs=x, timestep=self.dec_timestep, K=K, V=V)

        if self.full_model:
            self.dropout1(z, training=self.training)  # (B,S,D)
            out = self.layernorm1(z + x)
            r = self.ffn(out)
            r = self.dropout2(r, training=self.training)
            r = self.layernorm2(r + out)
        else:
            r = z

        predictions = self.output_layer(r)  # (B,P,1,F_y)

        if tf.reduce_sum(tf.cast(tf.math.is_nan(predictions), dtype=tf.int32)) > 0:
            print("BUG: NAN VALUES IN PREDICTIONS")
        # storing r in R:
        R_past = R[:, :, :self.dec_timestep, :]
        R_future = R[:, :, self.dec_timestep + 1:, :]
        R = tf.concat([R_past, r, R_future], axis=-2)

        # -------- SMC Algo ---------------------------------------------------------------------------------------------------------
        if self.noise:
            w = self.compute_w_classification(predictions=predictions, y=y)
            i_t = tf.random.categorical(w, self.num_particles)  # (B,P,1)
            w, i_t = tf.stop_gradient(w), tf.stop_gradient(i_t)
            # resample K, V, and R
            if self.len_resampling is None or self.dec_timestep < self.len_resampling:
                KVR = tf.concat([K, V, R], axis=-1)
                KVR = resample(KVR, i_t)
                K, V, R = tf.split(KVR, num_or_size_splits=3, axis=-1)
            # Getting internal noises for computing the loss.
            internal_noises = [self.attention_smc.noise_q, self.attention_smc.noise_z]
            output = [r, attn_weights, internal_noises, w]  # attn_weights > shape (B,P,1,S). noises: (B,P,1,D).
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
    temp_cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=True, dff=24,
                                num_heads=3)
    temp_cell.add_SMC_parameters(dict_sigmas=dict_sigmas, num_particles=num_particles)
