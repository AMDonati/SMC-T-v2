from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf
from tensorflow.keras.utils import plot_model


class GPT2Decoder(tf.keras.Model):
    def __init__(self, freeze_weights=True):
        super(GPT2Decoder, self).__init__()
        self.model = TFGPT2LMHeadModel.from_pretrained("cache/gpt2")
        if freeze_weights:
            self.model.trainable = False
            for weight in self.model.weights:
                weight._trainable = False
        self.tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def _reshape_inputs(self, input):
        if input is not None:
            input_ = tf.reshape(input, shape=(input.shape[0] * input.shape[1], input.shape[-2]))
        else:
            input_ = input
        return input_

    def call(self, input, attention_mask=None, look_ahead_mask=None):
        bs = input.shape[0]
        num_particles = input.shape[1]
        input_ = self._reshape_inputs(input)
        attention_mask_ = self._reshape_inputs(attention_mask)
        outputs = self.model(input_ids=input_, attention_mask=attention_mask_, output_hidden_states=True,
                             output_attentions=True)
        last_hidden_state = outputs.hidden_states[-2]  # shape (P*B,S,768) # hidden before last attention block.
        attention_weights = outputs.attentions
        last_hidden_state = tf.reshape(last_hidden_state, shape=(
        bs, num_particles, last_hidden_state.shape[-2], last_hidden_state.shape[-1]))
        #last_tf_block = self.model.transformer.h.layers[-1]
        #call() missing 5 required positional arguments: 'layer_past', 'attention_mask', 'head_mask', 'use_cache', and 'output_attentions'
        return last_hidden_state, attention_weights

    def call_fullGPT2(self, input, attention_mask=None):
        input_ = self._reshape_inputs(input)
        attention_mask_ = self._reshape_inputs(attention_mask)
        outputs = self.model(input_ids=input_, attention_mask=attention_mask_, output_hidden_states=True,
                             output_attentions=True)
        last_hidden_state = outputs.hidden_states[-1]
        logits = outputs.logits
        key_values = outputs.past_key_values[-1]
        (K,V) = tf.split(key_values, 2, axis=0)
        return logits, (K,V), last_hidden_state

    def get_dict_variables(self):
        variables = self.model.variables
        dict_variables = {v.name: v for v in variables}
        last_layer_variables_names = [name for name in list(dict_variables.keys()) if "h_._11/" in name]
        self.model_prefix = ('/').join(last_layer_variables_names[0].split('/')[:2])+'/'
        self.last_layer_prefix = ('/').join(last_layer_variables_names[0].split('/')[:3])+'/'
        embeddings_output_variables_names = [self.model_prefix+'wpe/embeddings:0',
                                             self.model_prefix+'wte/weight:0',
                                             self.model_prefix+'ln_f/gamma:0',
                                             self.model_prefix+'ln_f/beta:0']
        selected_var_names = last_layer_variables_names + embeddings_output_variables_names
        selected_variables = {k: v for k, v in dict_variables.items() if k in selected_var_names}
        renamed_variables = []
        for key in selected_variables.keys():
            if self.last_layer_prefix in key:
                renamed_variables.append(key.split(self.last_layer_prefix)[1])
            else:
                renamed_variables.append(key.split(self.model_prefix)[1])
        selected_variables = {k: v for k, v in zip(renamed_variables, list(selected_variables.values()))}
        for weight in selected_variables.values():
            weight._trainable = False
        return dict_variables, selected_variables

    def check_attention_parameters(self, input, attention_mask=None):
        input_ = self._reshape_inputs(input)
        attention_mask_ = self._reshape_inputs(attention_mask)
        outputs = self.model(input_ids=input_, attention_mask=attention_mask_, output_hidden_states=True,
                             output_attentions=True)
        first_key_values = outputs.past_key_values[0]  # shape (2,S,12,1,768/12)
        first_key_values = tf.reshape(first_key_values, shape=(
        2, first_key_values.shape[1], first_key_values.shape[2] * first_key_values.shape[-1]))
        embedding_output = outputs.hidden_states[0]  # (6,1,768) # output of embeddings.
        layer_norm = self.model.layers[0].ln_f
        gamma_beta_ln1 = self.variables[2:4]
        layer_norm.beta = gamma_beta_ln1[1]
        layer_norm.gamma = gamma_beta_ln1[0]
        input_attention = layer_norm(embedding_output)
        attn_weights = self.variables[4]  # 768 * 2304
        attn_bias = self.variables[5]  # (1,2034)
        queries_keys_values = tf.matmul(input_attention, attn_weights) + attn_bias
        Q, K, V = tf.split(tf.squeeze(queries_keys_values), 3, axis=-1)
        K_, V_ = tf.split(tf.squeeze(first_key_values), 2, axis=0)
        return (Q, K, V), (tf.squeeze(K_), tf.squeeze(V_))


if __name__ == '__main__':
    from src.models.classic_layers import Linear
    gpt2decoder = GPT2Decoder()
    # tf.keras.utils.plot_model(
    #     gpt2decoder.model.transformer.h, to_file='model.png', show_shapes=True,
    #     show_layer_names=True, expand_nested=True, dpi=96
    # )
    tf_block1 = gpt2decoder.model.transformer.h.layers[-1]
    inputs = tf.random.categorical(tf.math.log([[0.1, 0.1, 0.2, 0.3, 0.15, 0.05, 0.1]]), num_samples=6, dtype=tf.int32)
    print(inputs.numpy)
    all_variables, selected_variables = gpt2decoder.get_dict_variables()
    (Q, K, V), (K_, V_) = gpt2decoder.check_attention_parameters(inputs)
    hidden_state, attention_weights = gpt2decoder(inputs)

    # get GPT2 logits
    logits = gpt2decoder.call_fullGPT2(inputs)

    # check iniialization with GPT2weights.
    kernel_init = selected_variables['attn/c_proj/weight:0']
    bias_init = selected_variables['attn/c_proj/bias:0']
    layer = Linear(name="dense_1", units=768, kernel_init=kernel_init.numpy(), bias_init=bias_init.numpy())
    inputs_ = tf.tile(tf.expand_dims(inputs, axis=-1), multiples=[1,1,768])
    inputs_ = tf.cast(inputs_, tf.float32)
    outputs = layer(inputs_)
    print("done")

#
# layer = keras.layers.Dense(3)
# layer.build((None, 4))  # Create the weights
# layer.trainable = False  # Freeze the layer

# set_weights([weights,bias]) for a dense layer.

#LN_F: epsilon = 1e-5.
# idem for layer_norm1, layer_norm2.