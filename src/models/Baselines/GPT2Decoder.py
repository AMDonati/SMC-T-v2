from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf

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
            input_ = tf.reshape(input, shape=(input.shape[0]*input.shape[1], input.shape[-2]))
        else:
            input_ = input
        return input_

    def call(self, input, attention_mask=None, look_ahead_mask=None):
        bs = input.shape[0]
        num_particles = input.shape[1]
        input_ = self._reshape_inputs(input)
        attention_mask_ = self._reshape_inputs(attention_mask)
        outputs = self.model(input_ids=input_, attention_mask=attention_mask_, output_hidden_states=True, output_attentions=True)
        last_hidden_state = outputs.hidden_states[-2]# shape (P*B,S,768) # hidden before last attention block.
        attention_weights = outputs.attentions
        last_hidden_state = tf.reshape(last_hidden_state, shape=(bs, num_particles, last_hidden_state.shape[-2], last_hidden_state.shape[-1]))
        return last_hidden_state, attention_weights

if __name__ == '__main__':
    gpt2decoder = GPT2Decoder()
    print("done")


#
# layer = keras.layers.Dense(3)
# layer.build((None, 4))  # Create the weights
# layer.trainable = False  # Freeze the layer