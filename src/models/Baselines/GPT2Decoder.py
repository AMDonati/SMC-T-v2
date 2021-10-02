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

    def call(self, input, attn_mask=None):
        if attn_mask is None:
            outputs = self.model(input_ids=input, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input, attention_mask=attn_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-2] # hidden before last attention block.
        return last_hidden_state

if __name__ == '__main__':
    gpt2decoder = GPT2Decoder()



#
# layer = keras.layers.Dense(3)
# layer.build((None, 4))  # Create the weights
# layer.trainable = False  # Freeze the layer