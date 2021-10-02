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

    def call(self, input, attention_mask=None, look_ahead_mask=None):
        if attention_mask is None:
            outputs = self.model(input_ids=input, output_hidden_states=True, output_attentions=True)
        else:
            outputs = self.model(input_ids=input, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
        last_hidden_state = tf.squeeze(outputs.hidden_states[-2], axis=-2) # shape (B,P,S,768) # hidden before last attention block.
        attention_weights = outputs.attentions
        return last_hidden_state, attention_weights

if __name__ == '__main__':
    gpt2decoder = GPT2Decoder()



#
# layer = keras.layers.Dense(3)
# layer.build((None, 4))  # Create the weights
# layer.trainable = False  # Freeze the layer