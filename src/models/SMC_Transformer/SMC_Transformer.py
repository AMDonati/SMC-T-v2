# imports
import tensorflow as tf
from src.models.SMC_Transformer.SMC_TransformerCell import SMC_Transf_Cell
from src.models.Baselines.Transformer_without_enc import Decoder
from src.models.SMC_Transformer.transformer_utils import create_look_ahead_mask
import collections
from src.models.Baselines.GPT2Decoder import GPT2Decoder

# use this instead: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN?version=stable
NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])


# ------------------------CREATE THE SMC TRANSFORMER MODEL ------------------------------------------------------------

class SMC_Transformer(tf.keras.Model):

    def __init__(self, d_model, output_size, seq_len, full_model, dff, num_layers=1, num_heads=1, maximum_position_encoding=50,
                 rate=0., attn_window=None, reduce_gpt2output=False, sample_freq=1):
        super(SMC_Transformer, self).__init__()

        self.d_model = d_model

        # set Decoder
        if num_layers == 1:
            self.decoder = None if num_layers == 1 else Decoder(num_layers=num_layers - 1, d_model=d_model,
                                                                output_size=output_size, num_heads=num_heads,
                                                                dff=dff, full_model=full_model,
                                                                maximum_position_encoding=maximum_position_encoding,
                                                                rate=rate, dim=4)
        elif num_layers > 1:
            self.decoder = Decoder(num_layers=num_layers - 1, d_model=d_model, output_size=output_size,
                                   num_heads=num_heads,
                                   dff=dff, full_model=full_model,
                                   maximum_position_encoding=maximum_position_encoding,
                                   rate=rate, dim=4)
        elif num_layers == 0:
            self.decoder = GPT2Decoder()
            if reduce_gpt2output:
                self.gpt2_projection_layer = tf.keras.layers.Dense(d_model)
            else:
                self.gpt2_projection_layer = None
                self.d_model = 768

        self.cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=full_model,
                                    dff=dff, attn_window=attn_window, num_heads=num_heads, sampl_freq=sample_freq)

        # for pre_processing words in the one_layer case.
        self.embedding = tf.keras.layers.Embedding(input_dim=output_size, output_dim=d_model) # for classification case.
        self.final_layer = self.cell.output_layer
        self.output_size = output_size
        self.seq_len = seq_len
        self.full_model = full_model
        self.dff = dff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.sample_freq = sample_freq

    def compute_SMC_loss(self, targets, predictions, attention_mask=None):
        assert self.cell.noise == self.cell.attention_smc.noise
        if self.cell.noise:
            list_logvar = [self.cell.attention_smc.logvar_k, self.cell.attention_smc.logvar_q,
                           self.cell.attention_smc.logvar_v,
                           self.cell.attention_smc.logvar_z]  # (D,D) or scalar.
            loss_parts, loss_parts_no_log = [], []

            for noise, logvar in zip(self.internal_noises, list_logvar):
                loss_part = 1 / 2 * (1 / tf.exp(logvar)) * tf.einsum('bijk,bijk->bij', noise, noise)
                loss_parts.append(loss_part)
            smc_loss = tf.stack(loss_parts, axis=0)  # (4,B,P,S)
            smc_loss = tf.reduce_sum(smc_loss, axis=0)  # sum of loss parts. # (B,P,S)
            smc_loss = tf.reduce_mean(smc_loss)  # mean over all other dims.
        else:
            smc_loss = 0.

        classic_loss = self.compute_classic_loss(targets=targets, predictions=predictions, attention_mask=attention_mask)
        total_loss = smc_loss + classic_loss
        return total_loss, classic_loss

    def compute_classic_loss(self, targets, predictions, attention_mask=None):
        targets = tf.tile(targets, multiples=[1,self.cell.num_particles, 1, 1])
        ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        classic_loss = ce(y_true=targets, y_pred=predictions)
        if attention_mask is not None:
            attn_mask = tf.tile(tf.expand_dims(attention_mask, axis=1), multiples=[1, self.cell.num_particles, 1])
            attn_mask = tf.cast(attn_mask, dtype=tf.float32)
            classic_loss = classic_loss * attn_mask
        classic_loss = tf.reduce_mean(classic_loss)
        return classic_loss

    def get_encoded_input(self, inputs, attention_mask=None):
        if self.decoder is None:
            if tf.shape(inputs)[1] == 1:
                inputs = tf.tile(inputs, multiples=[1, self.cell.num_particles, 1, 1])
            input_tensor_processed = self.embedding(inputs)  # (B,P,S,D)
            input_tensor_processed = tf.squeeze(input_tensor_processed, axis=-2)
            input_tensor_processed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        elif self.decoder.__class__ == Decoder:
            if tf.shape(inputs)[1] == 1:
                inputs = tf.tile(inputs, multiples=[1, self.cell.num_particles, 1, 1])
            seq_len = tf.shape(inputs)[-2]
            look_ahead_mask = create_look_ahead_mask(seq_len)
            input_tensor_processed, _ = self.decoder(inputs, look_ahead_mask=look_ahead_mask)
        elif self.decoder.__class__ == GPT2Decoder:
            input_tensor_processed, _ = self.decoder(inputs, attention_mask=attention_mask)  # (B,S,D)
            input_tensor_processed = tf.expand_dims(input_tensor_processed, axis=1)
            input_tensor_processed = tf.tile(input_tensor_processed, multiples=[1, self.cell.num_particles, 1, 1])
            if self.gpt2_projection_layer is not None:
                input_tensor_processed = self.gpt2_projection_layer(input_tensor_processed)
        return input_tensor_processed

    def reinit_sample_freq(self, sample_freq):
        self.sample_freq = sample_freq
        self.cell.sampl_freq = sample_freq

    def reinit_seq_len(self, seq_len):
        self.seq_len = seq_len
        self.cell.seq_len = seq_len

    def prepare_inputs_for_rnn(self, tensor, permute=True):
        '''

        Args:
            tensor: shape (B,P,sample_freq, S/sample_freq, D)
        Returns:

        '''
        tensor = tf.reshape(tensor, shape=(
        tensor.shape[0], tensor.shape[1], self.sample_freq, int(tensor.shape[-2] / self.sample_freq), tensor.shape[-1]))
        if permute:
            tensor = tf.transpose(tensor, perm=[0, 3, 1, 2, 4])
        return tensor

    def reshape_ouputs(self, outputs):
        '''outputs = 5 dim.''' # shape (B,S', P, sample_freq, outputs.shape[-1])
        outputs = tf.transpose(outputs, perm=[0, 2, 1, 3, 4])  # shape (B,P, S' sample_freq, outputs.shape[-1])
        outputs = tf.reshape(outputs, shape=(outputs.shape[0], outputs.shape[1], outputs.shape[-2]*outputs.shape[2], outputs.shape[-1]))
        return outputs


    def call(self, inputs, targets, attention_mask=None):
        '''
        :param inputs: input_data: shape (B,P,S,F_x) with P=1 during training.
        :param targets: target_data: shape (B,P,S,F_y) with P=1 during training. F_y can be different from F_x.
        :return:
        '''

        input_tensor_processed = self.get_encoded_input(inputs, attention_mask=attention_mask)
        if tf.shape(targets)[1] == 1:
            targets = tf.tile(targets, multiples=[1, self.cell.num_particles, 1, 1])

        # 'dummy' initialization of cell's internal state for memory efficiency.
        shape = (tf.shape(input_tensor_processed)[0], tf.shape(input_tensor_processed)[1], self.seq_len,
                 self.d_model)  # S+1: trick because of dummy init.
        K0 = tf.zeros(shape=shape, dtype=tf.float32)
        initial_state = NestedState(K=K0,
                                    V=K0,
                                    R=K0)

        def step_function(inputs, states):
            return self.cell(inputs, states)

        x = self.prepare_inputs_for_rnn(input_tensor_processed, permute=True)
        y = self.prepare_inputs_for_rnn(targets, permute=True)
        inputs_for_rnn = NestedInput(x=x, y=y)  # y > (B,P,sample_freq,S,1), #x > (B,S',sample_freq, P,D))
        last_output, outputs, new_states = tf.keras.backend.rnn(step_function=step_function,
                                                                inputs=inputs_for_rnn,
                                                                initial_states=initial_state)
        self.cell.dec_timestep = 0  # reset decoding timestep of the cell to 0.
        self.cell.cell_count = 0  # additional counter to avoid duplicate of first timestep.

        # ------------------ EXTRACTING OUTPUTS OF THE RNN LAYER ------------------------------------------------------
        R = self.reshape_ouputs(outputs[0])
        attn_weights = self.reshape_ouputs(outputs[1])

        # states
        K, V, R_resampl = new_states[0], new_states[1], new_states[2]  # (B,P,S,D)

        pred_resampl = self.final_layer(R_resampl)  # (B,P,S,C) used to compute the categorical cross_entropy loss.
        pred = self.final_layer(R)

        # computing resampled noises for K, and V.
        self.noise_K_resampled = K - self.cell.attention_smc.wk(input_tensor_processed)
        self.noise_V_resampled = V - self.cell.attention_smc.wv(input_tensor_processed)

        if self.cell.noise:
            self.noise_q = self.reshape_ouputs(outputs[-1][0])
            self.noise_z = self.reshape_ouputs(outputs[-1][1])  # (B,P,S,D)
            self.internal_noises = [self.noise_K_resampled, self.noise_q, self.noise_V_resampled, self.noise_z] #TODO: resampled also the other noises.

        return (pred, pred_resampl), (K, V, R_resampl), attn_weights


if __name__ == "__main__":
    b = 8
    seq_len = 10
    F = 1
    d_model = 6
    full_model = True
    dff = 24
    num_particles = 2
    sigma = 0.1
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigma for _ in range(4)]))

    print("test NLP / classification case....")

    inputs = tf.constant([[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]], shape=(1, 12, F),
                         dtype=tf.int32)  # ok works with len(tf.shape(inputs)==3.
    inputs = tf.tile(inputs, multiples=[b, 1, 1])
    inputs = tf.expand_dims(inputs, axis=1)
    print('inputs', inputs.shape)

    targets = tf.constant([[[2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]]], shape=(1, 12, F),
                          dtype=tf.int32)  # ok works with len(tf.shape(inputs)==3.
    targets = tf.tile(targets, multiples=[b, 1, 1])
    targets = tf.expand_dims(targets, axis=1)
    print('targets', targets.shape)

    print("..............................TEST ONE LAYER CASE ...........................................")

    transformer = SMC_Transformer(d_model=d_model, output_size=50, seq_len=12, full_model=full_model, dff=dff,
                                  attn_window=4, sample_freq=3)

    transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                        num_particles=num_particles)

    (predictions, _), (K, V, R), attn_weights = transformer(inputs=inputs, targets=targets)

    print('predictions', predictions.shape)
    print('K', K.shape)
    print('attention weights', attn_weights.shape)

    print("....................test of computing SMC loss.....................................")
    smc_loss = transformer.compute_SMC_loss(targets=targets, predictions=predictions)

    print(".....................................TEST GPT2 DECODER .....................................................")
    inputs_ = tf.constant([[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], [[1], [2], [3], [4], [5], [6], [7], [20256], [20256], [20256]]], shape=(2, seq_len), dtype=tf.int32)
    targets_ = tf.constant([[[2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
                          [[2], [3], [4], [5], [6], [7], [8], [20256], [20256], [20256]]], shape=(2, seq_len, 1), dtype=tf.int32)
    targets_ = tf.expand_dims(targets_, axis=1)
    attention_mask = tf.constant([[1]*10, [1]*7+[0]*3], dtype=tf.int32)

    transformer = SMC_Transformer(d_model=64, output_size=20257, seq_len=seq_len, full_model=full_model, dff=dff,
                                  attn_window=4, num_layers=0, reduce_gpt2output=True)
    transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                        num_particles=num_particles)

    (predictions, _), (K, V, R), attn_weights = transformer(inputs=inputs_, targets=targets_, attention_mask=attention_mask)
    smc_loss = transformer.compute_SMC_loss(targets=targets_, predictions=predictions, attention_mask=attention_mask)

    # --------------------------------------------  TEST MULTI-LAYER CASE -------------------------------------

    print("..............................TEST MULTI-LAYER / ONE-HEAD CASE ...............................................")

    transformer = SMC_Transformer(d_model=d_model, output_size=50, seq_len=seq_len, full_model=full_model, dff=dff,
                                  num_layers=2, num_heads=1)
    print("Decoder num layers:", transformer.decoder.num_layers)

    print("...........................TESTING THE ADDITION OF THE SMC ALGORITHM ............................")
    num_particles = 20
    sigma = 0.1
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigma for _ in range(4)]))

    transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                        num_particles=num_particles)

    (predictions, _), (K, V, R), attn_weights = transformer(inputs=inputs, targets=targets)

    print('predictions', predictions.shape)
    print('K', K.shape)
    print('attention weights', attn_weights.shape)

    # ---------------------------------------------test when adding SMC during inference----------------------------------------------------------

    print('TESTING NOT RESAMPLING FOR INFERENCE....')
    transformer.cell.add_stop_resampling(5)
    (pred, pred_resampl), (K, V, R), attn_weights = transformer(inputs=inputs, targets=targets)

    # ------------------------------------------- test of compute_smc_loss -------------------------------------------------------------------------
    # test of tf.einsum:
    b = 1
    P = 1
    seq_len = 5
    d = 2
    temp_mu = tf.constant([[[[0.1, 1], [0.2, 2], [0.3, 3], [0.4, 4], [0.5, 5]]]], shape=(1, 1, seq_len, d))
    temp_mu_exp = tf.expand_dims(temp_mu, axis=-2)
    mult = tf.matmul(temp_mu_exp, temp_mu_exp, transpose_b=True)
    mult_2 = tf.einsum('bijk,bijk->bij', temp_mu, temp_mu)
    solution = tf.constant([1.01, 4.04, 9.09, 16.16, 25.25])

    # batch_size = 2
    temp_mu_2 = tf.concat([temp_mu, 2 * temp_mu], axis=0)
    mult2_2 = tf.einsum('bijk,bijk->bij', temp_mu_2, temp_mu_2)

    # p = 2
    temp_mu_3 = tf.concat([temp_mu, 2 * temp_mu], axis=1)
    mult3_2 = tf.einsum('bijk,bijk->bij', temp_mu_3, temp_mu_3)

    smc_loss = transformer.compute_SMC_loss(targets=targets, predictions=pred)
    print('smc loss', smc_loss.numpy())
