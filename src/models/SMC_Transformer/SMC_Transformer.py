# imports
import tensorflow as tf
from src.models.SMC_Transformer.SMC_TransformerCell import SMC_Transf_Cell
from src.models.Baselines.Transformer_without_enc import Decoder
from src.models.SMC_Transformer.transformer_utils import create_look_ahead_mask
import collections
from src.models.Baselines.GPT2Decoder import GPT2Decoder
import tensorflow_probability as tfp
from src.models.classic_layers import Linear
import math

# use this instead: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN?version=stable
NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])


# ------------------------CREATE THE SMC TRANSFORMER MODEL ------------------------------------------------------------

class SMC_Transformer(tf.keras.Model):

    def __init__(self, d_model, output_size, seq_len, full_model, dff, num_layers=1, num_heads=1, maximum_position_encoding=50,
                 rate=0., attn_window=None, init_weights=1):
        super(SMC_Transformer, self).__init__()

        # set Decoder
        #GPT2Decoder
        if num_layers == 0:
            self.decoder = GPT2Decoder()
            num_heads = 12
            d_model = 768
            dff = 3072
            self.layer_norm_final = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='layer_norm_final')
            _, self.init_variables = self.decoder.get_dict_variables()
            if init_weights == 0:
                self.init_variables = None

        if num_layers == 1:
            self.decoder = None if num_layers == 1 else Decoder(num_layers=num_layers - 1, d_model=d_model,
                                                                output_size=output_size, num_heads=num_heads,
                                                                dff=dff, full_model=full_model,
                                                                maximum_position_encoding=maximum_position_encoding,
                                                                rate=rate, dim=4)
            self.init_variables = None
        elif num_layers > 1:
            self.decoder = Decoder(num_layers=num_layers - 1, d_model=d_model, output_size=output_size,
                                   num_heads=num_heads,
                                   dff=dff, full_model=full_model,
                                   maximum_position_encoding=maximum_position_encoding,
                                   rate=rate, dim=4)
            self.init_variables = None

        self.cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=full_model,
                                    dff=dff, attn_window=attn_window, num_heads=num_heads, init_variables=self.init_variables, rate=rate)

        # for pre_processing words in the one_layer case.
        self.embedding = tf.keras.layers.Embedding(input_dim=output_size, output_dim=d_model, name="embedding") # for classification case.
        self.output_size = output_size
        self.seq_len = seq_len
        self.full_model = full_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff

        self.get_layers()

    def init_with_gpt2_params(self):
        if self.init_variables is not None:
            selected_variables = self.init_variables
            self.cell.layernorm1.gamma = selected_variables["ln_1/gamma:0"]
            self.cell.layernorm1.beta = selected_variables["ln_1/beta:0"]
            for weight in self.cell.layernorm1.weights:
                weight._trainable = True
            self.cell.layernorm2.gamma = selected_variables["ln_2/gamma:0"]
            self.cell.layernorm2.beta = selected_variables["ln_2/beta:0"]
            for weight in self.cell.layernorm2.weights:
                weight._trainable = True
            self.layer_norm_final.gamma = selected_variables['ln_f/gamma:0']
            self.layer_norm_final.beta = selected_variables['ln_f/beta:0']
            for weight in self.layer_norm_final.weights:
                weight._trainable = True
            print("initializing the SMC Transformer with GPT2 pretrained weights...")

    def get_layers(self):
        layers = self.cell.attention_smc.layers + self.cell.layers
        if self.num_layers == 0:
            layers.append(self.layer_norm_final)
        layers.append(self.cell.output_layer)
        self.layers_ = layers

    def compute_log_gaussian_density(self, logvar, noise):
        if len(tf.shape(logvar)) == 0:
            diag_logvar = 0.5 * logvar * tf.ones(shape=noise.shape[-1], dtype=tf.float32)
            diag_std = tf.math.exp(diag_logvar)
        else:
            diag_std = tf.linalg.diag_part(tf.exp(logvar * 0.5))
        gaussian_distrib = tfp.distributions.MultivariateNormalDiag(scale_diag=diag_std)
        log_prob = gaussian_distrib.log_prob(noise)
        return -log_prob

    def compute_log_gaussian_density_for_noise_net(self, logvar, noise): #TODO: check this function.
        bs = noise.shape[0]
        S = noise.shape[-2]
        logvar = tf.reduce_mean(logvar, axis=1) # same values along all particles.
        #log_probs = np.zeros(shape=(bs, self.cell.num_particles, S))
        loss_per_sample = []
        for b in range(bs):
            loss_per_timestep = []
            for s in range(S):
                logvar_ = logvar[b,s] # shape (d_model).
                noise_ = noise[b,:,s] # shape (P, d_model)
                gauss_ = tfp.distributions.MultivariateNormalDiag(scale_diag=tf.exp(logvar_*0.5))
                log_prob_ = gauss_.log_prob(noise_)
                loss_per_timestep.append(-log_prob_)
            loss_per_sample.append(tf.stack(loss_per_timestep, axis=-1))
        return tf.stack(loss_per_sample) # shape (B,P,S)

    def get_logvar_from_inputs(self, inputs):
        inputs = self.get_encoded_input(inputs)
        logvar_k, logvar_q, logvar_v, logvar_z = tf.split(self.cell.attention_smc.noise_network(inputs), num_or_size_splits=4, axis=-1)
        return [logvar_k, logvar_q, logvar_v, logvar_z]

    def compute_SMC_loss(self, inputs, targets, predictions, attention_mask=None): #TODO: add inputs here.
        assert self.cell.noise == self.cell.attention_smc.noise
        if self.cell.noise:
            if self.cell.attention_smc.noise_network is None:
                list_logvar = [self.cell.attention_smc.logvar_k, self.cell.attention_smc.logvar_q,
                           self.cell.attention_smc.logvar_v,
                           self.cell.attention_smc.logvar_z]  # (D,D) or scalar.
            else:
                list_logvar = self.get_logvar_from_inputs(inputs)
            loss_parts, loss_parts_no_log = [], []
            for noise, logvar in zip(self.internal_noises, list_logvar):
                if self.cell.attention_smc.noise_network is None:
                    loss_part = self.compute_log_gaussian_density(logvar=logvar, noise=noise)
                else:
                    loss_part = self.compute_log_gaussian_density_for_noise_net(logvar, noise)
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
        if attention_mask is None:
            # padding_mask = tf.math.logical_not(tf.math.equal(targets, 0))
            # padding_mask = tf.squeeze(tf.cast(padding_mask, dtype=classic_loss.dtype))  # shape (B,S)
            # classic_loss *= padding_mask
            # classic_loss = tf.reduce_sum(classic_loss) / tf.reduce_sum(padding_mask)
            classic_loss = tf.reduce_mean(classic_loss)
        else:
            attn_mask = tf.squeeze(tf.tile(attention_mask, multiples=[1, self.cell.num_particles, 1, 1]), axis=-1)
            attn_mask = tf.cast(attn_mask, dtype=classic_loss.dtype)
            classic_loss = classic_loss * attn_mask
            classic_loss = tf.reduce_sum(classic_loss) / tf.reduce_sum(attn_mask)
        #classic_loss = tf.reduce_mean(classic_loss)
        return classic_loss

    def get_encoded_input(self, inputs, attention_mask=None):
        if self.decoder is None:
            if tf.shape(inputs)[1] == 1:
                inputs = tf.tile(inputs, multiples=[1, self.cell.num_particles, 1, 1])
            input_tensor_processed = self.embedding(inputs)  # (B,P,S,D)
            input_tensor_processed = tf.squeeze(input_tensor_processed, axis=-2)
            input_tensor_processed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        else:
            if tf.shape(inputs)[1] == 1:
                inputs = tf.tile(inputs, multiples=[1, self.cell.num_particles, 1, 1])
            if attention_mask is not None and tf.shape(attention_mask)[1] == 1:
                attention_mask = tf.tile(attention_mask, multiples=[1, self.cell.num_particles, 1, 1])
            seq_len = tf.shape(inputs)[-2]
            look_ahead_mask = create_look_ahead_mask(seq_len)
            input_tensor_processed, _ = self.decoder(inputs, look_ahead_mask=look_ahead_mask, attention_mask=attention_mask) #TODO: bug here with GPT2output.
        return input_tensor_processed

    def call(self, inputs, targets, attention_mask=None):
        '''
        :param inputs: input_data: shape (B,P,S,F_x) with P=1 during training.
        :param targets: target_data: shape (B,P,S,F_y) with P=1 during training. F_y can be different from F_x.
        :return:
        '''
        # check dimensionality of inputs (B,P,S,F) with P = 1 during training.
        #assert len(tf.shape(inputs)) == len(tf.shape(targets)) == 4

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

        x = tf.transpose(input_tensor_processed,
                         perm=[0, 2, 1, 3])  # shape (B,S,P,D) so that it can be processed by the RNN_cell & RNN_layer.
        targets = tf.transpose(targets, perm=[0, 2, 1, 3])
        inputs_for_rnn = NestedInput(x=x, y=targets)  # y > (B,P,S,F_y), #x > (B,S,P,D))
        last_output, outputs, new_states = tf.keras.backend.rnn(step_function=step_function,
                                                                inputs=inputs_for_rnn,
                                                                initial_states=initial_state)
        self.cell.dec_timestep = 0  # reset decoding timestep of the cell to 0.
        self.cell.cell_count = 0  # additional counter to avoid duplicate of first timestep.

        # ------------------ EXTRACTING OUTPUTS OF THE RNN LAYER ------------------------------------------------------
        R = tf.transpose(tf.squeeze(outputs[0], axis=-2), perm=[0, 2, 1, 3])  # (B,P,S,D) # R not resampled.
        attn_weights = tf.squeeze(outputs[1], axis=-2)
        if len(tf.shape(attn_weights)) == 4: # one-head case
            attn_weights = tf.expand_dims(attn_weights, axis=-2) # (B,S,P,H,S)
        attn_weights = tf.transpose(attn_weights, perm=[0, 2, 3, 1, 4]) # (B,P,H,S,S)
        # states
        K, V, R_resampl = new_states[0], new_states[1], new_states[2]  # (B,P,S,D)

        # add a final layer norm for GPT2decoder.
        if self.num_layers == 0:
            R_resampl = self.layer_norm_final(R_resampl)
            R = self.layer_norm_final(R)

        pred_resampl = self.cell.output_layer(R_resampl)  # (B,P,S,C) used to compute the categorical cross_entropy loss.
        pred = self.cell.output_layer(R)

        # computing resampled noises for K, and V.
        self.noise_K_resampled = K - self.cell.attention_smc.wk(input_tensor_processed)
        self.noise_V_resampled = V - self.cell.attention_smc.wv(input_tensor_processed)

        if self.cell.noise:
            noise_q = outputs[-2][0]
            self.noise_q = tf.transpose(tf.squeeze(noise_q, axis=-2), perm=[0, 2, 1, 3])
            noise_z = outputs[-2][1]
            self.noise_z = tf.transpose(tf.squeeze(noise_z, axis=-2), perm=[0, 2, 1, 3])
            last_filtering_weights = tf.transpose(outputs[-1], perm=[0, 2, 1]) # B,P,S
            self.internal_noises = [self.noise_K_resampled, self.noise_q, self.noise_V_resampled, self.noise_z] #TODO: resampled also the other noises.
        else:
            last_filtering_weights = tf.ones(shape=(pred.shape[0], pred.shape[1], pred.shape[2]), dtype=tf.float32)

        return (pred, pred_resampl), (K, V, R_resampl), last_filtering_weights


if __name__ == "__main__":
    b = 8
    seq_len = 10
    F = 1
    d_model = 6
    full_model = True
    dff = 24
    num_particles = 4
    sigma = 0.1
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigma for _ in range(4)]))

    print("test NLP / classification case....")

    inputs = tf.constant([[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]], shape=(1, seq_len, F),
                         dtype=tf.int32)  # ok works with len(tf.shape(inputs)==3.
    inputs = tf.tile(inputs, multiples=[b, 1, 1])
    inputs = tf.expand_dims(inputs, axis=1)
    print('inputs', inputs.shape)

    targets = tf.constant([[[2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]], shape=(1, seq_len, F),
                          dtype=tf.int32)  # ok works with len(tf.shape(inputs)==3.
    targets = tf.tile(targets, multiples=[b, 1, 1])
    targets = tf.expand_dims(targets, axis=1)
    print('targets', targets.shape)

    print("..............................TEST ONE LAYER CASE ...........................................")

    transformer = SMC_Transformer(d_model=d_model, output_size=50, seq_len=seq_len, full_model=full_model, dff=dff,
                                  attn_window=4)

    transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                        num_particles=num_particles)

    (predictions, _), (K, V, R), attn_weights = transformer(inputs=inputs, targets=targets)

    print('predictions', predictions.shape)
    print('K', K.shape)
    print('attention weights', attn_weights.shape)

    print("....................test of computing SMC loss.....................................")
    smc_loss = transformer.compute_SMC_loss(targets=targets, predictions=predictions)


    print(".....................................TEST GPT2 DECODER .....................................................")
    inputs_ = tf.constant([[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], [[1], [2], [3], [4], [5], [6], [7], [20256], [20256], [20256]]], shape=(2, 1, seq_len, 1), dtype=tf.int32)
    targets_ = tf.constant([[[2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
                          [[2], [3], [4], [5], [6], [7], [8], [20256], [20256], [20256]]], shape=(2, 1, seq_len, 1), dtype=tf.int32)
    #targets_ = tf.expand_dims(targets_, axis=1)
    attention_mask = tf.constant([[1]*10, [1]*7+[0]*3], shape=(2,1,seq_len,1), dtype=tf.int32)

    transformer = SMC_Transformer(d_model=64, output_size=50257, seq_len=seq_len, full_model=full_model, dff=dff,
                                  attn_window=4, num_layers=0)

    gpt2decoder = transformer.decoder
    outputs_gpt2, (K_g, V_g), last_hidden_state = gpt2decoder.call_fullGPT2(inputs_, attention_mask)

    (predictions, _), (K, V, R), attn_weights = transformer(inputs=inputs_, targets=targets_,
                                                            attention_mask=attention_mask)
    transformer.init_with_gpt2_params()

    (predictions, _), (K, V, R), attn_weights = transformer(inputs=inputs_, targets=targets_,
                                                            attention_mask=attention_mask)

    Kg = tf.reshape(K_g, shape=(2,10,768))
    K = tf.squeeze(K)
    predictions = tf.squeeze(predictions)

    # cell output layer with GPT2 weights working.
    # idem for K,V first element.

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
