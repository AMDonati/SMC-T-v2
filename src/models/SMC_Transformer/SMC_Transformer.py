# imports
import tensorflow as tf
from src.models.SMC_Transformer.SMC_TransformerCell import SMC_Transf_Cell
from src.models.Baselines.Transformer_without_enc import Decoder
from src.models.SMC_Transformer.transformer_utils import create_look_ahead_mask
import collections

# use this instead: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN?version=stable
NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])


# ------------------------CREATE THE SMC TRANSFORMER MODEL ------------------------------------------------------------

class SMC_Transformer(tf.keras.Model):

    def __init__(self, d_model, output_size, seq_len, full_model, dff, num_layers=1, num_heads=1, maximum_position_encoding=50,
                 rate=0., attn_window=None, task="classification"):
        super(SMC_Transformer, self).__init__()

        self.cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=full_model,
                                    dff=dff, attn_window=attn_window, num_heads=num_heads, task=task)

        self.decoder = None if num_layers == 1 else Decoder(num_layers=num_layers - 1, d_model=d_model, num_heads=num_heads,
                                                            dff=dff, full_model=full_model,
                                                            maximum_position_encoding=maximum_position_encoding,
                                                            rate=rate, dim=4)
        self.task = task
        # for pre_processing words in the one_layer case.
        self.input_dense_projection = tf.keras.layers.Dense(d_model, name='projection_layer_ts') # for regression case.
        self.embedding = tf.keras.layers.Embedding(input_dim=output_size, output_dim=d_model) # for classification case.
        # for regression case.
        self.final_layer = self.cell.output_layer
        self.output_size = output_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.full_model = full_model
        self.dff = dff
        self.num_layers = num_layers
        self.num_heads = num_heads

    def compute_SMC_loss(self, targets, predictions):
        assert self.cell.noise == self.cell.attention_smc.noise == True
        list_Sigmas = [self.cell.attention_smc.sigma_k, self.cell.attention_smc.sigma_q,
                       self.cell.attention_smc.sigma_v, \
                       self.cell.attention_smc.sigma_z]  # (D,D) or scalar.
        loss_parts, loss_parts_no_log = [], []

        for noise, Sigma in zip(self.internal_noises, list_Sigmas):
            loss_part = 1 / 2 * (1 / Sigma) * tf.einsum('bijk,bijk->bij', noise, noise)
            loss_parts.append(loss_part)
        smc_loss = tf.stack(loss_parts, axis=0)  # (4,B,P,S)
        smc_loss = tf.reduce_sum(smc_loss, axis=0)  # sum of loss parts. # (B,P,S)
        smc_loss = tf.reduce_mean(smc_loss)  # mean over all other dims.

        classic_loss = self.compute_classic_loss(targets=targets, predictions=predictions)

        total_loss = smc_loss + classic_loss
        return total_loss, classic_loss

    def compute_classic_loss(self, targets, predictions):
        if self.task == "regression":
            # "classic loss" part:
            diff = tf.cast(targets, tf.float32) - tf.cast(predictions, tf.float32)  # shape (B,P,S,F_y)
            classic_loss = 1 / 2 * (1 / self.cell.Sigma_obs) * tf.einsum('bijk,bijk->bij', diff, diff)
            classic_loss = tf.reduce_mean(classic_loss)
        elif self.task == "classification":
            targets = tf.tile(targets, multiples=[1,self.cell.num_particles, 1, 1])
            ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
            classic_loss = ce(y_true=targets, y_pred=predictions)
            classic_loss = tf.reduce_mean(classic_loss)
        return classic_loss

    def get_encoded_input(self, inputs):
        if self.task == "regression":
            embedding_layer = self.input_dense_projection
        elif self.task == "classification":
            embedding_layer = self.embedding
        if tf.shape(inputs)[1] == 1:
            inputs = tf.tile(inputs, multiples=[1, self.cell.num_particles, 1, 1])
        if self.decoder is None:                                             # tiling inputs if needed on the particles dimensions.
            input_tensor_processed = embedding_layer(inputs)  # (B,P,S,D)
            input_tensor_processed = tf.squeeze(input_tensor_processed, axis=-2)
            input_tensor_processed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        else:
            seq_len = tf.shape(inputs)[-2]
            look_ahead_mask = create_look_ahead_mask(seq_len)
            input_tensor_processed, _ = self.decoder(inputs, training=False, look_ahead_mask=look_ahead_mask)  # (B,S,D)

        return input_tensor_processed

    def call(self, inputs, targets):
        '''
        :param inputs: input_data: shape (B,P,S,F_x) with P=1 during training.
        :param targets: target_data: shape (B,P,S,F_y) with P=1 during training. F_y can be different from F_x.
        :return:
        '''
        # check dimensionality of inputs (B,P,S,F) with P = 1 during training.
        assert len(tf.shape(inputs)) == len(tf.shape(targets)) == 4

        input_tensor_processed = self.get_encoded_input(inputs)
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
        outputs = [tf.squeeze(out, axis=-2) for out in outputs] # [R=logits before last layer, attention weights, (internal noises)]
        R = tf.transpose(outputs[0], perm=[0, 2, 1, 3])  # (B,P,S,D) # R not resampled.
        attn_weights = outputs[1]
        if len(tf.shape(attn_weights)) == 4: # one-head case
            attn_weights = tf.expand_dims(attn_weights, axis=-2) # (B,S,P,H,S)
        attn_weights = tf.transpose(attn_weights, perm=[0, 2, 3, 1, 4]) # (B,P,H,S,S)
        # states
        K, V, R_resampl = new_states[0], new_states[1], new_states[2]  # (B,P,S,D)

        pred_resampl = self.final_layer(R_resampl)  # (B,P,S,C) used to compute the categorical cross_entropy loss.
        pred = self.final_layer(R)

        # computing resampled noises for K, and V.
        self.noise_K_resampled = K - self.cell.attention_smc.wk(input_tensor_processed)
        self.noise_V_resampled = V - self.cell.attention_smc.wv(input_tensor_processed)

        if self.cell.noise:
            self.noise_q = tf.transpose(outputs[-1][0, :, :, :, :], perm=[0, 2, 1, 3])  # (B,P,S,D).
            self.noise_z = tf.transpose(outputs[-1][1, :, :, :, :], perm=[0, 2, 1, 3])  # (B,P,S,D)
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
    sigma_obs = 0.5
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
                                  attn_window=4, task="classification")

    transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                        sigma_obs=sigma_obs,
                                        num_particles=num_particles)

    (predictions, _), (K, V, R), attn_weights = transformer(inputs=inputs, targets=targets)

    print('predictions', predictions.shape)
    print('K', K.shape)
    print('attention weights', attn_weights.shape)

    print("....................test of computing SMC loss.....................................")
    smc_loss = transformer.compute_SMC_loss(targets=targets, predictions=predictions)


    # --------------------------------------------  TEST MULTI-LAYER CASE -------------------------------------

    print("..............................TEST MULTI-LAYER / ONE-HEAD CASE ...............................................")

    transformer = SMC_Transformer(d_model=d_model, output_size=1, seq_len=seq_len, full_model=full_model, dff=dff,
                                  num_layers=2, num_heads=1)
    print("Decoder num layers:", transformer.decoder.num_layers)

    print("...........................TESTING THE ADDITION OF THE SMC ALGORITHM ............................")
    num_particles = 20
    sigma = 0.1
    sigma_obs = 0.5
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigma for _ in range(4)]))

    transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                        sigma_obs=sigma_obs,
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
