# imports
import tensorflow as tf
from src.models.SMC_Transformer.SMC_TransformerCell import SMC_Transf_Cell
import collections

# use this instead: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN?version=stable
NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])


# ------------------------CREATE THE SMC TRANSFORMER MODEL ------------------------------------------------------------

class SMC_Transformer(tf.keras.Model):

    def __init__(self, d_model, output_size, seq_len, full_model, dff, attn_window=None):
        super(SMC_Transformer, self).__init__()

        self.cell = SMC_Transf_Cell(d_model=d_model, output_size=output_size, seq_len=seq_len, full_model=full_model,
                                    dff=dff, attn_window=attn_window)

        # for pre_processing words in the one_layer case.
        self.input_dense_projection = tf.keras.layers.Dense(d_model, name='projection_layer_ts')  # for regression case.
        self.final_layer = self.cell.output_layer
        self.output_size = output_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.full_model = full_model
        self.dff = dff

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

        # "classic loss" part:
        diff = tf.cast(targets, tf.float32) - tf.cast(predictions, tf.float32)  # shape (B,P,S,F_y)
        classic_loss = 1 / 2 * (1 / self.cell.Sigma_obs) * tf.einsum('bijk,bijk->bij', diff, diff)
        classic_loss = tf.reduce_mean(classic_loss)

        total_loss = smc_loss + classic_loss
        return total_loss

    def call(self, inputs, targets):
        '''
        :param inputs: input_data: shape (B,P,S,F_x) with P=1 during training.
        :param targets: target_data: shape (B,P,S,F_y) with P=1 during training. F_y can be different from F_x.
        :return:
        '''
        # check dimensionality of inputs (B,P,S,F) with P = 1 during training.
        assert len(tf.shape(inputs)) == len(tf.shape(targets)) == 4
        seq_len = tf.shape(inputs)[-2]
        # assert self.seq_len == seq_len
        if tf.shape(inputs)[1] == 1:
            inputs = tf.tile(inputs, multiples=[1, self.cell.num_particles, 1,
                                                1])  # tiling inputs if needed on the particles dimensions.
            targets = tf.tile(targets, multiples=[1, self.cell.num_particles, 1, 1])

        input_tensor_processed = self.input_dense_projection(inputs)  # (B,P,S,D)
        input_tensor_processed *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

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
        outputs = [tf.squeeze(out, axis=-2) for out in outputs]
        R = tf.transpose(outputs[0], perm=[0, 2, 1, 3])  # (B,P,S,D) # R not resampled.
        attn_weights = tf.transpose(outputs[1], perm=[0, 2, 1, 3])
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
            self.internal_noises = [self.noise_K_resampled, self.noise_q, self.noise_V_resampled, self.noise_z]

        return (pred, pred_resampl), (K, V, R_resampl), attn_weights


if __name__ == "__main__":
    b = 8
    seq_len = 10
    F = 1
    d_model = 6
    full_model = True
    dff = 24

    inputs = tf.constant([[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]], shape=(1, seq_len, F),
                         dtype=tf.float32)  # ok works with len(tf.shape(inputs)==3.
    inputs = tf.tile(inputs, multiples=[b, 1, 1])
    inputs = tf.expand_dims(inputs, axis=1)
    print('inputs', inputs.shape)

    targets = tf.constant([[[2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]], shape=(1, seq_len, F),
                          dtype=tf.float32)  # ok works with len(tf.shape(inputs)==3.
    targets = tf.tile(targets, multiples=[b, 1, 1])
    targets = tf.expand_dims(targets, axis=1)
    print('targets', targets.shape)

    transformer = SMC_Transformer(d_model=d_model, output_size=1, seq_len=seq_len, full_model=full_model, dff=dff,
                                  attn_window=4)
    #(predictions, _), (K, V, R), attn_weights = transformer(inputs=inputs, targets=targets)

    # print('predictions', predictions.shape)
    # print('K', K.shape)
    # print('attention weights', attn_weights.shape)

    # ---------------------------------------------test when adding SMC during inference----------------------------------------------------------
    num_particles = 10
    sigma = 0.1
    sigma_obs = 0.5
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigma for _ in range(4)]))

    transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                        sigma_obs=sigma_obs,
                                        num_particles=num_particles)
    #(pred, pred_resampl), (K, V, R), attn_weights = transformer(inputs=inputs, targets=targets)
    # print('predictions', pred.shape)
    # print('predictions resampled', pred_resampl.shape)
    # print('K', K.shape)
    # print('attention weights', attn_weights.shape)

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

    # --------------------------------------------- code draft -------------------------------------------------------------------------------------
    # def compute_SMC_loss(self):
    #
    #   assert self.cell.noise == self.cell.attention_smc.noise == True
    #   list_noises = [self.internal_noises[i] for i in range(4)] # (B,P,S,D).
    #   list_sigmas = [self.cell.attention_smc.sigma_k, self.cell.attention_smc.sigma_q, self.cell.attention_smc.sigma_v, \
    #                                        self.cell.attention_smc.sigma_z] # (D,D) or scalar.
    #   loss_parts = []
    #   for noise, sigma in zip(list_noises, list_sigmas):
    #     if len(tf.shape(sigma)) == 0: # scalar case.
    #       temp = tf.scalar_mul((sigma_obs)**-2, noise)
    #     else:
    #       Sigma_inv = tf.matmul(tf.linalg.inv(sigma), tf.linalg.inv(sigma), transpose_b=True) # (D,D)
    #       temp = tf.einsum('bijk,kk->bijk', noise, Sigma_inv)
    #     loss_part = tf.einsum('bijk,bijk->bij', temp, noise)
    #     loss_parts.append(loss_part)
    #
    #   smc_loss = (1/2) * tf.stack(loss_parts, axis=0) # (4,B,P,S) # multiplication by 1/2 because the smc loss is (-log likelihood).
    #   smc_loss = tf.reduce_sum(smc_loss, axis=0) # sum of loss parts. # (B,P,S)
    #   smc_loss = tf.reduce_mean(smc_loss) #mean over all other dims.
    #
    #   return smc_loss
