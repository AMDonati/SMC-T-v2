import tensorflow as tf
from src.models.SMC_Transformer.transformer_utils import create_look_ahead_mask

# -------------------------------- TRAIN STEP FUNCTIONS ---------------------------------------------------------------------
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


# @tf.function(input_signature=train_step_signature) #TODO: debug this problem
def train_step_classic_T(inputs, targets, transformer, optimizer):
    '''training step for the classic Transformer model'''
    seq_len = tf.shape(inputs)[-2]
    mask_transformer = create_look_ahead_mask(seq_len)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inputs=inputs, training=True, mask=mask_transformer)
        loss = tf.keras.losses.MSE(targets, predictions)
        loss = tf.reduce_mean(loss)  # averaging loss over the seq and batch dims.
        gradients = tape.gradient(loss, transformer.trainable_variables)

    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss


# --------------SMC Transformer train_step-----------------------------------------------------------------------------------------------------
# @tf.function(input_signature=train_step_signature)
def train_step_SMC_T(inputs, targets, smc_transformer, optimizer, it, sigma_obs_update=0):
    '''
    :param it:
    :param inputs:
    :param targets:
    :param smc_transformer:
    :param optimizer:
    :return:
    '''

    assert len(tf.shape(inputs)) == len(tf.shape(targets)) == 4
    #sigma_obs_update = int(709/32) * 8

    with tf.GradientTape() as tape:
        (preds, preds_resampl), _, _ = smc_transformer(inputs=inputs,
                                                       targets=targets)  # predictions: shape (B,P,S,F_y) with P=1 during training.
        targets_tiled = tf.tile(targets, multiples=[1, smc_transformer.cell.num_particles, 1, 1])
        classic_loss = tf.keras.losses.MSE(targets_tiled, preds_resampl)  # (B,P,S)
        classic_loss = tf.reduce_mean(classic_loss)  # mean over all dimensions.


        if smc_transformer.cell.noise:
            # EM estimation of the noise parameters
            err_k = smc_transformer.noise_K_resampled * smc_transformer.noise_K_resampled
            err_k = tf.reduce_mean(err_k, axis=[1,2,3])
            err_q = smc_transformer.noise_q * smc_transformer.noise_q
            err_q = tf.reduce_mean(err_q, axis=[1,2,3])
            err_v = smc_transformer.noise_V_resampled * smc_transformer.noise_V_resampled
            err_v = tf.reduce_mean(err_v, axis=[1,2,3])
            err_z = smc_transformer.noise_z * smc_transformer.noise_z
            err_z = tf.reduce_mean(err_z, axis=[1,2,3])

            # smc_transformer.cell.attention_smc.sigma_v = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_v + it ** (-0.6) * err_v
            # smc_transformer.cell.attention_smc.sigma_k = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_k + it ** (-0.6) * err_k
            # smc_transformer.cell.attention_smc.sigma_q = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_q + it ** (-0.6) * err_q
            # smc_transformer.cell.attention_smc.sigma_z = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_z + it ** (-0.6) * err_z

            if it >= sigma_obs_update:
                it_obs = it - sigma_obs_update
                # EM estimation of Sigma_obs:
                err_obs = tf.cast(targets_tiled, tf.float32) - tf.cast(preds_resampl, tf.float32)
                new_sigma_obs = err_obs * err_obs
                new_sigma_obs = tf.reduce_mean(new_sigma_obs, axis=[1,2,3])

            for j in range(err_v.shape[0]):
                smc_transformer.cell.attention_smc.sigma_v = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_v + it ** (-0.6) * err_v[j]
                smc_transformer.cell.attention_smc.sigma_k = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_k + it ** (-0.6) * err_k[j]
                smc_transformer.cell.attention_smc.sigma_q = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_q + it ** (-0.6) * err_q[j]
                smc_transformer.cell.attention_smc.sigma_z = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_z + it ** (-0.6) * err_z[j]
                if it >= sigma_obs_update:
                    smc_transformer.cell.Sigma_obs = (1 - it_obs ** (-0.6)) * smc_transformer.cell.Sigma_obs + it_obs ** (-0.6) * new_sigma_obs[j]

            smc_loss = smc_transformer.compute_SMC_loss(predictions=preds_resampl, targets=targets_tiled)
            loss = smc_loss
            mse_metric_avg_pred = tf.keras.losses.MSE(targets, tf.reduce_mean(preds, axis=1, keepdims=True))  # (B,1,S)
            mse_metric_avg_pred = tf.reduce_mean(mse_metric_avg_pred)
        else:
            loss = classic_loss

        gradients = tape.gradient(loss, smc_transformer.trainable_variables)

        # To debug the loss.
        # trainable_variables = list(smc_transformer.trainable_variables)
        # trainable_variables_names = [t.name for t in trainable_variables]
        # var_and_grad_dict = dict(zip(trainable_variables_names, gradients))
        # print('dict of variables and associated gradients', var_and_grad_dict)

    optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))

    if smc_transformer.cell.noise:
        return loss, mse_metric_avg_pred
    else:
        return loss, None

