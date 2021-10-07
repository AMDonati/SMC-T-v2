import tensorflow as tf

def compute_categorical_cross_entropy(targets, preds, num_particles, attention_mask=None):
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    # resampling_weights = smc_transformer.cell.list_weights[-1] #TODO: add comparison with best particle.
    # best_particles = tf.math.argmax(resampling_weights)
    ce_metric_avg_pred = ce(y_true=targets, y_pred=tf.reduce_mean(preds, axis=1, keepdims=True))  # (B,1,S)
    if attention_mask is not None:
        attn_mask = tf.tile(tf.expand_dims(attention_mask, axis=1), multiples=[1, num_particles, 1])
        attn_mask = tf.cast(attn_mask, dtype=tf.float32)
        ce_metric_avg_pred = ce_metric_avg_pred * attn_mask
    ce_metric_avg_pred = tf.reduce_mean(ce_metric_avg_pred)
    return ce_metric_avg_pred

def EM(smc_transformer, it):
    if smc_transformer.cell.noise:
        # EM estimation of the noise parameters
        err_k = smc_transformer.noise_K_resampled * smc_transformer.noise_K_resampled
        err_k = tf.reduce_mean(err_k, axis=[1, 2, 3])
        err_q = smc_transformer.noise_q * smc_transformer.noise_q
        err_q = tf.reduce_mean(err_q, axis=[1, 2, 3])
        err_v = smc_transformer.noise_V_resampled * smc_transformer.noise_V_resampled
        err_v = tf.reduce_mean(err_v, axis=[1, 2, 3])
        err_z = smc_transformer.noise_z * smc_transformer.noise_z
        err_z = tf.reduce_mean(err_z, axis=[1, 2, 3])

        for j in range(err_v.shape[0]):
            smc_transformer.cell.attention_smc.sigma_v = (1 - it ** (
                -0.6)) * smc_transformer.cell.attention_smc.sigma_v + it ** (-0.6) * err_v[j]
            smc_transformer.cell.attention_smc.sigma_k = (1 - it ** (
                -0.6)) * smc_transformer.cell.attention_smc.sigma_k + it ** (-0.6) * err_k[j]
            smc_transformer.cell.attention_smc.sigma_q = (1 - it ** (
                -0.6)) * smc_transformer.cell.attention_smc.sigma_q + it ** (-0.6) * err_q[j]
            smc_transformer.cell.attention_smc.sigma_z = (1 - it ** (
                -0.6)) * smc_transformer.cell.attention_smc.sigma_z + it ** (-0.6) * err_z[j]
    return smc_transformer