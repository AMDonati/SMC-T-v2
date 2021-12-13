import tensorflow as tf

def compute_categorical_cross_entropy(targets, preds, num_particles, attention_mask=None):
    ce = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")
    preds = tf.nn.softmax(preds, axis=-1)
    # resampling_weights = smc_transformer.cell.list_weights[-1] #TODO: add comparison with best particle.
    # best_particles = tf.math.argmax(resampling_weights)
    ce_metric_avg_pred = ce(y_true=targets, y_pred=tf.reduce_mean(preds, axis=1, keepdims=True))  # (B,1,S)
    if attention_mask is not None:
        attn_mask = tf.squeeze(tf.tile(attention_mask, multiples=[1, num_particles, 1, 1]), axis=-1)
        attn_mask = tf.cast(attn_mask, dtype=tf.float32)
        ce_metric_avg_pred = ce_metric_avg_pred * attn_mask
    ce_metric_avg_pred = tf.reduce_mean(ce_metric_avg_pred)
    return ce_metric_avg_pred

def EM(smc_transformer, it, EM_param):
    print("updating variance of noise with an EM...")
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
            smc_transformer.cell.attention_smc.logvar_v = EM_update(err_v[j], smc_transformer.cell.attention_smc.logvar_v, it, EM_param)
            smc_transformer.cell.attention_smc.logvar_k = EM_update(err_k[j],
                                                                    smc_transformer.cell.attention_smc.logvar_k, it,
                                                                    EM_param)
            smc_transformer.cell.attention_smc.logvar_q = EM_update(err_q[j],
                                                                    smc_transformer.cell.attention_smc.logvar_q, it,
                                                                    EM_param)
            smc_transformer.cell.attention_smc.logvar_z = EM_update(err_z[j],
                                                                    smc_transformer.cell.attention_smc.logvar_z, it,
                                                                    EM_param)
    return smc_transformer

def EM_update(err, logvar, it, EM_param):
    var = (1 - it ** (
        -EM_param)) * tf.math.exp(logvar) + it ** (-EM_param) * err
    return tf.math.log(var)