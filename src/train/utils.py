import tensorflow as tf

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