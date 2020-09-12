import numpy as np
import tensorflow as tf
from src.utils.utils_train import write_to_csv

def split_input_target(data):
    inp = data[:, :, :-1, :]
    tar = data[:, :, 1:, :]
    return inp, tar

def inference_onestep(smc_transformer, test_sample, save_path, past_len=40):
    # forward pass on test_sample_past
    inp, tar = split_input_target(test_sample)
    smc_transformer.cell.add_stop_resampling(past_len)
    (preds, _), (K, V, _), _ = smc_transformer(inp, tar)  # K,V shape (1, P, 60, 1)
    preds = tf.squeeze(preds) # (P,60)
    mean_preds = tf.reduce_mean(preds, axis=0)  # (shape 60)
    preds_future = preds[:, past_len:]
    if save_path is not None:
        np.save(save_path, mean_preds)
    return preds_future, mean_preds

def inference_multistep(smc_transformer, test_sample, save_path=None, save_path_preds=None, past_len=40, future_len=20):
    P = smc_transformer.cell.num_particles
    d_model = smc_transformer.d_model
    sigma_obs = tf.math.sqrt(smc_transformer.cell.Sigma_obs)
    # forward pass on test_sample_past
    inp, tar = split_input_target(test_sample[:, :, :past_len + 1, :])
    smc_transformer.seq_len = past_len
    smc_transformer.cell.add_stop_resampling(past_len)
    for i in range(future_len+1):
        (preds, _), _, _ = smc_transformer(inp, tar)  # K,V shape (1, P, 40, D)
        last_pred = preds[:,:,-1,:]
        last_pred = last_pred + tf.random.normal(shape=last_pred.shape, stddev=sigma_obs)
        last_pred = tf.expand_dims(last_pred, axis=-2)
        if i == 0:
            inp = tf.tile(inp, multiples=[1,P,1,1])
            tar = tf.tile(tar, multiples=[1,P,1,1])
        inp = tf.concat([inp, last_pred], axis=-2)
        tar = tf.concat([tar, tf.zeros(shape=(tar.shape[0], tar.shape[1], 1, tar.shape[-1]))], axis=-2)
        smc_transformer.seq_len += 1

    mean_preds = tf.reduce_mean(preds, axis=1)
    mean_preds = tf.squeeze(mean_preds)
    preds_future = preds[:,:,past_len:,:]
    preds_future = tf.squeeze(preds_future)
    if save_path is not None:
        np.save(save_path, mean_preds)
    if save_path_preds is not None:
        np.save(save_path_preds, preds_future)
    return preds_future, mean_preds

def get_distrib_all_timesteps(preds, sigma_obs, P, save_path_distrib, len_future=20):
    distrib_future_timesteps = []
    for t in range(len_future):
        mean_NP = preds[:, t]
        emp_distrib = get_empirical_distrib(mean_NP, sigma_obs=sigma_obs, N_est=1000, P=P)
        distrib_future_timesteps.append(emp_distrib)
    distrib_future_timesteps = np.stack(distrib_future_timesteps, axis=0)
    print('distrib future timesteps', distrib_future_timesteps.shape)
    np.save(save_path_distrib, distrib_future_timesteps)


def get_empirical_distrib(mean_NP, sigma_obs, N_est, P):
    emp_distrib = np.zeros(shape=N_est)
    for i in range(N_est):
        ind_p = np.random.randint(0, P)
        sampled_mean = mean_NP[ind_p]
        sample = sampled_mean + tf.random.normal(shape=sampled_mean.shape, stddev=sigma_obs)
        emp_distrib[i] = sample.numpy()
    return emp_distrib


def EM_after_training(smc_transformer, inputs, targets, save_path, iterations=30):

    targets_tiled = tf.tile(targets, multiples=[1, smc_transformer.cell.num_particles, 1, 1])

    for it in range(1, iterations+1):
        (preds, preds_resampl), _, _ = smc_transformer(inputs=inputs,
                                                       targets=targets)
        # EM estimation of the noise parameters
        err_k = smc_transformer.noise_K_resampled * smc_transformer.noise_K_resampled
        err_k = tf.reduce_mean(err_k)
        err_q = smc_transformer.noise_q * smc_transformer.noise_q
        err_q = tf.reduce_mean(err_q)
        err_v = smc_transformer.noise_V_resampled * smc_transformer.noise_V_resampled
        err_v = tf.reduce_mean(err_v)
        err_z = smc_transformer.noise_z * smc_transformer.noise_z
        err_z = tf.reduce_mean(err_z)
        # EM estimation of Sigma_obs:
        err_obs = tf.cast(targets_tiled, tf.float32) - tf.cast(preds_resampl, tf.float32)
        new_sigma_obs = err_obs * err_obs
        new_sigma_obs = tf.reduce_mean(new_sigma_obs)

        smc_transformer.cell.attention_smc.sigma_v = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_v + it ** (
            -0.6) * err_v
        smc_transformer.cell.attention_smc.sigma_k = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_k + it ** (
            -0.6) * err_k
        smc_transformer.cell.attention_smc.sigma_q = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_q + it ** (
            -0.6) * err_q
        smc_transformer.cell.attention_smc.sigma_z = (1 - it ** (-0.6)) * smc_transformer.cell.attention_smc.sigma_z + it ** (
            -0.6) * err_z
        smc_transformer.cell.Sigma_obs = (1 - it ** (-0.6)) * smc_transformer.cell.Sigma_obs + it ** (-0.6) * new_sigma_obs
        print('it:', it)
        print("sigma_obs: {}, sigma_k: {}, sigma_q: {}, sigma_v: {}, sigma_z: {}".format(
            smc_transformer.cell.Sigma_obs,
            smc_transformer.cell.attention_smc.sigma_k,
            smc_transformer.cell.attention_smc.sigma_q,
            smc_transformer.cell.attention_smc.sigma_v,
            smc_transformer.cell.attention_smc.sigma_z
        ))

    dict_sigmas = dict(zip(['sigma_obs', 'sigma_k', 'sigma_q', 'sigma_v', 'sigma_z'],
                           [smc_transformer.cell.Sigma_obs,
            smc_transformer.cell.attention_smc.sigma_k,
            smc_transformer.cell.attention_smc.sigma_q,
            smc_transformer.cell.attention_smc.sigma_v,
            smc_transformer.cell.attention_smc.sigma_z]))

    write_to_csv(output_dir=save_path, dic=dict_sigmas)

    return smc_transformer.cell.Sigma_obs, (smc_transformer.cell.attention_smc.sigma_k,
            smc_transformer.cell.attention_smc.sigma_q,
            smc_transformer.cell.attention_smc.sigma_v,
            smc_transformer.cell.attention_smc.sigma_z)