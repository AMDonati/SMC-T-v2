import numpy as np
import tensorflow as tf
from utils.utils_train import write_to_csv

def split_input_target(data):
    inp = data[:, :, :-1, :]
    tar = data[:, :, 1:, :]
    return inp, tar


def expand_and_tile_attn_params(params, N, P, d_model):
    params = tf.expand_dims(params, axis=1)
    params = tf.tile(params, multiples=[1, N, 1, 1, 1])
    params = tf.reshape(params, shape=[1, N * P, 40, d_model])
    params_future = tf.zeros(shape=(1, N * P, 20, d_model))
    params = tf.concat([params, params_future], axis=-2)
    return params


def inference_onestep(smc_transformer, test_sample, N, save_path, past_len=40, future_len=20):
    P = smc_transformer.cell.num_particles
    d_model = smc_transformer.d_model

    # forward pass on test_sample_past
    inp, tar = split_input_target(test_sample[:, :, :past_len + 1, :])
    smc_transformer.seq_len = past_len
    (preds, _), (K, V, _), _ = smc_transformer(inp, tar)  # K,V shape (1, P, 40, D)
    new_K = expand_and_tile_attn_params(K, N=N, P=P, d_model=d_model)
    new_V = expand_and_tile_attn_params(V, N=N, P=P, d_model=d_model)

    # inference for the future
    smc_transformer.seq_len = past_len + future_len
    inf_inp, _ = split_input_target(test_sample[:, :, past_len:, :])
    inf_inp = tf.tile(inf_inp, multiples=[1, N * P, 1, 1])
    inf_inp = smc_transformer.input_dense_projection(inf_inp)  # pre_processing of inf_inp.
    preds_NP = []
    for i in range(future_len):
        x = tf.expand_dims(inf_inp[:, :, i, :], axis=-2)
        t = i + past_len
        pred_NP, (new_K, new_V) = smc_transformer.cell.call_inference(inputs=x, states=(new_K, new_V), timestep=t)
        preds_NP.append(pred_NP)
    preds_NP = tf.stack(preds_NP, axis=-2)
    preds_NP = tf.squeeze(preds_NP)

    mean_preds_future = tf.reduce_mean(preds_NP, axis=0) # (shape 40)
    preds = tf.squeeze(preds)
    mean_preds_past = tf.reduce_mean(preds, axis=0) # (shape 60)
    mean_preds = tf.concat([mean_preds_past, mean_preds_future], axis=0)
    mean_preds = mean_preds.numpy()
    np.save(save_path, mean_preds)

    return preds_NP, mean_preds

def inference_onestep_2(smc_transformer, test_sample, save_path, past_len=40):
    # forward pass on test_sample_past
    inp, tar = split_input_target(test_sample)
    (preds, _), (K, V, _), _ = smc_transformer(inp, tar)  # K,V shape (1, P, 60, 1)
    preds = tf.squeeze(preds) #(P,60)
    mean_preds = tf.reduce_mean(preds, axis=0)  # (shape 60)
    preds_future = preds[:, past_len:]
    if save_path is not None:
        np.save(save_path, mean_preds)
    return preds_future, mean_preds

def inference_onestep_3(smc_transformer, test_sample, save_path, past_len=40):
    # forward pass on test_sample_past
    inp, tar = split_input_target(test_sample)
    smc_transformer.cell.add_stop_resampling(past_len)
    (preds, _), (K, V, _), _ = smc_transformer(inp, tar)  # K,V shape (1, P, 60, 1)
    preds = tf.squeeze(preds) #(P,60)
    mean_preds = tf.reduce_mean(preds, axis=0)  # (shape 60)
    preds_future = preds[:, past_len:]
    if save_path is not None:
        np.save(save_path, mean_preds)
    return preds_future, mean_preds

def inference_multistep(smc_transformer, test_sample, N, save_path=None, past_len=40, future_len=20):
    P = smc_transformer.cell.num_particles
    d_model = smc_transformer.d_model
    sigma_obs = tf.math.sqrt(smc_transformer.cell.Sigma_obs)
    # forward pass on test_sample_past
    inp, tar = split_input_target(test_sample[:, :, :past_len + 1, :])
    smc_transformer.seq_len = past_len
    (preds, _), (K, V, _), _ = smc_transformer(inp, tar)  # K,V shape (1, P, 40, D)
    new_K = expand_and_tile_attn_params(K, N=N, P=P, d_model=d_model)
    new_V = expand_and_tile_attn_params(V, N=N, P=P, d_model=d_model)

    # inference for the future
    smc_transformer.seq_len = past_len + future_len
    inf_inp = tf.expand_dims(test_sample[:,:,past_len,:], axis=-2)
    inf_inp = tf.tile(inf_inp, multiples=[1, N * P, 1, 1])
      # pre_processing of inf_inp.
    means_NP = []
    pred_NP = inf_inp
    for i in range(future_len):
        t = i + past_len
        pred_NP = smc_transformer.input_dense_projection(pred_NP)
        mean_NP, (new_K, new_V) = smc_transformer.cell.call_inference(inputs=pred_NP, states=(new_K, new_V), timestep=t)
        pred_NP = mean_NP + tf.random.normal(shape=mean_NP.shape, stddev=sigma_obs)
        means_NP.append(mean_NP)
    means_NP = tf.stack(means_NP, axis=-2)
    means_NP = tf.squeeze(means_NP)

    mean_preds_future = tf.reduce_mean(means_NP, axis=0) # (shape 20)
    preds = tf.squeeze(preds)
    mean_preds_past = tf.reduce_mean(preds, axis=0) # (shape 60)
    mean_preds = tf.concat([mean_preds_past, mean_preds_future], axis=0)
    mean_preds = mean_preds.numpy()
    if save_path is not None:
        np.save(save_path, mean_preds)

    return means_NP, mean_preds

def inference_multistep_2(smc_transformer, test_sample, save_path=None, save_path_preds=None, past_len=40, future_len=20):
    P = smc_transformer.cell.num_particles
    d_model = smc_transformer.d_model
    sigma_obs = tf.math.sqrt(smc_transformer.cell.Sigma_obs)
    # forward pass on test_sample_past
    inp, tar = split_input_target(test_sample[:, :, :past_len + 1, :])
    smc_transformer.seq_len = past_len
    smc_transformer.cell.add_stop_resampling(past_len)
    for i in range(future_len+1):
        (preds, _), _, _ = smc_transformer(inp, tar)  # K,V shape (1, P, 40, D)
        last_pred = tf.reduce_mean(preds[:,:,-1,:], axis=1, keepdims=True)
        last_pred = last_pred + tf.random.normal(shape=last_pred.shape, stddev=sigma_obs)
        last_pred = tf.expand_dims(last_pred, axis=-2)
        inp = tf.concat([inp, last_pred], axis=-2)
        tar = tf.concat([tar, tf.zeros(shape=(tar.shape[0], tar.shape[1], 1, tar.shape[-1]))], axis=-2)
        smc_transformer.seq_len += 1

    mean_preds = tf.reduce_mean(preds, axis=1)
    mean_preds = tf.squeeze(mean_preds)
    preds_future = preds[:,:,40:,:]
    preds_future = tf.squeeze(preds_future)
    if save_path is not None:
        np.save(save_path, mean_preds)
    if save_path is not None:
        np.save(save_path_preds, preds_future)

    return preds_future, mean_preds

def inference_multistep_3(smc_transformer, test_sample, save_path=None, save_path_preds=None, past_len=40, future_len=20):
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
    preds_future = preds[:,:,40:,:]
    preds_future = tf.squeeze(preds_future)
    if save_path is not None:
        np.save(save_path, mean_preds)
    if save_path_preds is not None:
        np.save(save_path_preds, preds_future)

    return preds_future, mean_preds


def inference_multistep_4(smc_transformer, test_sample, N=10, save_path=None, save_path_preds=None, past_len=40, future_len=20):
    P = smc_transformer.cell.num_particles
    d_model = smc_transformer.d_model
    sigma_obs = tf.math.sqrt(smc_transformer.cell.Sigma_obs)
    # forward pass on test_sample_past
    inp, tar = split_input_target(test_sample[:, :, :past_len + 1, :])
    inp = tf.tile(inp, multiples=[N,1,1,1])
    tar = tf.tile(tar, multiples=[N,1,1,1])
    smc_transformer.seq_len = past_len
    smc_transformer.cell.add_stop_resampling(past_len)
    for i in range(future_len+1):
        (preds, _), _, _ = smc_transformer(inp, tar)  # K,V shape (N, P, 40, D)
        inp_p = np.random.randint(0, P, size=N)
        last_pred = tf.gather(preds[:, :, -1, :], inp_p, axis=1)
        last_pred = tf.linalg.diag_part(last_pred)
        last_pred = tf.expand_dims(last_pred, axis=1)
        last_pred = last_pred + tf.random.normal(shape=last_pred.shape, stddev=sigma_obs)
        last_pred = tf.expand_dims(last_pred, axis=-2)
        inp = tf.concat([inp, last_pred], axis=-2)
        tar = tf.concat([tar, tf.zeros(shape=(tar.shape[0], tar.shape[1], 1, tar.shape[-1]))], axis=-2)
        smc_transformer.seq_len += 1

    mean_preds = tf.reduce_mean(preds, axis=1)
    mean_preds = tf.squeeze(mean_preds)
    preds_future = preds[:,:,40:,:]
    preds_future = tf.squeeze(preds_future)
    if save_path is not None:
        np.save(save_path, mean_preds)
    if save_path is not None:
        np.save(save_path_preds, preds_future)

    return preds_future, mean_preds



# def inference_multistep_4(smc_transformer, test_sample, N=2, save_path=None, save_path_preds=None, past_len=40, future_len=20):
#     P = smc_transformer.cell.num_particles
#     d_model = smc_transformer.d_model
#     sigma_obs = tf.math.sqrt(smc_transformer.cell.Sigma_obs)
#     # forward pass on test_sample_past
#     smc_transformer.cell.add_stop_resampling(past_len)
#     N_mean_preds = []
#     N_preds_t = []
#     for n in range(N):
#         preds_t = []
#         smc_transformer.seq_len = past_len
#         inp, tar = split_input_target(test_sample[:, :, :past_len + 1, :])
#         for i in range(future_len+1):
#             (preds, _), _, _ = smc_transformer(inp, tar)  # K,V shape (1, P, 40, D)
#             inp_p = np.random.randint(P)
#             last_pred = tf.expand_dims(preds[:,inp_p,-1,:], axis=1)
#             preds_t.append(last_pred)
#             last_pred = last_pred + tf.random.normal(shape=last_pred.shape, stddev=sigma_obs)
#             last_pred = tf.expand_dims(last_pred, axis=-2)
#             inp = tf.concat([inp, last_pred], axis=-2)
#             tar = tf.concat([tar, tf.zeros(shape=(tar.shape[0], tar.shape[1], 1, tar.shape[-1]))], axis=-2)
#             smc_transformer.seq_len += 1
#
#         mean_preds = tf.reduce_mean(preds, axis=1)
#         mean_preds = tf.squeeze(mean_preds)
#         N_mean_preds.append(mean_preds)
#         preds_t = tf.stack(preds_t, axis=0)
#         N_preds_t.append(preds_t)
#
#     N_preds_t= tf.stack(N_preds_t, axis=0) # (N,20)
#     N_preds_t = tf.squeeze(N_preds_t)
#
#     N_mean_preds = tf.stack(N_mean_preds, axis=0) # (N,S)
#
#     if save_path is not None:
#         np.save(save_path, N_mean_preds)
#     if save_path_preds is not None:
#         np.save(save_path_preds, N_preds_t)
#
#     return N_preds_t, N_mean_preds



def get_empirical_distrib(mean_NP, sigma_obs, N_est, N, P):
    emp_distrib = np.zeros(shape=N_est)
    mean_NP = tf.reshape(mean_NP, shape=(N, P))
    for i in range(N_est):
        ind_n = np.random.randint(0, N)
        ind_p = np.random.randint(0, P)
        sampled_mean = mean_NP[ind_n, ind_p]
        sample = sampled_mean + tf.random.normal(shape=sampled_mean.shape, stddev=sigma_obs)
        emp_distrib[i] = sample.numpy()
    return emp_distrib

def get_empirical_distrib_2(mean_NP, sigma_obs, N_est, P):
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