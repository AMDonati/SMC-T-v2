import numpy as np
import tensorflow as tf
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
import os
from utils.utils_train import restoring_checkpoint, write_to_csv
from train.loss_functions import CustomSchedule


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


def EM_after_training(smc_transformer, inputs, targets, save_path, iterations=100):

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


if __name__ == '__main__':

    # ---------- Load Test Data -----------------------------------------------------------------------------------------------------------------

    data_path = '../../data/covid_test_data.npy'
    test_data = np.load(data_path)

    # ---------- Load SMC Transformer with learned params ------------------------------------------------------------------------------------
    # out_path = "../../output/covid_SMC_T/covid_smc_t_10_p"
    # list_sigmas = [0.5045, 0.4787, 0.4313, 0.5879]
    # dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], list_sigmas))
    # Sigma_obs = 0.0368
    # num_particles = 10

    out_path = "../../output/covid_SMC_T/covid_smc_t_60_p"
    list_sigmas = [0.51636, 0.49633, 0.46728, 0.48299]
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], list_sigmas))
    Sigma_obs = 0.03406
    num_particles = 60

    d_model = 8
    dff = 16
    N = 100

    smc_transformer = SMC_Transformer(d_model=d_model,
                                      output_size=1,
                                      seq_len=60,
                                      full_model=True,
                                      dff=dff)

    # get checkpoint path for SMC_Transformer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    checkpoint_path = os.path.join(out_path, "checkpoints")
    smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer_1")
    smc_T_ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                                     optimizer=optimizer)
    smc_T_ckpt_manager = tf.train.CheckpointManager(smc_T_ckpt, smc_T_ckpt_path, max_to_keep=50)
    num_epochs_smc_T = restoring_checkpoint(ckpt_manager=smc_T_ckpt_manager, ckpt=smc_T_ckpt,
                                            args_load_ckpt=True, logger=None)

    smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas, sigma_obs=Sigma_obs, num_particles=num_particles)

    # ------------------------------------- check test loss ----------------------------------------------------------------------------------

    # computing loss on test_dataset:
    inputs, targets = split_input_target(test_data[:,np.newaxis,:,:])
    (preds_test, preds_test_resampl), _, _ = smc_transformer(inputs=inputs,
                                                            targets=targets)  # predictions test are the ones not resampled.
    test_metric_avg_pred = tf.keras.losses.MSE(targets, tf.reduce_mean(preds_test, axis=1, keepdims=True))  # (B,1,S)
    test_metric_avg_pred = tf.reduce_mean(test_metric_avg_pred, axis=[1,2])
    top_k, top_i = tf.math.top_k(test_metric_avg_pred, k=15)
    print('indices with lowest loss', top_i)
    test_metric_avg_pred = tf.reduce_mean(test_metric_avg_pred)
    print('test loss', test_metric_avg_pred)

    # ------ sigmas estimation post-training --------------------------------------------------------------------------------------------------
    index = 76
    test_sample = test_data[index]
    print('test_sample', test_sample)
    test_sample = tf.convert_to_tensor(test_sample)
    test_sample = tf.reshape(test_sample, shape=(1, 1, test_sample.shape[-2], test_sample.shape[-1]))

    inputs, targets = split_input_target(test_sample[:, :, :41, :])

    smc_transformer.seq_len = 40
    save_path_EM = save_path = os.path.join(out_path, 'sigmas_after_EM_{}.npy'.format(index))

    Sigma_obs, sigmas = EM_after_training(smc_transformer=smc_transformer, inputs=inputs, targets=targets, save_path=save_path_EM)

    # ---------------------------- launching inference ----------------------------------------------------------------------------------------
    save_path_means = os.path.join(out_path, 'mean_preds_sample_{}_N_{}.npy'.format(index, N))
    save_path_means_multi = os.path.join(out_path, 'mean_preds_sample_{}_N_{}_multi.npy'.format(index, N))
    preds_NP, mean_preds = inference_onestep(smc_transformer=smc_transformer,
                                             test_sample=test_sample,
                                             N=N,
                                             save_path=save_path_means)
    means_NP_multi, mean_preds_multi = inference_multistep(smc_transformer=smc_transformer,
                                                           test_sample=test_sample,
                                                           N=N,
                                                           save_path=save_path_means_multi)

    sigma_obs = tf.math.sqrt(smc_transformer.cell.Sigma_obs)
    P = smc_transformer.cell.num_particles

    save_path_distrib = os.path.join(out_path, 'distrib_future_timesteps_sample_{}_N_{}.npy'.format(index, N))
    distrib_future_timesteps = []
    for t in range(20):
        mean_NP = preds_NP[:, t]
        emp_distrib = get_empirical_distrib(mean_NP, sigma_obs=sigma_obs, N_est=1000, N=N, P=P)
        distrib_future_timesteps.append(emp_distrib)
    distrib_future_timesteps = np.stack(distrib_future_timesteps, axis=0)
    print('distrib future timesteps', distrib_future_timesteps.shape)
    np.save(save_path_distrib, distrib_future_timesteps)

    save_path_distrib_multi = os.path.join(out_path, 'distrib_future_timesteps_sample_{}_N_{}_multi.npy'.format(index, N))
    distrib_future_timesteps = []
    for t in range(20):
        mean_NP = means_NP_multi[:, t]
        emp_distrib = get_empirical_distrib(mean_NP, sigma_obs=sigma_obs, N_est=1000, N=N, P=P)
        distrib_future_timesteps.append(emp_distrib)
    distrib_future_timesteps = np.stack(distrib_future_timesteps, axis=0)
    print('distrib future timesteps', distrib_future_timesteps.shape)
    np.save(save_path_distrib_multi, distrib_future_timesteps)


