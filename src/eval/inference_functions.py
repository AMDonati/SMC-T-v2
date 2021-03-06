import numpy as np
import tensorflow as tf

def split_input_target(data):
    inp = data[:, :, :-1, :]
    tar = data[:, :, 1:, :]
    return inp, tar

def inference_onestep(smc_transformer, inputs, targets, save_path, past_len=40):
    # forward pass on test_sample_past
    smc_transformer.cell.add_stop_resampling(past_len)
    (preds, _), (K, V, _), _ = smc_transformer(inputs, targets)  # K,V shape (1, P, 60, 1)
    mean_preds = tf.reduce_mean(preds, axis=1)  # (shape 60)
    preds_future = preds[:, :, past_len:, :]
    if save_path is not None:
        np.save(save_path, mean_preds)
    return preds_future, mean_preds

def inference_multistep(smc_transformer, inputs, targets, future_input_features=None, past_len=40, future_len=20, save_path=None, save_path_preds=None):
    P = smc_transformer.cell.num_particles
    sigma_obs = tf.math.sqrt(smc_transformer.cell.Sigma_obs)
    # forward pass on test_sample_past
    smc_transformer.seq_len = past_len
    smc_transformer.cell.add_stop_resampling(past_len)
    for i in range(future_len+1):
        (preds, _), _, _ = smc_transformer(inputs, targets)  # K,V shape (1, P, 40, D)
        last_pred = preds[:,:,-1,:]
        last_pred = last_pred + tf.random.normal(shape=last_pred.shape, stddev=sigma_obs)
        if i == 0:
            inputs = tf.tile(inputs, multiples=[1,P,1,1])
            targets = tf.tile(targets, multiples=[1,P,1,1])
        if i < future_len:
            if future_input_features is not None:
                input_features = tf.tile(future_input_features[:,:,i,:], multiples=[1,P,1])
                last_pred = tf.concat([last_pred, input_features], axis=-1)
            last_pred = tf.expand_dims(last_pred, axis=-2)
            inputs = tf.concat([inputs, last_pred], axis=-2)
            targets = tf.concat([targets, tf.zeros(shape=(targets.shape[0], targets.shape[1], 1, targets.shape[-1]))], axis=-2) # dummy target (not used when resampling is stopped.)
            smc_transformer.seq_len += 1
    mean_preds = tf.reduce_mean(preds, axis=1)
    preds_future = preds[:,:,past_len:,:]
    if save_path is not None:
        np.save(save_path, mean_preds)
    if save_path_preds is not None:
        np.save(save_path_preds, preds_future)
    return preds_future, mean_preds

def get_distrib_all_timesteps(preds, sigma_obs, P, save_path_distrib, N_est=10, len_future=20):
    distrib_future_timesteps = []
    for t in range(len_future):
        mean_NP = preds[:,:,t,:]
        emp_distrib = get_empirical_distrib(mean_NP, sigma_obs=sigma_obs, N_est=N_est, P=P)
        distrib_future_timesteps.append(emp_distrib)
    distrib_future_timesteps = np.stack(distrib_future_timesteps, axis=0) # shape (S,N_est,B,F)
    distrib_future_timesteps = np.transpose(distrib_future_timesteps, axes=[2,1,0,3]) # shape (B,N_est,S,F)
    print('distrib future timesteps', distrib_future_timesteps.shape)
    if save_path_distrib is not None:
        np.save(save_path_distrib, distrib_future_timesteps)
    return distrib_future_timesteps

def get_empirical_distrib(mean_NP, sigma_obs, N_est, P):
    emp_distrib = []
    for i in range(N_est): #TODO: parallelize the process.
        ind_p = np.random.randint(0, P)
        sampled_mean = mean_NP[:, ind_p, :]
        sample = sampled_mean + tf.random.normal(shape=sampled_mean.shape, stddev=sigma_obs)
        emp_distrib.append(sample)
    emp_distrib = np.stack(emp_distrib, axis=0)
    return emp_distrib
