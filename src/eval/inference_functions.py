import tensorflow as tf
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.Baselines.Transformer_without_enc import Transformer
from models.Baselines.LSTMs import build_LSTM_for_regression
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
import numpy as np
import scipy.stats
import os
import ot
import tensorflow_probability as tfp

from utils.KL_divergences_estimators import naive_estimator
from eval.Transformer_dropout import MC_Dropout_Transformer


#import scipy.stats.wasserstein_distance as wass_distance
#import scipy.stats.entropy as KL_distance
#https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.norm.html


def inference_Baseline_T_MC_Dropout_1D(inputs, transformer, transformer_w_dropout, num_mc_samples, num_timesteps, output_path):
  '''
  :param inputs: (B,S,F)
  :param transformer:
  :param num_mc_samples:
  :param num_timsteps:
  :return:
  '''
  s = tf.shape(inputs)[1] - num_timesteps
  inp_model = inputs[:, :s, :]
  inp_inference = inputs[:, s:, :]
  # forward pass on the first s inputs:
  predictions, _ = transformer(inputs=inp_model,
                                      training=False,
                                      mask=create_look_ahead_mask(s)) # predictions (B,s,1)

  last_pred = predictions [:,-1,:] # (B,1)
  #last_pred = tf.expand_dims(last_pred, axis=1)
  list_true_preds = []
  for t in range(num_timesteps):
    obs_feat = inp_inference[:,t,1:]
    new_input = tf.concat([last_pred,obs_feat], axis=1) # (B,F)
    new_input = tf.expand_dims(new_input, axis=1)
    inp_model = tf.concat([inp_model, new_input], axis=1) # (B,s+1,F)
    seq_len = tf.shape(inp_model)[1]
    if t == num_timesteps - 1:
      MC_Dropout_predictions = MC_Dropout_Transformer(transformer=transformer_w_dropout,
                                                      test_dataset=inp_model,
                                                      seq_len=seq_len,
                                                      num_samples=num_mc_samples,
                                                      task='synthetic',
                                                      stats=None,
                                                      output_path=None,
                                                      logger=None)  # (B,N,S,1)
    predictions, _ = transformer(inputs=inp_model,
                                   training=False,
                                   mask=create_look_ahead_mask(seq_len))
    last_pred = predictions[:,-1,:]
    list_true_preds.append(last_pred)

  # select only the inference part (number of timesteps):
  MC_Dropout_predictions = MC_Dropout_predictions[:,:,s:,:].numpy()
  list_true_preds = [x.numpy() for x in list_true_preds]

  MC_preds_path = os.path.join(output_path, 'Baseline_T_MC_Dropout_preds_inference.npy')
  list_true_preds_path = os.path.join(output_path, 'Baseline_T_true_preds.npy')
  np.save(file=MC_preds_path, arr=MC_Dropout_predictions)
  np.save(file=list_true_preds_path, arr=list_true_preds)

  return MC_Dropout_predictions, list_true_preds


def MC_Dropout_LSTM(lstm_model, inp_model, mc_samples):
  '''
  :param LSTM_hparams: shape_input_1, shape_input_2, shape_ouput, num_units, dropout_rate
  :param inp_model: array of shape (B,S,F)
  :param mc_samples:
  :return:
  '''
  list_predictions = []
  for i in range(mc_samples):
      predictions_test = lstm_model(inputs=inp_model) # (B,S,1)
      list_predictions.append(predictions_test)
  predictions_test_MC_Dropout = tf.stack(list_predictions, axis=1)  # shape (B, N, S, 1)

  return predictions_test_MC_Dropout

def inference_LSTM_MC_Dropout_1D(inputs, lstm_model, lstm_w_dropout, num_mc_samples, num_timesteps, output_path):
  '''
  :param inputs:
  :param lstm_model:
  :param lstm_w_dropout:
  :param num_mc_samples:
  :param num_timesteps:
  :param output_path:
  :return:
  '''
  s = tf.shape(inputs)[1] - num_timesteps
  inp_model = inputs[:, :s, :]
  inp_inference = inputs[:, s:, :]
  # forward pass on the first s inputs:
  predictions = lstm_model(inp_model) # predictions (B,s,1)

  last_pred = predictions[:,-1,:] # (B,1)
  #last_pred = tf.expand_dims(last_pred, axis=1)
  list_true_preds = []
  for t in range(num_timesteps):
    obs_feat = inp_inference[:,t,1:]
    new_input = tf.concat([last_pred,obs_feat], axis=1) # (B,F)
    new_input = tf.expand_dims(new_input, axis=1)
    inp_model = tf.concat([inp_model, new_input], axis=1) # (B,s+1,F)
    if t == num_timesteps - 1:
      MC_Dropout_predictions = MC_Dropout_LSTM(lstm_model=lstm_w_dropout, inp_model=inp_model, mc_samples=num_mc_samples)
    predictions = lstm_model(inp_model)
    last_pred = predictions[:,-1,:]  # (B,1)
    list_true_preds.append(last_pred)

  # select only the inference part (number of timesteps):
  MC_Dropout_predictions = MC_Dropout_predictions[:,:,s:,:].numpy()
  list_true_preds = [x.numpy() for x in list_true_preds]

  MC_preds_path = os.path.join(output_path, 'LSTM_MC_Dropout_preds_inference.npy')
  list_true_preds_path = os.path.join(output_path, 'LSTM_true_preds.npy')
  np.save(file=MC_preds_path, arr=MC_Dropout_predictions)
  np.save(file=list_true_preds_path, arr=list_true_preds)

  return MC_Dropout_predictions, list_true_preds


def inference_function_multistep(inputs, smc_transformer, N_prop, N_est, num_particles, num_timesteps, sigma, output_path, sample_pred=True, layer_norm=True):
  '''
  :param inputs: shape (B,S,F)
  :param smc_transformer:
  :param N_prop:
  :param N_est:
  :param num_particles:
  :param num_timesteps:
  :param sample_pred:
  :return:
  '''
  list_X_pred_NP, list_r_NP = [], []
  N = N_prop

  # call of the smc_transformer on inputs:
  s = tf.shape(inputs)[1] - num_timesteps
  mask = create_look_ahead_mask(s)
  smc_transformer.noise_SMC_layer = True
  smc_transformer.cell.noise = True
  smc_transformer.cell.mha_smc.noise = True
  smc_transformer.num_particles = num_particles
  smc_transformer.cell.num_particles = num_particles
  smc_transformer.sigma = sigma
  smc_transformer.cell.mha_smc.sigma_scalar = sigma

  inp_model = inputs[:, :s, :]  # (1:s-1) used as the input tensor of the SMC Cell.
  true_labels = inputs[:, 1:s, :]  # (B,s,F)
  inp_inference = inputs[:, s:, :]
  outputs, _ = smc_transformer(inputs=inp_model,
                                  training=False,
                                  mask=mask)
  predictions, _, w_s, (K0_s, V0_s, Us) = outputs # predictions (B,P,s,F)

  # ----- computation of the learned std ------------------------------------------------------------------------------------------------------------------------------
  # computing 'learned' omega:
  true_labels = tf.expand_dims(true_labels, axis=1)  # (B,1,s,F)
  true_labels = tf.tile(true_labels, multiples=[1, num_particles, 1, 1])  # (B,P,s,F)
  diff = true_labels - predictions  # (B,P,s,F)
  diff = tf.expand_dims(diff, axis=-2)  # (B,P,s,1,F)
  matmul = tf.matmul(diff, diff, transpose_a=True)  # (B,P,s,F,F)
  w_s_reshaped = tf.reshape(w_s, shape=(tf.shape(w_s)[0], tf.shape(w_s)[1], 1, 1, 1))
  covariance_matrix = tf.reduce_sum(w_s_reshaped * matmul, axis=1)  # (B,s,F,F)
  covariance_matrix = tf.reduce_mean(covariance_matrix, axis=1)  # (B,F,F)
  covariance_matrix = tf.reduce_mean(covariance_matrix, axis=0)  # (F,F)
  variance_part = tf.linalg.diag_part(covariance_matrix)  # (F,) ?
  learned_std_multivariate = tf.sqrt(variance_part)

  # set omega as the learned variance
  smc_transformer.omega = covariance_matrix
  smc_transformer.cell.omega = covariance_matrix

  # preprocessing initial input:
  input = inputs[:, -1, :] # (B,1,F)
  input = tf.expand_dims(input, axis=1)  # (B,1,F)
  input = tf.expand_dims(input, axis=1) # (B,1,1,F)
  input = tf.tile(input, multiples=[1, num_particles, 1, 1])  # (B,P,1,F)

  # adding zeros to KO_s and V0_s
  shape_future = (tf.shape(K0_s)[0], num_particles, num_timesteps, tf.shape(K0_s)[-1])
  future_K = tf.zeros(shape=shape_future)
  future_V = tf.zeros(shape=shape_future)
  K_init = tf.concat([K0_s, future_K], axis=2)
  V_init = tf.concat([V0_s, future_V], axis=2)
  K = K_init
  V= V_init
  X_pred_NP = input # (B,P,1,F)

  for t in range(num_timesteps):
    X_pred_NP = smc_transformer.input_dense_projection(X_pred_NP) # (B,P,1,D) or # (B,N*P,1,D)
    X_pred_NP, r_N_P, (K, V) = smc_transformer.cell.inference_function(inputs=X_pred_NP, K=K, V=V, num_samples=N, t=t+s, inf_timestep=t, layer_norm=layer_norm)
    list_X_pred_NP.append(X_pred_NP) # (B,P,1,F)
    list_r_NP.append(r_N_P)

  # ---------------------------------------------------------sampling N_est predictions for each timestep -------------------------------#
  list_preds_multistep = []
  if sample_pred:
    for r in list_r_NP:
      # reshape to have a tensor of shape (B,N,P,1,D)
      new_shape = (tf.shape(r)[0], -1, N, num_particles, tf.shape(r)[-2], tf.shape(r)[-1])
      r = tf.reshape(r, shape=new_shape)  # (B,-1,N,P,1,D)
      r = tf.squeeze(r, axis=1)  # (B,N,P,1,D)

      list_pred_t = []
      for _ in range(N_est):
        # select n* and p*:
        p_ = tf.random.categorical(logits=w_s, num_samples=1)
        uniform_logits = tf.constant([[1 / N for _ in range(N)] for _ in range(tf.shape(r)[0])])
        n_ = tf.random.categorical(logits=uniform_logits, num_samples=1)
        r_ = tf.gather(r, p_, axis=2, batch_dims=1)
        r_ = tf.gather(r_, n_, axis=1, batch_dims=1)  # (B,1,1,1,D)
        r_ = tf.squeeze(tf.squeeze(r_, axis=1), axis=1)  # (B,1,D)
        mean_r_ = smc_transformer.final_layer(r_)  # (B,1,F)
        mean_r_ = tf.squeeze(mean_r_) # (B,F)
        distrib = tfp.distributions.MultivariateNormalFullCovariance(loc=mean_r_, covariance_matrix=covariance_matrix)
        X_pred = distrib.sample(sample_shape=()) # (B,F)

        #X_pred = mean_r_ + tf.random.normal(shape=tf.shape(mean_r_), stddev=omega)  # (B,1,F)
        list_pred_t.append(X_pred)
      tensor_pred_t = tf.stack(list_pred_t, axis=1)  # (B, N_est,F)
      list_preds_multistep.append(tensor_pred_t.numpy())

    # -------------------------------- computing mean for each N*P Gaussian distribution to plot the complete distribution -----------------
    # get the mean of the mix of gaussian distributions from r
    list_mean_NP = [smc_transformer.final_layer(r_NP) for r_NP in list_r_NP]
    list_mean_NP = [mean.numpy() for mean in list_mean_NP]  # transform list_mean_NP in a numpy array
    list_mean_NP = [m.reshape(m.shape[0], N, num_particles, m.shape[-1]) for m in list_mean_NP]  # reshape (B,NP,1,1) to (B,N,P,F)

    # ------------------------------- save arrays for plotting the predicted probability density function------------------------------------
    list_X_pred_NP = [tf.squeeze(x, axis=-2).numpy() for x in list_X_pred_NP]
    w_s = w_s.numpy()
    gaussian_means_path = output_path + '/' + 'pred_gaussian_means_per_timestep_P_{}.npy'.format(num_particles)
    sampled_distrib_path = output_path + '/' + 'preds_sampled_per_timestep_P_{}.npy'.format(num_particles)
    sampling_weights_path = output_path + '/' + 'sampling_weights_P_{}.npy'.format(num_particles)
    all_preds_path = output_path + '/' + 'list_X_pred_NP_P_{}.npy'.format(num_particles)
    covariance_matrix_path = output_path + '/' + 'covariance_matrix_P_{}.npy'.format(num_particles)

    np.save(file=gaussian_means_path, arr=list_mean_NP)
    np.save(file=sampled_distrib_path, arr=list_preds_multistep)
    np.save(file=sampling_weights_path, arr=w_s)
    np.save(file=all_preds_path, arr=list_X_pred_NP)
    np.save(file=covariance_matrix_path, arr=covariance_matrix)

  return (list_r_NP, list_X_pred_NP), list_preds_multistep, w_s, (learned_std_multivariate, covariance_matrix)

def omega_estimation_algorithm(smc_transformer, input_model, mask, num_updates, alpha, k):
  batch_size = tf.shape(input_model)[0]
  assert batch_size == 1

  omega_init = smc_transformer.omega
  list_omegas_estimated = [omega_init]
  for i in range(1, num_updates+1):
    outputs, _ = smc_transformer(inputs=input_model,
                                 training=False,
                                 mask=mask)  # (B,P,s,1)

    predictions, _, w_s, (K0_s, V0_s, U_s) = outputs # U_s > (B,P,1,F)
    U_s = tf.squeeze(U_s, axis=-2) # (B,P,F)
    grad = tf.reduce_mean(U_s, axis=1)
    rate = (k/i)**alpha
    new_variance = (smc_transformer.omega)**2 + tf.scalar_mul(rate,grad) # (B,F). B=1, F=1.
    new_variance = tf.reshape(new_variance, shape=()) # scalar.
    new_variance = tf.math.maximum(0, new_variance)  # trick to avoid non-negative values.
    new_omega = tf.sqrt(new_variance).numpy()
    list_omegas_estimated.append(new_omega)
    # updates internal parameters of the smc transformer with the new omega:
    smc_transformer.omega = new_omega
    smc_transformer.cell.omega = new_omega

  return list_omegas_estimated, smc_transformer, (w_s, K0_s, V0_s)


def inference_function_multistep_1D(inputs, smc_transformer, N_prop, N_est, num_particles, num_timesteps, sigma, omega, output_path, sample_pred=True, layer_norm=True):
  '''
  :param inputs: shape (B,S,F)
  :param smc_transformer:
  :param N_prop:
  :param N_est:
  :param num_particles:
  :param num_timesteps:
  :param sample_pred:
  :return:
  '''
  # ------ inference function -------------------------------------------------------------------------------------------------------------

  list_X_pred_NP, list_r_NP = [], []
  N = N_prop

  # call of the smc_transformer on inputs:
  s = tf.shape(inputs)[1] - num_timesteps
  mask = create_look_ahead_mask(s-1)
  smc_transformer.noise_SMC_layer = True
  smc_transformer.cell.noise = True
  smc_transformer.cell.mha_smc.noise = True
  smc_transformer.num_particles = num_particles
  smc_transformer.cell.num_particles = num_particles
  smc_transformer.cell.mha_smc.num_particles = num_particles
  smc_transformer.sigma = sigma
  smc_transformer.cell.mha_smc.sigma_scalar = sigma
  smc_transformer.omega = omega
  smc_transformer.cell.omega = omega

  inp_model = inputs[:,:s,:] # (1:s-1) used as the input tensor of the SMC Cell.
  #true_labels = inputs[:,1:s,0] # (B,s)
  inp_inference = inputs[:,s:,:]

  # adapting the seq length to the length of input_model seq.
  smc_transformer.seq_len = tf.shape(inp_model)[1] - 1
  smc_transformer.cell.seq_len = tf.shape(inp_model)[1] - 1
  outputs, _ = smc_transformer(inputs=inp_model,
                                  training=False,
                                  mask=mask) # (B,P,s,1)

  predictions, _, w_s, (K0_s, V0_s, R0_s) = outputs

  # # -------- online estimation of omega -----------------------------------------------------------------------------------------------------------
  # list_omegas_estimated, smc_transformer, (w_s, K0_s, V0_s) = omega_estimation_algorithm(smc_transformer=smc_transformer,
  #                                                                                        input_model=inp_model,
  #                                                                                        mask=mask,
  #                                                                                        num_updates=num_updates,
  #                                                                                        alpha=alpha,
  #                                                                                        k=k)
  # omega = list_omegas_estimated[-1]
  # assert smc_transformer.omega == omega
  # assert smc_transformer.cell.omega == omega

  # preprocessing initial input:
  input = inp_model[:, -1, :] # (B,1,F) # use last input of inp_model (only use as target in the forward pass above.)
  input = tf.expand_dims(input, axis=1)  # (B,1,F)
  input = tf.expand_dims(input, axis=1) # (B,1,1,F)
  input = tf.tile(input, multiples=[1, num_particles, 1, 1])  # (B,P,1,F)

  # adding zeros to KO_s and V0_s
  shape_future = (tf.shape(K0_s)[0], num_particles, num_timesteps, tf.shape(K0_s)[-1])
  future_K = tf.zeros(shape=shape_future)
  future_V = tf.zeros(shape=shape_future)
  K_init = tf.concat([K0_s, future_K], axis=2)
  V_init = tf.concat([V0_s, future_V], axis=2)
  K = K_init # (B,P,s+num_timesteps,D)
  V = V_init
  X_pred_NP = input # (B,P,1,F)

  # adjust seq_len parameter
  smc_transformer.seq_len = tf.shape(K)[2]
  smc_transformer.cell.seq_len = tf.shape(K)[2]

  for t in range(num_timesteps):
    if tf.shape(X_pred_NP)[-1] == 1:
      NP = tf.shape(X_pred_NP)[1]
      obs_features = inp_inference[:,t,1:] # (B,F=2)
      obs_features = tf.expand_dims(obs_features, axis=1) # (B,1,F=2)
      obs_features_NP = tf.expand_dims(obs_features, axis=1) # (B,1,1,F=2)
      obs_features_NP = tf.tile(obs_features_NP, multiples=[1,NP,1,1]) # (B,NP,1,F=2)
      X_pred_NP = tf.concat([X_pred_NP, obs_features_NP], axis=-1)
    X_pred_NP = smc_transformer.input_dense_projection(X_pred_NP) # (B,P,1,D) or # (B,N*P,1,D)
    X_pred_NP, r_N_P, (K, V) = smc_transformer.cell.inference_function(inputs=X_pred_NP, K=K, V=V, num_samples=N, t=t+s-1, inf_timestep=t, layer_norm=layer_norm)
    list_X_pred_NP.append(X_pred_NP)
    list_r_NP.append(r_N_P)

  # ---------------------------------------------------------sampling N_est predictions for each timestep -------------------------------#
  if sample_pred:
    list_preds_multistep = []
    for r in list_r_NP:
      # reshape to have a tensor of shape (B,N,P,1,D)
      new_shape = (tf.shape(r)[0], -1, N, num_particles, tf.shape(r)[-2], tf.shape(r)[-1])
      r = tf.reshape(r, shape=new_shape) # (B,-1,N,P,1,D)
      r = tf.squeeze(r, axis=1) # (B,N,P,1,D)

      list_pred_t = []
      for _ in range(N_est):
        # select n* and p*:
        p_ = tf.random.categorical(logits=w_s, num_samples=1)
        uniform_logits = tf.constant([[1/N for _ in range(N)] for _ in range(tf.shape(r)[0])])
        n_ = tf.random.categorical(logits=uniform_logits, num_samples=1)
        r_ = tf.gather(r, p_, axis=2, batch_dims=1)
        r_ = tf.gather(r_, n_, axis=1, batch_dims=1) # (B,1,1,1,D)
        r_ = tf.squeeze(tf.squeeze(r_, axis=1), axis=1) # (B,1,D)
        mean_r_ = smc_transformer.final_layer(r_)  # (B,1,1)
        X_pred = mean_r_ + tf.random.normal(shape=tf.shape(mean_r_), stddev=omega) # (B,1,F=1)
        list_pred_t.append(X_pred)
      tensor_pred_t = tf.stack(list_pred_t, axis=1)  # (B,N_est,1,F=1)
      tensor_pred_t = tf.squeeze(tf.squeeze(tensor_pred_t, axis=-1), axis=-1) # (B, N_est)
      list_preds_multistep.append(tensor_pred_t.numpy())

    # -------------------------------- computing mean for each N*P Gaussian distribution to plot the complete distribution -----------------
    # get the mean of the mix of gaussian distributions from r
    list_mean_NP = [smc_transformer.final_layer(r_NP) for r_NP in list_r_NP]
    list_mean_NP = [mean.numpy() for mean in list_mean_NP] # transform list_mean_NP in a numpy array
    list_mean_NP = [m.reshape(m.shape[0], N, num_particles, m.shape[-1]) for m in list_mean_NP] # reshape (B,NP,1,1) to (B,N,P,1)

    # ------------------------------- save arrays for plotting the predicted probability density function------------------------------------
    list_X_pred_NP = [tf.squeeze(x, axis=-2).numpy() for x in list_X_pred_NP]
    w_s = w_s.numpy()
    gaussian_means_path = output_path + '/' + 'pred_gaussian_means_per_timestep_P_{}.npy'.format(num_particles)
    sampled_distrib_path = output_path + '/' + 'preds_sampled_per_timestep_P_{}.npy'.format(num_particles)
    sampling_weights_path = output_path + '/' + 'sampling_weights_P_{}.npy'.format(num_particles)
    all_preds_path = output_path + '/' + 'list_X_pred_NP_P_{}.npy'.format(num_particles)

    np.save(file=gaussian_means_path, arr=list_mean_NP)
    np.save(file=sampled_distrib_path, arr=list_preds_multistep)
    np.save(file=sampling_weights_path, arr=w_s)
    np.save(file=all_preds_path, arr=list_X_pred_NP)

  return (list_mean_NP, list_X_pred_NP), list_preds_multistep, w_s


def generate_empirical_distribution_1D(inputs, matrix_A, cov_matrix, N_est, num_timesteps, output_path):
  '''
  :param inputs: ts input data of shape (B,S,F)
  :param matrix_A: matrix of autogressive model > shape (F * F)
  :param cov_matrix: covariance matrix for the gaussian noise > shape (F,1)
  :param N_est: number of samples to draw for the empirical distribution
  :param num_timesteps: number of timesteps for the inference
  :return:
  '''
  s = tf.shape(inputs)[1] - num_timesteps
  inp_model = inputs[:, :s, :]
  inp_inference = inputs[:, s:, :]
  last_input = inp_model[:,-1,:] # (B,F)
  num_features = tf.shape(last_input)[-1]
  batch_size = tf.shape(last_input)[0]
  list_preds_sampled = []
  list_mean_per_timestep = []

  for t in range(num_timesteps):
    list_pred_t = []
    mean = tf.matmul(last_input, matrix_A)
    list_mean_per_timestep.append(mean)

    for n in range(N_est):
      new_input = mean + tf.random.normal(stddev=cov_matrix, shape=(1, num_features)) # (B,F)
      new_input = tf.expand_dims(new_input[:,0], axis=-1) # (B,1)
      list_pred_t.append(new_input)

    tensor_pred_t = tf.stack(list_pred_t, axis=1)  # (B,N_est,1)
    list_preds_sampled.append(tensor_pred_t)
    # compute new_input for next timestep
    sample_ind = np.random.randint(0, N_est)
    last_input = list_pred_t[sample_ind]  # (B,1)
    obs_features = inp_inference[:,t,1:] #(B,F_obs=2)
    last_input = tf.concat([last_input, obs_features], axis=-1)

  # transforming tensors into numpy arrays.
  list_preds_sampled = [tf.squeeze(x, axis=-1).numpy() for x in list_preds_sampled] # (B,N_est)
  list_mean_per_timestep = [m.numpy() for m in list_mean_per_timestep]
  # saving information in .npy files
  true_distrib_path = output_path + '/' + 'true_empirical_distrib.npy'
  true_means_path = output_path + '/' + 'true_gaussian_means.npy'
  np.save(file=true_distrib_path, arr=list_preds_sampled)
  np.save(file=true_means_path, arr=list_mean_per_timestep)

  return list_preds_sampled, list_mean_per_timestep

def generate_empirical_distribution(inputs, matrix_A, cov_matrix, N_est, num_timesteps, output_path):
  '''
  :param inputs: ts input data of shape (B,S,F)
  :param matrix_A: matrix of autogressive model > shape (F * F)
  :param cov_matrix: covariance matrix for the gaussian noise > shape (F,1)
  :param N_est: number of samples to draw for the empirical distribution
  :param num_timesteps: number of timesteps for the inference
  :return:
  '''
  s = tf.shape(inputs)[1] - num_timesteps
  inp_model = inputs[:, :s, :]
  inp_inference = inputs[:, s:, :]
  last_input = inp_model[:, -1, :]  # (B,F)
  num_features = tf.shape(last_input)[-1]
  list_preds_sampled = []
  list_mean_per_timestep = []

  for t in range(num_timesteps):
    list_pred_t = []
    mean = tf.matmul(last_input, matrix_A)
    list_mean_per_timestep.append(mean)
    for n in range(N_est):
      new_input = mean + tf.random.normal(stddev=cov_matrix, shape=(1, num_features)) # (B,F)
      list_pred_t.append(new_input)
    sample_ind = np.random.randint(0, N_est)
    last_input = list_pred_t[sample_ind] # (B,F)
    tensor_pred_t = tf.stack(list_pred_t, axis=1)  # (B,N_est,F)
    list_preds_sampled.append(tensor_pred_t)

  list_preds_sampled = [x.numpy() for x in list_preds_sampled]
  list_mean_per_timestep = [m.numpy for m in list_mean_per_timestep]

  #----- saving true empirical distrib and gaussian mean for plotting needs ------------------------
  true_distrib_path = output_path + '/' + 'true_empirical_distrib.npy'
  true_means_path = output_path + '/' + 'true_gaussian_means.npy'
  np.save(file=true_distrib_path, arr=list_preds_sampled)
  np.save(file=true_means_path, arr=list_mean_per_timestep)

  return list_preds_sampled, list_mean_per_timestep


# -------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  num_particles_training = 1
  seq_len = 24
  b = 1
  F = 3 # multivariate case.
  num_layers = 1
  d_model = 12
  num_heads = 4
  dff = 48
  maximum_position_encoding = seq_len
  sigma = 0.05
  data_type = 'time_series_multi'
  task_type = 'regression'
   # vocabulary size or number of classes.
  noise_encoder = False
  noise_SMC_layer = False
  rate = 0
  omega = 0.3
  target_feature = None
  C = F if target_feature is None else 1
  maximum_position_encoding = 50

  sample_transformer = SMC_Transformer(d_model=d_model, output_size=C, seq_len=seq_len, full_model=False, dff=)

  inputs = tf.random.uniform(shape=(b,seq_len+1,F))

  # ----- test of weighted average for the computation of the learned variance -------------------------------------------------------
  w = tf.constant([[0.2, 0.8], [0.5, 0.5]], dtype=tf.float32) #(B,P)
  diff = tf.constant([[[1,2,3], [4,5,6]], [[1,0,1], [10,100,1000]]], dtype=tf.float32) # (B,P,S)
  w = tf.expand_dims(w, axis=1) # (B,1,P)
  temp = tf.matmul(w, diff) # (B,P,F,F)

  # ------------------------------- test for multi-dim case -------------------------------------------------------------------
  # temp_input = tf.constant([[-4, 1], [-20, 0], [10,-1]], shape=(1,1,3,2), dtype=tf.float32) # (S,F)
  # temp_input = tf.tile(temp_input, [2,5,1,1])
  # temp_input = tf.expand_dims(temp_input, axis=-2) # (B,P,S,1,F)
  # #temp_input = tf.random.uniform(shape=(8,10,20,1,3))
  # matmul = tf.matmul(temp_input, temp_input, transpose_a=True) # (2,5,3,2,2)
  # w = tf.ones(shape=(8,10,1,1,1), dtype=tf.float32)
  # mult = w * matmul

  # ---------------------------- test of numpy.random.normal - multivariate case -----------------------------------------------
  temp_mean = tf.random.uniform(shape=(8,3))
  temp_scale = tf.constant([[0.2, 0.01, 0.08], [0, 0.3, 0.05], [0.01, 0.02, 0.4]], dtype=tf.float32)
  #temp_scale = tf.linalg.diag_part(temp_scale)
  output = tfp.distributions.MultivariateNormalFullCovariance(loc=temp_mean, covariance_matrix=temp_scale)
  samples_temp = output.sample(sample_shape=(25,))

  # ---------------------- test of multi-step inference function - multivariate case----------------------------------------------------------------------
  num_samples = 5
  N_est = 25
  num_timesteps = 4
  num_particles_inference = 10

  output_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/temp'

  (list_mean_NP, list_X_pred_NP), list_preds_sampled, w_s, (learned_std, covariance_matrix) = inference_function_multistep(
    inputs=inputs,
    smc_transformer=sample_transformer,
    N_prop=num_samples,
    N_est=N_est,
    num_particles=num_particles_inference,
    num_timesteps=num_timesteps,
    output_path=output_path,
    sample_pred=True,
    sigma=0.1)

  # ------------------- test for multistep inference function - univariate case -----------------------------------------------------------------

  target_feature = 0
  C = 1
  omega = 0.3
  num_updates = 5
  alpha = 0.7

  sample_transformer = SMC_Transformer(d_model=d_model, output_size=C, seq_len=seq_len, full_model=False, dff=)

  (list_mean_NP, list_X_pred_NP), list_preds_sampled, w_s, list_learned_std = inference_function_multistep_1D(inputs=inputs,
                                                                                            smc_transformer=sample_transformer,
                                                                                            N_prop=num_samples,
                                                                                            N_est=N_est,
                                                                                            num_particles=num_particles_inference,
                                                                                            num_timesteps=num_timesteps,
                                                                                            output_path=output_path,
                                                                                            sample_pred=True,
                                                                                            sigma=0.1,
                                                                                            num_updates=num_updates,
                                                                                            alpha=alpha)

  print('number of timesteps predicted', len(list_preds_sampled))
  print('example of preds', list_preds_sampled[0])
  print('number of examples per preds', (list_preds_sampled[0].shape[1]))
  print('std learned', list_learned_std)

  #list_gaussian_means = np.load(file=output_path + '/' + 'pred_gaussian_means_per_timestep.npy')

  #--------------- test of generate_empirical_distribution ----------------------------------------------------------------------------

  cov_matrix_3D = tf.constant([0.2, 0.3, 0.4], dtype=tf.float32)
  A_3D = tf.constant([[0.8, 0.1, 0], [0.2, 0.9, 0.2], [0, 0.1, 0.85]], dtype=tf.float32)

  list_empirical_dist, list_true_means = generate_empirical_distribution_1D(inputs=inputs,
                                                                            matrix_A=A_3D,
                                                                            cov_matrix=cov_matrix_3D,
                                                                            N_est=N_est,
                                                                            num_timesteps=num_timesteps,
                                                                            output_path=output_path)


  print('number of timesteps predicted - empirical distribution', len(list_preds_sampled))
  print('example of preds - empirical distribution', list_preds_sampled[0])
  print('number of examples per preds', list_preds_sampled[0].shape[1])

  #true_gaussian_means = np.load(output_path + '/' + 'true_gaussian_means.npy')


  # -------------- test of generate empirical distribution - multivariate case -------------------------------------------------------------

  list_empirical_dist, list_true_means = generate_empirical_distribution(inputs=inputs,
                                                                            matrix_A=A_3D,
                                                                            cov_matrix=cov_matrix_3D,
                                                                            N_est=N_est,
                                                                            num_timesteps=num_timesteps,
                                                                            output_path=output_path)

  #------ test of MC Dropout inference function -----------------------------------------------------------------------------------------
  B = 8
  S = 24
  num_feat = 3
  num_layers = 1
  d_model = 2
  dff = 8
  num_heads = 1
  target_vocab_size = 1
  maximum_position_encoding_baseline = 50
  rate = 0
  num_mc_samples = 25

  test_dataset = tf.random.uniform(shape=(B, S, num_feat))
  transformer = Transformer(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            target_vocab_size=target_vocab_size,
                            maximum_position_encoding=maximum_position_encoding_baseline,
                            data_type=data_type,
                            rate=rate)
  transformer_w_dropout = Transformer(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            target_vocab_size=target_vocab_size,
                            maximum_position_encoding=maximum_position_encoding_baseline,
                            data_type=data_type,
                            rate=0.1)

  output_path_T = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/temp'

  MC_Dropout_predictions, list_true_preds = inference_Baseline_T_MC_Dropout_1D(inputs=test_dataset,
                                                              transformer=transformer,
                                                              transformer_w_dropout=transformer_w_dropout,
                                                              num_mc_samples=num_mc_samples,
                                                              num_timesteps=1,
                                                              output_path=output_path_T)

  # ------------ test of MC-DROPOUT LSTM predictions ----------------------------------------------------------------------------------

  lstm = build_LSTM_for_regression(shape_input_1=S, shape_input_2=num_feat, shape_output=target_vocab_size, rnn_units=20, dropout_rate=0)
  lstm_w_dropout = build_LSTM_for_regression(shape_input_1=S, shape_input_2=num_feat, shape_output=target_vocab_size, rnn_units=20, dropout_rate=0.1)
  LSTM_MCDropout_predictions, list_LSTM_preds = inference_LSTM_MC_Dropout_1D(inputs=test_dataset,
                                                                             lstm_model=lstm,
                                                                             lstm_w_dropout=lstm_w_dropout,
                                                                             num_mc_samples=num_mc_samples,
                                                                             num_timesteps=4,
                                                                             output_path=output_path_T)

  # ----------- computation of the KL divergence ----------------------------------------------------------------------------------------
  #KL_measure = tf.keras.losses.KLDivergence()
  #KL_measure_2 = tf.keras.losses.KLDivergence(reduction=losses_utils.ReductionV2.NONE)

  for t, (true_distrib, pred_distrib) in enumerate(zip(list_empirical_dist, list_preds_sampled)):
    N_est = pred_distrib.shape[1]
    std_pred_distrib = np.std(pred_distrib, axis=1)
    std_pred_distrib = np.mean(std_pred_distrib, axis=0)
    #KL_dist = scipy.stats.entropy(pk=pred_distrib, qk=true_distrib, axis=1)
    wass_dist = ot.emd2_1d(x_a=true_distrib[0,:], x_b=pred_distrib[0,:])
    KL_dist = naive_estimator(true_distrib[0,:].reshape(N_est,1), pred_distrib[0,:].reshape(N_est,1))
    #wass_dist = scipy.stats.wasserstein_distance(true_distrib, pred_distrib)

    #print('KL distance for timestep {}: {}'.format(t, KL_distance))



# ------------------------------- OLD CODE FUNCTIONS ------------------------------------------------------------------------------------------------------------

