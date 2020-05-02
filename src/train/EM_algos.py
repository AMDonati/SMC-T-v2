import tensorflow as tf
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
import numpy as np


def EM_training_algo_1D(train_data, train_labels, smc_transformer, num_particles, sigma, omega_init, num_iter):
  '''
  :param train_data: shape (B,S,F)
  :param train_labels:  shape (B,S,1)
  :param smc_transformer:
  :param num_particles:
  :param sigma:
  :param omega_init:
  :param num_iter: for the EM algo.
  :return:
  '''
  # ------ inference function -------------------------------------------------------------------------------------------------------------
  K = num_iter

  # call of the smc_transformer on inputs:
  s = tf.shape(train_data)[1] - 1
  mask = create_look_ahead_mask(s)
  smc_transformer.noise_SMC_layer = True
  smc_transformer.cell.noise = True
  smc_transformer.cell.mha_smc.noise = True
  smc_transformer.num_particles = num_particles
  smc_transformer.cell.num_particles = num_particles
  smc_transformer.sigma = sigma
  smc_transformer.cell.mha_smc.sigma_scalar = sigma
  smc_transformer.omega = omega_init
  smc_transformer.cell.omega = omega_init

  list_sigma_obs = []
  list_std_obs = []
  for _ in range(K):
    outputs, _ = smc_transformer(inputs=train_data,
                                 training=False,
                                 mask=mask)  # (B,P,s,1)
    predictions, _, w_s, (K,V,R) = outputs
    index = np.random.randint(0,num_particles)
    index = 20
    sample_pred = predictions[index,:,:,0]
    sample_pred = sample_pred.numpy()
    K_sampl = K[index,:,:,0].numpy()
    R_sampl = R[index,:,:,0].numpy()
    w_sampl = w_s[index,:].numpy()

    # check if w contains a nan number
    bool_tens = tf.math.is_nan(w_s)
    has_nan = tf.math.reduce_any(bool_tens).numpy()
    assert has_nan == False

    # compute $\sigma_obs_k$
    true_labels = tf.expand_dims(train_labels, axis=1)  # (B,1,s)
    true_labels = tf.tile(true_labels, multiples=[1, num_particles, 1, 1])  # (B,P,s,1)
    square_diff = tf.square(predictions - true_labels)  # (B,P,s,1)
    square_diff = tf.squeeze(square_diff, axis=-1) # (B,P,s)
    sigma_obs_k = tf.reduce_mean(square_diff, axis=-1) # (B,P) # mean over timesteps.
    sigma_obs_k = tf.reduce_mean(sigma_obs_k, axis=-1)  # (B)
    sigma_obs_k = tf.reduce_mean(sigma_obs_k, axis=0)  # scalar.
    list_sigma_obs.append(sigma_obs_k.numpy())

    std_k = tf.sqrt(sigma_obs_k)  # tf.random.normal and np.random.normal takes as input the std and not the variance.
    list_std_obs.append(std_k.numpy())
    # update std_k in the smc transformer.
    smc_transformer.omega = std_k
    smc_transformer.cell.omega = std_k

  return list_sigma_obs, list_std_obs

if __name__ == "__main__":
  num_particles_training = 1
  seq_len = 24
  b = 64
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
  rate = 0
  omega = 0.3
  target_feature = 0
  C = F if target_feature is None else 1
  maximum_position_encoding = 50

  sample_transformer = SMC_Transformer(d_model=d_model, output_size=C, seq_len=seq_len, full_model=False, dff=)

  train_data = tf.random.uniform(shape=(b,seq_len+1,F))
  train_labels = tf.random.uniform(shape=(b,seq_len,1))

  num_particles = 10
  omega_init = 0.1
  num_iter = 50

  list_sigma_obs, list_std_k = EM_training_algo_1D(train_data=train_data,
                                                   train_labels=train_labels,
                                                   smc_transformer=sample_transformer,
                                                   num_particles=num_particles,
                                                   sigma=sigma,
                                                   omega_init=omega_init,
                                                   num_iter=num_iter)
