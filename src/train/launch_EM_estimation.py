import tensorflow as tf
import numpy as np

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from train.EM_algos import EM_training_algo_1D

from utils.utils_train import create_run_dir
from utils.utils_train import create_logger
from utils.utils_train import restoring_checkpoint
from train.loss_functions import CustomSchedule

import argparse
import json
import os

if __name__ == "__main__":

  #---- parsing arguments --------------------------------------------------------------
  parser = argparse.ArgumentParser()
  results_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/post_UAI_exp/results_ws155_632020'
  exp_path = 'time_series_multi_synthetic_heads_2_depth_6_dff_24_pos-enc_50_pdrop_0_b_256_target-feat_0_cv_False__particles_1_noise_False_sigma_0.05'
  default_out_folder = os.path.join(results_path, exp_path)
  default_data_folder = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data/train_data_synthetic_3_feat.npy'

  parser.add_argument("-out_folder", default=default_out_folder, type=str, help="path for the output folder with training result")
  parser.add_argument("-data_path", default=default_data_folder, type=str, help="path for the train data folder")
  #parser.add_argument("-sigma", default=0.05, type=float, help="value of the internal noise")
  #parser.add_argument("-omega", default=0.1, type=float, help="value of the external covariance of the gaussian noise")
  parser.add_argument("-num_iter", default=20, type=int, help="number of iterations for EM algo")

  args = parser.parse_args()
  output_path = args.out_folder

  config_path = os.path.join(output_path, 'config.json')
  train_data_path = args.data_path

  # ------ uploading the hparams info -----------------------------------------------------------------------------------------------------------------------------------

  with open(config_path) as f:
    hparams = json.load(f)

  # model params
  num_layers = hparams["model"]["num_layers"]
  num_heads = hparams["model"]["num_heads"]
  d_model = hparams["model"]["d_model"]
  dff = hparams["model"]["dff"]
  rate = hparams["model"]["rate"]  # p_dropout
  max_pos_enc_bas_str = hparams["model"]["maximum_position_encoding_baseline"]
  maximum_position_encoding_baseline = None if max_pos_enc_bas_str == "None" else max_pos_enc_bas_str
  max_pos_enc_smc_str = hparams["model"]["maximum_position_encoding_smc"]
  maximum_position_encoding_smc = None if max_pos_enc_smc_str == "None" else max_pos_enc_smc_str
  #mc_dropout_samples = hparams["model"]["mc_dropout_samples"]

  # task params
  data_type = hparams["task"]["data_type"]
  task_type = hparams["task"]["task_type"]
  task = hparams["task"]["task"]

  # smc params
  num_particles = hparams["smc"]["num_particles"]
  noise_encoder_str = hparams["smc"]["noise_encoder"]
  noise_encoder = True if noise_encoder_str == "True" else False
  noise_SMC_layer_str = hparams["smc"]["noise_SMC_layer"]
  noise_SMC_layer = True if noise_SMC_layer_str == "True" else False
  sigma = hparams["smc"]["sigma"]
  #omega = 1
  if task == 'synthetic':
    omega = hparams["smc"]["omega"]
  # computing manually resampling parameter
  resampling = True

  # optim params
  BATCH_SIZE = hparams["optim"]["BATCH_SIZE"]
  learning_rate = hparams["optim"]["learning_rate"]
  EPOCHS = hparams["optim"]["EPOCHS"]
  custom_schedule = hparams["optim"]["custom_schedule"]

  # adding RNN hyper-parameters
  rnn_bs = BATCH_SIZE
  rnn_units = hparams["RNN_hparams"]["rnn_units"]
  rnn_dropout_rate = hparams["RNN_hparams"]["rnn_dropout_rate"]

  # loading data arguments for the regression case
  file_path = hparams["data"]["file_path"]
  TRAIN_SPLIT = hparams["data"]["TRAIN_SPLIT"]
  VAL_SPLIT = hparams["data"]["VAL_SPLIT"]
  VAL_SPLIT_cv = hparams["data"]["VAL_SPLIT_cv"]
  cv_str = hparams["data"]["cv"]
  cv = True if cv_str == "True" else False
  target_feature = hparams["data"]["target_feature"]
  if target_feature == "None":
    target_feature = None

  if task_type == 'regression' and task == 'unistep-forcst':
    history = hparams["data"]["history"]
    step = hparams["data"]["step"]
    fname = hparams["data"]["fname"]
    col_name = hparams["data"]["col_name"]
    index_name = hparams["data"]["index_name"]

  # -------------- uploading the test dataset --------------------------------------------------------------------------------------------------------------------------
  train_dataset = np.load(train_data_path) # (B,S,F)
  train_dataset = train_dataset[:1000,:,:]
  seq_len = train_dataset.shape[1] - 1
  num_features = train_dataset.shape[-1]
  # convert it into a tf.tensor
  train_dataset = tf.convert_to_tensor(train_dataset, dtype=tf.float32)
  train_labels = train_dataset[:, 1:, target_feature]
  train_labels = tf.expand_dims(train_labels, axis=-1)
  # ---------------preparing the output path for inference -------------------------------------------------------------------------------------------------------------
  sigma = 0.05
  omega = 0.3
  num_iter = 20
  list_particles = [10,25,50,100]
  list_omega_init = [0.3,0.5,1]

  checkpoint_path = os.path.join(output_path, "checkpoints")

  # -------------- create the logging -----------------------------------------------------------------------------------------------------------------------------------
  out_file_log = output_path + '/' + 'EM_log.log'
  logger = create_logger(out_file_log)

  # --------------- restoring checkpoint for smc transformer ------------------------------------------------------------------------------------------------------------
  # define optimizer
  if custom_schedule == "True":
    learning_rate = CustomSchedule(d_model)

  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)
  target_vocab_size = num_features if target_feature is None else 1
  # create a SMC_Transformer
  smc_transformer = SMC_Transformer(num_layers=num_layers,
                                    d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    output_size=target_vocab_size,
                                    maximum_position_encoding=maximum_position_encoding_smc,
                                    num_particles=num_particles,
                                    sigma=sigma,
                                    sigma_obs=omega,
                                    rate=rate,
                                    noise_encoder=noise_encoder,
                                    noise_SMC_layer=noise_SMC_layer,
                                    seq_len=seq_len,
                                    data_type=data_type,
                                    task_type=task_type,
                                    resampling=resampling,
                                    target_feature=target_feature)

  # get checkpoint path for SMC_Transformer
  smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer")
  # create checkpoint manager
  smc_T_ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                                   optimizer=optimizer)

  smc_T_ckpt_manager = tf.train.CheckpointManager(smc_T_ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS)
  # restore latest checkpoint from out folder
  num_epochs_smc_T = restoring_checkpoint(ckpt_manager=smc_T_ckpt_manager, ckpt=smc_T_ckpt,
                                          args_load_ckpt=True, logger=logger)

  # ------------------------------------- compute latest statistics as a check -----------------------------------------------------------------------------------------
  # ------------------------------------- EM algo to learn $\sigmas...$ --------------------------------------------------------------------------------------------------
  #list_particles = [100]
  for omega_init in list_omega_init:

    logger.info('initial std...: {}'.format(omega_init))

    for num_particles in list_particles:

      #num_particles = 10
      logger.info('EM results for number of particles: {}'.format(num_particles))

      list_sigma_obs, list_std_k = EM_training_algo_1D(train_data=train_dataset,
                                                       train_labels=train_labels,
                                                       smc_transformer=smc_transformer,
                                                       num_particles=num_particles,
                                                       sigma=sigma,
                                                       omega_init=omega_init,
                                                       num_iter=num_iter)


      logger.info('learned std: {}'.format(list_std_k))
      logger.info('learned sigma_obs:{}'.format(list_sigma_obs))
      logger.info('<-------------------------------------------------------------------------------------------------------------------------------------------->')
