#TODO: implement the KL divergence.
#TODO: implement a 'learned' omega (cf discussion with Florian on Telegram.) > before that, check that new 'training' code works for one single feature.
# > OK, omega learned not easy to implement because used in the computation of w.
#TODO: separate scripts for MC-Dropout and inference on SMC Transformer.

# https://stackoverflow.com/questions/3220284/how-to-customize-the-time-format-for-python-logging

import tensorflow as tf
import numpy as np

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.Baselines.Transformer_without_enc import Transformer
from models.Baselines.LSTMs import build_LSTM_for_regression
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask

from utils.utils_train import create_run_dir
from utils.utils_train import create_logger
from utils.utils_train import restoring_checkpoint
from utils.utils_train import saving_inference_results
from train.loss_functions import CustomSchedule
from eval.inference_functions import inference_function_multistep_1D
from eval.inference_functions import inference_function_multistep
from eval.inference_functions import generate_empirical_distribution_1D
from eval.inference_functions import generate_empirical_distribution
from eval.inference_functions import inference_Baseline_T_MC_Dropout_1D
from eval.inference_functions import inference_LSTM_MC_Dropout_1D
import statistics
#import tensorflow_probability as tfp
import ot
from utils.KL_divergences_estimators import naive_estimator

import argparse
import json
import os

if __name__ == "__main__":

  #---- parsing arguments --------------------------------------------------------------
  #results_ws155_632020
  parser = argparse.ArgumentParser()
  results_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/post_UAI_exp/results_ws155_632020'
  exp_path = 'time_series_multi_synthetic_heads_2_depth_6_dff_24_pos-enc_50_pdrop_0_b_256_target-feat_0_cv_False__particles_1_noise_False_sigma_0.05'
  default_out_folder = os.path.join(results_path, exp_path)
  default_data_folder = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data/test_data_synthetic_3_feat.npy'
  default_Baseline_T_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/post_UAI_exp/time_series_multi_synthetic_heads_2_depth_6_dff_24_pos-enc_50_pdrop_0_b_256_target-feat_0_cv_False__rnn-units_10'

  parser.add_argument("-out_folder", default=default_out_folder, type=str, help="path for the output folder with training result")
  parser.add_argument("-baseline_T_path", default=default_Baseline_T_path, type=str,
                      help="path for the output folder with training results for the Baseline Transformer")
  parser.add_argument("-data_path", default=default_data_folder, type=str, help="path for the test data folder")
  parser.add_argument("-num_timesteps", default=4, type=int, help="number of timesteps for doing inference")
  parser.add_argument("-p_inf", default=[5,10,25], type=list, help="number of particles generated for inference")
  parser.add_argument("-N", default=10, type=int, help="number of samples for MC sampling")
  parser.add_argument("-N_est", default=500, type=int, help="number of samples for the empirical distributions")
  parser.add_argument("-sigma", default=0.05, type=float, help="value of the internal noise")
  parser.add_argument("-omega", default=0.1, type=float, help="value of the external covariance of the gaussian noise")
  parser.add_argument("-dropout_rate", default=0.1, type=float, help="dropout rate for MC Dropout algo.")
  parser.add_argument("-layer_norm", default=True, type=bool, help="layer norm or no layerm in Transformer model.")

  args=parser.parse_args()
  output_path = args.out_folder

  config_path = os.path.join(output_path, 'config.json')
  test_data_path = args.data_path

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
  resampling = False if num_particles == 1 else True

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
  test_dataset = np.load(test_data_path)
  seq_len = test_dataset.shape[1] - 1
  num_features = test_dataset.shape[-1]
  # convert it into a tf.tensor
  test_dataset = tf.convert_to_tensor(test_dataset, dtype=tf.float32)

  # ---------------preparing the output path for inference -------------------------------------------------------------------------------------------------------------
  num_timesteps = args.num_timesteps
  N = args.N
  sigma = args.sigma
  list_p_inf = args.p_inf
  N_est = args.N_est
  omega = args.omega
  dropout_rate = args.dropout_rate
  layer_norm = args.layer_norm

  k = 0.005
  alpha = 1
  num_updates = 50

  test_one_sample = False
  size_test_dataset = tf.shape(test_dataset)[0]
  index = np.random.randint(low=0, high=size_test_dataset)
  index = 3946

  output_path = args.out_folder
  checkpoint_path = os.path.join(output_path, "checkpoints")

  if not os.path.isdir(os.path.join(output_path, 'inference_results')):
    output_path = create_run_dir(path_dir=output_path, path_name='inference_results')
  else:
    output_path = os.path.join(output_path, 'inference_results')
  folder_template = 'num-timesteps_{}_N_{}_N-est_{}_sigma_{}_omega_0.76'
  out_folder = folder_template.format(num_timesteps, N, N_est, sigma)
  if test_one_sample:
    out_folder = out_folder + '_sample_{}'.format(index)

  if not os.path.isdir(os.path.join(output_path, out_folder)):
    output_path = create_run_dir(path_dir=output_path, path_name=out_folder)
  else:
    output_path=os.path.join(output_path, out_folder)


  # -------------- create the logging -----------------------------------------------------------------------------------------------------------------------------------
  out_file_log = output_path + '/' + 'inference_log.log'
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
                                    resampling=True,
                                    layer_norm=layer_norm,
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
  # ------------------------------------- compute inference timesteps --------------------------------------------------------------------------------------------------
  cov_matrix_3D = tf.constant([0.2, 0.3, 0.4], dtype=tf.float32)
  A_3D = tf.constant([[0.8, 0.1, 0], [0.2, 0.9, 0.2], [0, 0.1, 0.85]], dtype=tf.float32)
  list_KL_exp = []

  omega_inf = 0.76

  test_one_sample = False
  if test_one_sample:
    test_dataset = test_dataset[index,:,:]
    test_dataset = tf.expand_dims(test_dataset, axis=0)

  inference_smc_T = True
  if inference_smc_T:
    if test_one_sample:
      logger.info("results for sample: {}".format(index))

    logger.info('initial std...: {}'.format(omega))
    # logger.info('hyper-parameters for the stochastic approx. algo: alpha:{} - k:{} -num_updates: {}'.format(alpha, k,
    #                                                                                                         num_updates))
    if not layer_norm:
      logger.info("inference without layer norm...")
    # for p_inf in list_p_inf:
    #   logger.info('inference results for number of particles: {}'.format(p_inf))
    #   # re-initializing omega with initial scalar value
    #   #smc_transformer.omega = omega
    #   #smc_transformer.cell.omega = omega
    #
    #   if target_feature is None:
    #
    #     (list_mean_NP, list_X_pred_NP), list_preds_sampled, w_s, (learned_std, covariance_matrix) = inference_function_multistep(
    #       inputs=test_dataset,
    #       smc_transformer=smc_transformer,
    #       N_prop=N,
    #       N_est=N_est,
    #       num_particles=p_inf,
    #       num_timesteps=num_timesteps,
    #       sample_pred=True,
    #       sigma=sigma,
    #       output_path=output_path,
    #       layer_norm=layer_norm)
    #     logger.info('learned std: {}'.format(learned_std))
    #     logger.info('covariance matrix: {}'.format(covariance_matrix))
    #
    #
    #   else:
    #
    #     (list_mean_NP, list_X_pred_NP), list_preds_sampled, w_s = inference_function_multistep_1D(inputs=test_dataset,
    #                                                                                             smc_transformer=smc_transformer,
    #                                                                                             N_prop=N,
    #                                                                                             N_est=N_est,
    #                                                                                             num_particles=p_inf,
    #                                                                                             num_timesteps=num_timesteps,
    #                                                                                             sample_pred=True,
    #                                                                                             sigma=sigma,
    #                                                                                             omega=omega_inf,
    #                                                                                             output_path=output_path,
    #                                                                                             layer_norm=layer_norm)


        #logger.info('learned std: {}'.format(list_learned_std))
        #logger.info('<----------------------------------------------------------------------------------------------------------------------------->')

    list_empirical_dist, list_true_means = generate_empirical_distribution_1D(inputs=test_dataset,
                                                                                matrix_A=A_3D,
                                                                                cov_matrix=cov_matrix_3D,
                                                                                N_est=N_est,
                                                                                num_timesteps=num_timesteps,
                                                                                output_path=output_path)
    #list_empirical_dist, list_true_means = generate_empirical_distribution(inputs=test_dataset,
                                                                             # matrix_A=A_3D,
                                                                             # cov_matrix=cov_matrix_3D,
                                                                             # N_est=N_est,
                                                                             # num_timesteps=num_timesteps,
                                                                             # output_path=output_path)

    # --------------------------- compute distances ------------------------------------------------------------------------------------------------------------------
      #KL_measure = tf.keras.losses.KLDivergence()
      # KL_distance = KL_measure(y_true=true_distrib, y_pred=pred_distrib)
      # KL_distance_norm = KL_distance / N_est
      # KL_timesteps.append(KL_distance_norm.numpy())
      # KL_dist = scipy.stats.entropy(pk=pred_distrib, qk=pred_distrib)

      # KL_timesteps = []
      # for t, (true_distrib, pred_distrib) in enumerate(zip(list_empirical_dist, list_preds_sampled)):
      #   batch_size = pred_distrib.shape[0]
      #   num_samples = pred_distrib.shape[1]
      #   # distributions distance and variance of the predicted distribution.
      #   wassertein_dist_list = [ot.emd2_1d(x_a=true_distrib[i,:], x_b=pred_distrib[i,:]) for i in range(batch_size)]
      #   wassertein_dist = statistics.mean(wassertein_dist_list)
      #   #KL_distance_list = [naive_estimator(true_distrib[i,:].reshape(num_samples,1), pred_distrib[i,:].reshape(num_samples,1)) for i in range(batch_size)]
      #   #KL_dist = statistics.mean(KL_distance_list)
      #   std_pred_distrib = np.std(pred_distrib, axis=1)
      #   std_pred_distrib = np.mean(std_pred_distrib, axis=0)
      #   #logger.info('KL distance for timestep {}: {}'.format(t, KL_dist))
      #   logger.info('standard deviation of the predictive distribution: {}'.format(std_pred_distrib))
      #   logger.info('wassertein distance for timestep {}: {}'.format(t, wassertein_dist))
      #
      # #list_KL_exp.append(KL_timesteps)
      # logger.info("<--------------------------------------------------------------------------------------------------------------------------------------------------------->")

  # --------------------------- inference function for MC Dropout Baseline Transformer -----------------------------------------------------------------------------
  # get the hparams:
  inference_mc_dropout = False

  if inference_mc_dropout:
    Baseline_T_path = args.baseline_T_path
    config_T_path = os.path.join(Baseline_T_path, 'config.json')
    checkpoint_T_path = os.path.join(Baseline_T_path, "checkpoints")
    with open(config_T_path) as f:
      hparams_T = json.load(f)

    # model params
    num_layers = hparams_T["model"]["num_layers"]
    num_heads = hparams_T["model"]["num_heads"]
    d_model = hparams_T["model"]["d_model"]
    dff = hparams_T["model"]["dff"]
    rate = hparams_T["model"]["rate"]  # p_dropout
    max_pos_enc_bas_str = hparams_T["model"]["maximum_position_encoding_baseline"]
    maximum_position_encoding_baseline = None if max_pos_enc_bas_str == "None" else max_pos_enc_bas_str
    # task params
    data_type = hparams_T["task"]["data_type"]
    task_type = hparams_T["task"]["task_type"]
    task = hparams_T["task"]["task"]

    target_vocab_size = 1

    # define optimizer
    if custom_schedule == "True":
      learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

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
                              rate=dropout_rate)

    # restore 2 transformers
    baseline_T_ckpt_path = os.path.join(checkpoint_T_path, "transformer_baseline_1")
    baseline_T_ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    baseline_T_dropout_ckpt = tf.train.Checkpoint(transformer=transformer_w_dropout, optimizer=optimizer)
    baseline_T_ckpt_manager = tf.train.CheckpointManager(baseline_T_ckpt, baseline_T_ckpt_path, max_to_keep=EPOCHS)
    baseline_T_dropout_ckpt_manager = tf.train.CheckpointManager(baseline_T_dropout_ckpt, baseline_T_ckpt_path, max_to_keep=EPOCHS)
    _ = restoring_checkpoint(ckpt_manager=baseline_T_ckpt_manager,
                             ckpt=baseline_T_ckpt,
                             args_load_ckpt=True,
                             logger=logger)
    _ = restoring_checkpoint(ckpt_manager=baseline_T_dropout_ckpt_manager,
                             ckpt=baseline_T_dropout_ckpt,
                             args_load_ckpt=True,
                             logger=logger)
    test_dataset_T = test_dataset[:,:-1,:] # (5000, 24, 3)
    Y_test = test_dataset[:, 1:, 0]  # (5000,24)
    seq_len_T = tf.shape(Y_test)[-1]
    # checking Transformer loss on the test dataset.
    Y_preds, _ = transformer(inputs=test_dataset_T,
                          training=False,
                          mask=create_look_ahead_mask(seq_len_T))  # (5000,24,1)
    Y_preds = tf.squeeze(Y_preds)
    test_loss = tf.keras.losses.MSE(Y_test, Y_preds)
    test_loss = tf.reduce_mean(test_loss, axis=-1)
    print('Transformer loss', test_loss.numpy())

    logger.info("starting MC Dropout inference on a trained Baseline Transformer...")
    MC_Dropout_predictions, list_true_preds = inference_Baseline_T_MC_Dropout_1D(inputs=test_dataset_T,
                                                                transformer=transformer,
                                                                transformer_w_dropout=transformer_w_dropout,
                                                                num_mc_samples=N_est,
                                                                num_timesteps=num_timesteps,
                                                                output_path=Baseline_T_path)

    true_labels_inf = test_dataset[:, 21:, 0]  # (B,4)
    Transf_preds_inf = tf.stack(list_true_preds, axis=1)  # (B,4,1)
    Transf_preds_inf = tf.squeeze(Transf_preds_inf)

    loss_inference = tf.keras.losses.MSE(true_labels_inf, Transf_preds_inf)
    loss_inference = tf.reduce_mean(loss_inference, axis=-1)
    # loss_inference = tf.reduce_mean(loss_inference, axis=-1)
    loss_inference = loss_inference.numpy()
    print('loss inference Transformer', loss_inference) #TODO: compute the loss at one timestep.

    # ------------- MC Dropout on LSTM ------------------------------------------------------------------------------------------
    LSTM_path = args.baseline_T_path
    config_LSTM_path = os.path.join(LSTM_path, 'config.json')
    checkpoint_LSTM_path = os.path.join(LSTM_path, "checkpoints")
    with open(config_LSTM_path) as f:
      hparams_LSTM = json.load(f)

    # optim params
    learning_rate = hparams_LSTM["optim"]["learning_rate"]
    EPOCHS = hparams_LSTM["optim"]["EPOCHS"]
    # adding RNN hyper-parameters
    rnn_units = hparams_LSTM["RNN_hparams"]["rnn_units"]
    rnn_dropout_rate = hparams_LSTM["RNN_hparams"]["rnn_dropout_rate"]
    # data params
    target_feature = hparams["data"]["target_feature"]
    if target_feature == "None":
      target_feature = None
    target_vocab_size = 1 if target_feature is not None else num_features

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    lstm = build_LSTM_for_regression(shape_input_1=seq_len, shape_input_2=num_features, shape_output=target_vocab_size,
                                     rnn_units=rnn_units, dropout_rate=rnn_dropout_rate)
    lstm_w_dropout = build_LSTM_for_regression(shape_input_1=seq_len, shape_input_2=num_features,
                                               shape_output=target_vocab_size,
                                               rnn_units=20, dropout_rate=0.1)
    # restore 2 LSTM
    LSTM_ckpt_path = os.path.join(checkpoint_LSTM_path, "RNN_baseline_1")
    LSTM_ckpt = tf.train.Checkpoint(model=lstm, optimizer=optimizer)
    LSTM_dropout_ckpt = tf.train.Checkpoint(model=lstm_w_dropout, optimizer=optimizer)
    LSTM_ckpt_manager = tf.train.CheckpointManager(LSTM_ckpt, LSTM_ckpt_path, max_to_keep=EPOCHS)
    LSTM_dropout_ckpt_manager = tf.train.CheckpointManager(LSTM_dropout_ckpt, LSTM_ckpt_path, max_to_keep=EPOCHS)
    _ = restoring_checkpoint(ckpt_manager=LSTM_ckpt_manager,
                             ckpt=LSTM_ckpt,
                             args_load_ckpt=True,
                             logger=logger)
    _ = restoring_checkpoint(ckpt_manager=LSTM_dropout_ckpt_manager,
                             ckpt=LSTM_dropout_ckpt,
                             args_load_ckpt=True,
                             logger=logger)
    test_dataset_LSTM = test_dataset[:, :-1, :]  # (5000, 24, 3)
    Y_test = test_dataset[:,1:,0] # (5000,24)

    # checking LSTM loss on the test dataset.
    #Y_preds = lstm.predict(test_dataset_LSTM)
    Y_preds = lstm(test_dataset_LSTM) # (5000,24,1)
    Y_preds = tf.squeeze(Y_preds)
    test_loss = tf.keras.losses.MSE(Y_test, Y_preds)
    test_loss = tf.reduce_mean(test_loss, axis=-1)
    # test_loss_2 = tf.square(Y_test - Y_preds)
    # test_loss_2 = tf.reduce_mean(test_loss_2, axis=-1)
    # test_loss_2 = tf.reduce_mean(test_loss_2, axis=-1)
    print('test loss', test_loss.numpy())
    #print('test loss 2', test_loss_2.numpy())


    logger.info("starting MC Dropout inference on a trained LSTM...")
    LSTM_MC_Dropout_predictions, list_LSTM_preds = inference_LSTM_MC_Dropout_1D(inputs=test_dataset_LSTM,
                                                                                lstm_model=lstm,
                                                                                lstm_w_dropout=lstm_w_dropout,
                                                                                num_mc_samples=N_est,
                                                                                num_timesteps=num_timesteps,
                                                                                output_path=LSTM_path)
    true_labels_inf = test_dataset[:,21:,0] # (B,4)
    LSTM_preds_inf = tf.stack(list_LSTM_preds, axis=1) # (B,4,1)
    LSTM_preds_inf = tf.squeeze(LSTM_preds_inf)

    loss_inference = tf.keras.losses.MSE(true_labels_inf, LSTM_preds_inf)
    loss_inference = tf.reduce_mean(loss_inference, axis=-1)
    #loss_inference = tf.reduce_mean(loss_inference, axis=-1)
    loss_inference = loss_inference.numpy()
    print('loss inference LSTM', loss_inference)

  # ------------------------------------------------------- saving results on a csv file ---------------------------------------------------------------

  # keys = ['N_est = {}'.format(n) for n in list_num_samples]
  # keys = ['inference_timestep'] + keys
  # values = ['t+{}'.format(i+1) for i in range(num_timesteps)]
  # values = [values] + list_KL_exp
  # csv_fname='inference_results_KL_norm.csv'
  #
  # saving_inference_results(keys=keys,
  #                           values=values,
  #                           output_path=output_path,
  #                           csv_fname=csv_fname)
