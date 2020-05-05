#TODO - to improve the transformer performance - label smoothing and change order of layers.dense / layers.norm.

"""# to store:
# in a fichier .log: for each epoch, the average loss (train & val dataset),
the training accuracy (train & val datasets),
- 2 accuracies for the SMC Transformer), the time taken for each epoch, and the total training time
# - in a fichier .txt: the list of losses (train & val), accuracies (train & val) for plotting & comparing.
# - in files .npy: the output of the model (predictions, trajectories, attention_weights...).
# - checkpoints of the model in a file .ckpt.
# dictionary of hparams (cf Nicolas's script...).
"""
"""Transformer model parameters (from the tensorflow tutorial):
d_model: 512
num_heads: 8
dff: 1048
num_layers: 2
max_positional_encoding = target_vocab_size. 
"""
"""
ALL INPUTS CAN BE OF SHAPE (B,S,F) (F=1 for NLP / univariate time_series case, F > 1 for multivariate case.)
"""

import tensorflow as tf

from train.loss_functions import CustomSchedule
from train.train_functions import train_baseline_transformer
from train.train_functions import train_SMC_transformer
from train.train_functions import train_LSTM

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.Baselines.LSTMs import build_LSTM_for_regression

import sys
import os
import numpy as np
import shutil
import json
import argparse
import warnings

from preprocessing.time_series.df_to_dataset import df_to_data_regression
from preprocessing.time_series.df_to_dataset import data_to_dataset_4D
from preprocessing.time_series.df_to_dataset import split_input_target
from preprocessing.time_series.df_to_dataset import split_synthetic_dataset

from utils.utils_train import create_run_dir
from utils.utils_train import create_logger
from utils.utils_train import restoring_checkpoint

from eval.evaluation_functions import compute_latest_statistics
from eval.evaluation_functions import evaluate_SMC_Transformer
from eval.Transformer_dropout import MC_Dropout_Transformer


if __name__ == "__main__":

  warnings.simplefilter("ignore")

  # -------- parsing arguments ----------------------------------------------------------------------------------------------------------------------------------------------------

  out_folder_for_args ='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/exp_162_grad_not_zero_azure/'
  config_path_after_training = out_folder_for_args + 'time_series_multi_unistep-forcst_heads_1_depth_3_dff_12_pos-enc_50_pdrop_0.1_b_1048_cs_True__particles_25_noise_True_sigma_0.1_smc-pos-enc_None/config_nlp.json'
  config_after_training_synthetic = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/azure_synthetic_dataset_exp/time_series_multi_synthetic_heads_1_depth_2_dff_8_pos-enc_None_pdrop_0.1_b_256_cs_True__particles_10_noise_True_sigma_0.05_smc-pos-enc_None/config_nlp.json'
  out_folder_after_training_synthetic = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/azure_synthetic_dataset_exp/'

  out_folder_default = '../../output/post_UAI_exp'
  config_folder_default = '../../config/config_ts_reg_multi_synthetic_3feat.json'

  parser = argparse.ArgumentParser()

  parser.add_argument("-config", type=str, default=config_folder_default, help="path for the config file with hyperparameters")
  parser.add_argument("-out_folder", type=str, default=out_folder_default, help="path for the outputs folder")
  parser.add_argument("-data_folder", type=str, default='../../data/synthetic_dataset.npy', help="path for the data folder")

  #TODO: ask Florian why when removing default value, it is not working...
  parser.add_argument("-train_baseline", type=bool, default=False, help="Training a Baseline Transformer?")
  parser.add_argument("-train_smc_T", type=bool, default=True, help="Training the SMC Transformer?")
  parser.add_argument("-train_rnn", type=bool, default=False, help="Training a Baseline RNN?")
  parser.add_argument("-skip_training", type=bool, default=False, help="skip training and directly evaluate?")
  parser.add_argument("-eval", type=bool, default=True, help="evaluate after training?")

  parser.add_argument("-load_ckpt", type=bool, default=True, help="loading and restoring existing checkpoints?")
  args = parser.parse_args()
  config_path = args.config

  #------------------Uploading the hyperparameters info----------------------------------------------------------------------------------------------------------------------------

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
  layer_norm = hparams["model"]["layer_norm"]
  layer_norm = True if layer_norm == "True" else False
  mc_dropout_samples = hparams["model"]["mc_dropout_samples"]

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
  omega = 1
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

  test_loss = False

  #------------------ UPLOAD the training dataset ----------------------------------------------------------------------------------------------------------------------

  if task =='unistep-forcst':
    (train_data, val_data, test_data), original_df, stats = df_to_data_regression(file_path=file_path,
                                                                                  fname=fname,
                                                                                  col_name=col_name,
                                                                                  index_name=index_name,
                                                                                  TRAIN_SPLIT=TRAIN_SPLIT,
                                                                                  history=history,
                                                                                  step=step,
                                                                                  cv=cv)

    BUFFER_SIZE = 10000

  elif task == 'synthetic':
    X_data = np.load(file_path)
    stats = None
    train_data, val_data, test_data = split_synthetic_dataset(x_data=X_data,
                                                              TRAIN_SPLIT=TRAIN_SPLIT,
                                                              VAL_SPLIT=VAL_SPLIT,
                                                              VAL_SPLIT_cv=VAL_SPLIT_cv,
                                                              cv=cv)
    #val_data_path = 'data/val_data_synthetic_3_feat.npy'
    #train_data_path = 'data/train_data_synthetic_3_feat.npy'
    #test_data_path = 'data/test_data_synthetic_3_feat.npy'
    val_data_path = '../../data/val_data_synthetic_3_feat.npy'
    train_data_path = '../../data/train_data_synthetic_3_feat.npy'
    test_data_path = '../../data/test_data_synthetic_3_feat.npy'
    np.save(val_data_path, val_data)
    np.save(train_data_path, train_data)
    np.save(test_data_path, test_data)

    BUFFER_SIZE = 2000
  if not cv:
    print('train_data', train_data.shape)
    print('val_data', val_data.shape)
  else:
    print('train data sample', train_data[0].shape)
    print('val data sample', val_data[0].shape)

  if not cv:
    train_dataset, val_dataset, test_dataset, train_dataset_for_RNN, val_dataset_for_RNN, test_dataset_for_RNN = data_to_dataset_4D(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    split_fn=split_input_target,
    BUFFER_SIZE=BUFFER_SIZE,
    BATCH_SIZE=BATCH_SIZE,
    target_feature=target_feature,
    cv=cv)

    for (inp, tar) in train_dataset.take(1):
      print('input example', inp[0])
      print('target example', tar[0])

  else:
    list_train_dataset, list_val_dataset, test_dataset, list_train_dataset_for_RNN, list_val_dataset_for_RNN, test_dataset_for_RNN = data_to_dataset_4D(
      train_data=train_data,
      val_data=val_data,
      test_data=test_data,
      split_fn=split_input_target,
      BUFFER_SIZE=BUFFER_SIZE,
      BATCH_SIZE=BATCH_SIZE,
      target_feature=target_feature,
      cv=cv)

  if not cv:
    num_features = train_data.shape[-1]
    target_vocab_size = num_features if target_feature is None else 1
    seq_len = train_data.shape[1] - 1  # 24 observations
    training_samples = train_data.shape[0]
    steps_per_epochs = int(train_data.shape[0] / BATCH_SIZE)
  else:
    num_features = train_data[0].shape[-1]
    target_vocab_size = num_features if target_feature is None else 1
    seq_len = train_data[0].shape[1]-1
    training_samples = train_data[0].shape[0]
    steps_per_epochs = int(train_data[0].shape[0] / BATCH_SIZE)

  print("steps per epochs", steps_per_epochs)

  # -------define hyperparameters----------------------------------------------------------------------------------------------------------------
  optimizer_LSTM = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)
  if custom_schedule == "True":
    print("learning rate with custom schedule...")
    learning_rate = CustomSchedule(d_model)

  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

  # ------------- preparing the OUTPUT FOLDER--------------------------------------------------------------------------------------------

  output_path = args.out_folder
  folder_template='{}_{}_heads_{}_depth_{}_dff_{}_pos-enc_{}_pdrop_{}_b_{}_target-feat_{}_cv_{}'
  out_folder = folder_template.format(data_type,
                                      task,
                                      num_heads,
                                      d_model,
                                      dff,
                                      maximum_position_encoding_baseline,
                                      rate,
                                      BATCH_SIZE,
                                      target_feature,
                                      cv)

  if args.train_smc_T:
    out_folder = out_folder + '__particles_{}_noise_{}_sigma_{}'.format(num_particles,
                                                                        noise_SMC_layer,
                                                                        sigma)

  if args.train_rnn:
    out_folder = out_folder + '__rnn-units_{}'.format(rnn_units)

  output_path = create_run_dir(path_dir=output_path, path_name=out_folder)

  # copying the config file in the output directory
  if not os.path.exists(output_path+'/config.json'):
    shutil.copyfile(config_path, output_path+'/config.json')

  # ------------------ create the logging ------------------------------------------------------------------------------------------------

  out_file_log = output_path + '/' + 'training_log.log'
  logger = create_logger(out_file_log=out_file_log)
  #  creating the checkpoint manager:
  checkpoint_path = os.path.join(output_path, "checkpoints")
  if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

  #-------------------- Building the RNN Baseline & the associated training algo --------------------------------------------------------------------------
  model = build_LSTM_for_regression(shape_input_1=seq_len,
                                    shape_input_2=num_features,
                                    shape_output=target_vocab_size,
                                    rnn_units=rnn_units,
                                    dropout_rate=rnn_dropout_rate,
                                    training=True)

  #---------------------- TRAINING OF A SIMPLE RNN BASELINE -------------------------------------------------------------------------------------------------------
  if args.skip_training:
    logger.info("skipping training...")
  else:
    if args.train_rnn:
      if not cv:
        for (inp, _) in train_dataset_for_RNN.take(1):
          pred_temp = model(inp)
        print('LSTM summary', model.summary())

        train_LSTM(model=model,
                   optimizer=optimizer_LSTM,
                   EPOCHS=EPOCHS,
                   train_dataset=train_dataset_for_RNN,
                   val_dataset=val_dataset_for_RNN,
                   checkpoint_path=checkpoint_path,
                   args=args,
                   output_path=output_path,
                   logger=logger,
                   num_train=1)
      else:
        for train_num, (train_dataset_for_RNN, val_dataset_for_RNN) in enumerate(zip(list_train_dataset_for_RNN, list_val_dataset_for_RNN)):
          if train_num == 0:
            for (inp, _) in train_dataset_for_RNN.take(1):
              pred_temp = model(inp)
            print('LSTM summary', model.summary())

          logger.info("starting training of train/val split number {}".format(train_num))
          train_LSTM(model=model,
                     optimizer=optimizer,
                     EPOCHS=EPOCHS,
                     train_dataset=train_dataset_for_RNN,
                     val_dataset=val_dataset_for_RNN,
                     checkpoint_path=checkpoint_path,
                     args=args,
                     output_path=output_path,
                     logger=logger,
                     num_train=train_num+1)
          logger.info("training of a LSTM for train/val split number {} done...".format(train_num + 1))
          logger.info(
          "<---------------------------------------------------------------------------------------------------------------------------------------------------------->")

    #----------------- TRAINING -----------------------------------------------------------------------------------------------------------------------------------------------------------

    logger.info("model hyperparameters from the config file: {}".format(hparams["model"]))
    logger.info("smc hyperparameters from the config file: {}".format(hparams["smc"]))
    if args.train_rnn:
      logger.info("GRU model hparams from the config file: {}".format(hparams["RNN_hparams"]))

    # TF-2.0
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    sys.setrecursionlimit(100000)
    tf.config.experimental_run_functions_eagerly(True)  # to remove TensorInacessibleError

    # to choose which models to train.
    train_smc_transformer = args.train_smc_T
    train_classic_transformer = args.train_baseline

    # -------------------------------------------TRAIN ON THE DATASET - CLASSIC TRANSFORMER -------------------------------------------

    if train_classic_transformer:
      logger.info("training the baseline Transformer on a time-series dataset...")
      logger.info("number of training samples: {}".format(training_samples))
      logger.info("steps per epoch:{}".format(steps_per_epochs))

      if not cv:
        train_baseline_transformer(hparams=hparams,
                                 optimizer=optimizer,
                                 seq_len=seq_len,
                                 target_vocab_size=target_vocab_size,
                                 train_dataset=train_dataset,
                                 val_dataset=val_dataset,
                                 train_accuracy=train_accuracy,
                                 val_accuracy=val_accuracy,
                                 output_path=output_path,
                                 checkpoint_path=checkpoint_path,
                                 args=args,
                                 logger=logger,
                                 num_train=1)

      else:
        for train_num, (train_dataset, val_dataset) in enumerate(zip(list_train_dataset, list_val_dataset)):
          train_baseline_transformer(hparams=hparams,
                                     optimizer=optimizer,
                                     seq_len=seq_len,
                                     target_vocab_size=target_vocab_size,
                                     train_dataset=train_dataset,
                                     val_dataset=val_dataset,
                                     train_accuracy=train_accuracy,
                                     val_accuracy=val_accuracy,
                                     output_path=output_path,
                                     checkpoint_path=checkpoint_path,
                                     args=args,
                                     logger=logger,
                                     num_train=train_num+1)
          logger.info("training of a Baseline Transformer for train/val split number {} done...".format(train_num + 1))
          logger.info(
            "<---------------------------------------------------------------------------------------------------------------------------------------------------------->")


    #------------------- TRAINING ON THE DATASET - SMC_TRANSFORMER ----------------------------------------------------------------------------------------------------------------------

    if train_smc_transformer:
      logger.info('starting the training of the smc transformer...')
      logger.info("number of training samples: {}".format(training_samples))
      logger.info("steps per epoch: {}".format(steps_per_epochs))

      if not resampling:
        logger.info("no resampling because only one particle is taken...")

      if not cv:
        train_SMC_transformer(hparams=hparams,
                            optimizer=optimizer,
                            seq_len=seq_len,
                            target_vocab_size=target_vocab_size,
                            resampling=resampling,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            train_accuracy=train_accuracy,
                            output_path=output_path,
                            checkpoint_path=checkpoint_path,
                            args=args,
                            logger=logger,
                            num_train=1)
      else:
        for train_num, (train_dataset, val_dataset) in enumerate(zip(list_train_dataset, list_val_dataset)):
          train_SMC_transformer(hparams=hparams,
                                optimizer=optimizer,
                                seq_len=seq_len,
                                target_vocab_size=target_vocab_size,
                                resampling=resampling,
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                train_accuracy=train_accuracy,
                                output_path=output_path,
                                checkpoint_path=checkpoint_path,
                                args=args,
                                logger=logger,
                                num_train=train_num+1)
          logger.info("training of a SMC Transformer for train/val split number {} done...".format(train_num + 1))
          logger.info(
            "<---------------------------------------------------------------------------------------------------------------------------------------------------------->")

  # ----------------------------------- EVALUATION -------------------------------------------------------------------------------------------------------------------------------
  if args.eval:
    logger.info("start evaluation for SMC Transformer...")
    #TODO: add evaluation for LSTM & Baseline Transformer.
    # restoring latest checkpoint for SMC Transformer
    smc_transformer = SMC_Transformer(d_model=d_model, output_size=target_vocab_size, seq_len=seq_len, full_model=False,
                                      dff=)

    # creating checkpoint manager
    smc_T_ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                                     optimizer=optimizer)
    #smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer")
    smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer_1")
    smc_T_ckpt_manager = tf.train.CheckpointManager(smc_T_ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS)

    # if a checkpoint exists, restore the latest checkpoint.
    num_epochs_smc_T = restoring_checkpoint(ckpt_manager=smc_T_ckpt_manager, ckpt=smc_T_ckpt, args_load_ckpt=args.load_ckpt, logger=logger)

    # # restoring latest checkpoint from Baseline Transformer
    # if args.train_baseline:
    #   transformer = Transformer(num_layers=num_layers,
    #                             d_model=d_model,
    #                             num_heads=num_heads,
    #                             dff=dff,
    #                             target_vocab_size=target_vocab_size,
    #                             maximum_position_encoding=maximum_position_encoding_baseline,
    #                             data_type=data_type,
    #                             rate=rate)
    #
    #   baseline_T_ckpt_path = os.path.join(checkpoint_path, "transformer_baseline_1")
    #   baseline_T_ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    #   baseline_T_ckpt_manager = tf.train.CheckpointManager(baseline_T_ckpt, baseline_T_ckpt_path, max_to_keep=EPOCHS)
    #   num_epochs_baseline_T = restoring_checkpoint(ckpt_manager=baseline_T_ckpt_manager, ckpt=baseline_T_ckpt, args_load_ckpt=args.load_ckpt, logger=logger)
    #
    #   logger.info("starting evaluation of MC Dropout on the Baseline Transformer on the test set...")
    #   MC_Dropout_Transformer(transformer=transformer,
    #                          test_dataset=test_dataset,
    #                          seq_len=seq_len,
    #                          task=task,
    #                          stats=stats,
    #                          num_samples=mc_dropout_samples,
    #                          output_path=output_path,
    #                          logger=logger)

    # --------------------------------------------- compute latest statistics ---------------------------------------------------------------------------------------
    if args.train_smc_T:
      logger.info("<------------------------computing latest statistics on SMC Transformer----------------------------------------------------------------------------------------->")
      compute_latest_statistics(smc_transformer=smc_transformer,
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                seq_len=seq_len,
                                output_path=output_path,
                                logger=logger)

      # -----unistep evaluation with N = 1 ------------------------------------------------------------------------------------------------------------------------------#

      logger.info("starting evaluation of the SMC Transformer on the test set...")
      if task == 'synthetic':
        stats = None
      evaluate_SMC_Transformer(smc_transformer=smc_transformer,
                               test_dataset=test_dataset,
                               seq_len=seq_len,
                               task=task,
                               stats=stats,
                               output_path=output_path,
                               logger=logger)



