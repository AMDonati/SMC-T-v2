import tensorflow as tf
from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.train_step_functions import train_step_classic_T
from train.train_step_functions import train_step_SMC_T
from train.loss_functions import loss_function_regression

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

import time
import os
import statistics

from utils.utils_train import saving_training_history
from utils.utils_train import saving_model_outputs
from utils.utils_train import restoring_checkpoint

def train_LSTM(model, optimizer, EPOCHS, train_dataset_for_RNN, val_dataset_for_RNN, output_path, checkpoint_path, args, logger, num_train):
  #TODO: remove this function.
  LSTM_ckpt_path = os.path.join(checkpoint_path, "RNN_Baseline_{}".format(num_train))
  LSTM_ckpt_path = LSTM_ckpt_path+'/'+'LSTM-{epoch}'

  start_epoch = 0
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
      filepath=LSTM_ckpt_path,
      monitor='val_loss',
      save_best_only=True,
      save_weights_only=True,
      verbose=0)
  ]
  model.compile(optimizer=optimizer,
                loss='mse')

  # --- starting the training ... -----------------------------------------------
  start_training = time.time()
  rnn_history = model.fit(train_dataset_for_RNN,
                          epochs=EPOCHS,
                          validation_data=val_dataset_for_RNN,
                          callbacks=callbacks,
                          verbose=2)

  train_loss_history_rnn = rnn_history.history['loss']
  val_loss_history_rnn = rnn_history.history['val_loss']
  keys = ['train_loss', 'val_loss']
  values = [train_loss_history_rnn, val_loss_history_rnn]
  csv_fname = 'rnn_history_{}.csv'.format(num_train)

  saving_training_history(keys=keys,
                          values=values,
                          output_path=output_path,
                          csv_fname=csv_fname,
                          logger=logger,
                          start_epoch=start_epoch)

  logger.info('Training time for {} epochs: {}'.format(EPOCHS, time.time() - start_training))
  logger.info('training of a RNN Baseline for a timeseries dataset done...')
  logger.info(
    ">>>--------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")


def train_baseline_transformer(transformer, optimizer, EPOCHS, train_dataset, val_dataset, output_path, checkpoint_path, logger, num_train):
  # storing the losses & accuracy in a list for each epoch
  average_losses_baseline, val_losses_baseline = [], []

  # creating checkpoint manager
  ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
  baseline_ckpt_path = os.path.join(checkpoint_path, "transformer_baseline_{}".format(num_train))
  ckpt_manager = tf.train.CheckpointManager(ckpt, baseline_ckpt_path, max_to_keep=EPOCHS)
  # if a checkpoint exists, restore the latest checkpoint.
  start_epoch = restoring_checkpoint(ckpt_manager=ckpt_manager, args_load_ckpt=True, ckpt=ckpt, logger=logger)
  if start_epoch is None:
    start_epoch = 0

  start_training = time.time()

  if start_epoch > 0:
    if start_epoch >= EPOCHS:
      print("adding {} more epochs to existing training".format(EPOCHS))
      start_epoch = 0
    else:
      logger.info("starting training after checkpoint restoring from epoch {}".format(start_epoch))

  for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    logger.info("Epoch {}/{}".format(epoch + 1, EPOCHS))

    sum_train_loss, sum_val_loss = 0, 0

    for (batch, (inp, tar)) in enumerate(train_dataset):
      train_loss_batch = train_step_classic_T(inputs=inp,
                                              targets=tar,
                                              transformer=transformer,
                                              optimizer=optimizer)
      sum_train_loss += train_loss_batch

      if batch == 0 and epoch == 0:
        print('baseline transformer summary', transformer.summary())

    avg_train_loss = sum_train_loss / (batch + 1)

    for batch_val, (inp, tar) in enumerate(val_dataset):
      seq_len = tf.shape(inp)[-2]
      predictions_val, attn_weights_val = transformer(inputs=inp,
                                                      training=False,
                                                      mask=create_look_ahead_mask(seq_len))
      val_loss_batch = tf.keras.losses.MSE(tar, predictions_val)
      val_loss_batch = tf.reduce_mean(val_loss_batch, axis=-1)
      val_loss_batch = tf.reduce_mean(val_loss_batch, axis=-1)
      sum_val_loss += val_loss_batch

    avg_val_loss = sum_val_loss / (batch_val + 1)

    logger.info('train loss: {} - val loss: {}'.format(avg_train_loss.numpy(), avg_val_loss.numpy()))
    average_losses_baseline.append(avg_train_loss.numpy())
    val_losses_baseline.append(avg_val_loss.numpy())

    ckpt_manager.save()
    logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  logger.info('total training time for {} epochs:{}'.format(EPOCHS, time.time() - start_training))

  # storing history of losses and accuracies in a csv file
  keys = ['train loss', 'val loss']
  values = [average_losses_baseline, val_losses_baseline]
  csv_fname = 'baseline_history_{}.csv'.format(num_train)

  saving_training_history(keys=keys,
                          values=values,
                          output_path=output_path,
                          csv_fname=csv_fname,
                          logger=logger,
                          start_epoch=start_epoch)

  logger.info('training of a classic Transformer for a time-series dataset done...')
  logger.info(">>>-------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")


def train_SMC_transformer(hparams, optimizer, seq_len, target_vocab_size, resampling, train_dataset, val_dataset, train_accuracy, output_path, checkpoint_path, args, logger, num_train):
  num_layers = hparams["model"]["num_layers"]
  num_heads = hparams["model"]["num_heads"]
  d_model = hparams["model"]["d_model"]
  dff = hparams["model"]["dff"]
  rate = hparams["model"]["rate"]
  max_pos_enc_smc_str = hparams["model"]["maximum_position_encoding_smc"]
  maximum_position_encoding_smc = None if max_pos_enc_smc_str == "None" else max_pos_enc_smc_str
  layer_norm = hparams["model"]["layer_norm"]
  layer_norm = True if layer_norm == "True" else False
  num_particles = hparams["smc"]["num_particles"]
  noise_encoder_str = hparams["smc"]["noise_encoder"]
  noise_encoder = True if noise_encoder_str == "True" else False
  noise_SMC_layer_str = hparams["smc"]["noise_SMC_layer"]
  noise_SMC_layer = True if noise_SMC_layer_str == "True" else False
  sigma = hparams["smc"]["sigma"]
  omega = hparams["smc"]["omega"]
  EPOCHS = hparams["optim"]["EPOCHS"]
  data_type = hparams["task"]["data_type"]
  task_type = hparams["task"]["task_type"]
  target_feature = hparams["data"]["target_feature"]
  if target_feature == "None":
    target_feature = None

  smc_transformer = SMC_Transformer(num_layers=num_layers,
                                    d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    output_size=target_vocab_size,
                                    maximum_position_encoding=maximum_position_encoding_smc,
                                    num_particles=num_particles,
                                    sigma=sigma,
                                    sigma_obs=omega,
                                    noise_encoder=noise_encoder,
                                    noise_SMC_layer=noise_SMC_layer,
                                    seq_len=seq_len,
                                    data_type=data_type,
                                    task_type=task_type,
                                    resampling=resampling,
                                    layer_norm=layer_norm,
                                    target_feature=target_feature,
                                    rate=rate)

  # creating checkpoint manage
  ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                             optimizer=optimizer)
  smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer_{}".format(num_train))
  ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS)

  # if a checkpoint exists, restore the latest checkpoint.
  start_epoch = restoring_checkpoint(ckpt_manager=ckpt_manager, ckpt=ckpt, args_load_ckpt=args.load_ckpt, logger=logger)
  if start_epoch is None:
    start_epoch = 0

  # check the pass forward.
  for input_example_batch, target_example_batch in train_dataset.take(2):
    # input_model = tf.concat([input_example_batch, target_example_batch[:,-1,:]], axis = 1)
    (example_batch_predictions, traj, _, _), _ = smc_transformer(inputs=input_example_batch,
                                                                                     training=True,
                                                                                     mask=create_look_ahead_mask(
                                                                                       seq_len))
    print("predictions shape: {}".format(example_batch_predictions.shape))

  if start_epoch > 0:
    if start_epoch > EPOCHS:
      print("adding {} more epochs to existing training".format(EPOCHS))
      start_epoch = 0
    else:
      logger.info("starting training after checkpoint restoring from epoch {}".format(start_epoch))

  if not layer_norm:
    logger.info("Training a SMC Transformer without layer norm...")

  start_training = time.time()

  # preparing recording of loss and metrics information
  train_loss_history, train_loss_mse_history, train_loss_mse_avg_pred_history = [], [], []
  val_loss_history, val_loss_mse_history, val_loss_mse_avg_pred_history = [], [], []

  for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    logger.info('Epoch {}/{}'.format(epoch + 1, EPOCHS))
    train_accuracy.reset_states()
    sum_total_train_loss, sum_total_val_loss = 0, 0
    sum_train_loss, sum_val_loss = 0, 0
    sum_train_loss_avg_pred, sum_val_loss_avg_pred = 0, 0
    sum_train_loss_std, sum_val_loss_std = 0, 0

    # training step:
    for (batch, (inp, tar)) in enumerate(train_dataset):
      total_loss_batch, train_metrics, _ = train_step_SMC_T(inputs=inp,
                                                               targets=tar,
                                                               smc_transformer=smc_transformer,
                                                               optimizer=optimizer,
                                                               train_accuracy=train_accuracy,
                                                               classic_loss=True,
                                                               SMC_loss=True)
      mse_metric_batch, mse_loss_avg_pred_batch, mse_loss_std_batch = train_metrics
      sum_total_train_loss += total_loss_batch
      sum_train_loss += mse_metric_batch
      sum_train_loss_avg_pred += mse_loss_avg_pred_batch
      sum_train_loss_std += mse_loss_std_batch

    avg_total_train_loss = sum_total_train_loss / (batch + 1)
    avg_train_loss = sum_train_loss / (batch + 1)
    avg_train_loss_avg_pred = sum_train_loss_avg_pred / (batch + 1)
    avg_train_loss_std = sum_train_loss_std / (batch + 1)

    # compute the validation accuracy on the validation dataset:
    for batch_val, (inp, tar) in enumerate(val_dataset):
      (predictions_val, _, weights_val, ind_matrix_val), attn_weights_val = smc_transformer(
        inputs=inp,
        training=False,
        mask=create_look_ahead_mask(seq_len))
      total_val_loss_batch, val_loss_mse_batch, val_loss_mse_avg_pred_batch, val_loss_mse_std_batch = loss_function_regression(real=tar,
                                                                                                  predictions=predictions_val,
                                                                                                  weights=weights_val,
                                                                                                  transformer=smc_transformer)
      sum_total_val_loss += total_val_loss_batch
      sum_val_loss += val_loss_mse_batch
      sum_val_loss_avg_pred += val_loss_mse_avg_pred_batch
      sum_val_loss_std += val_loss_mse_std_batch

    avg_total_val_loss = sum_total_val_loss / (batch_val + 1)
    avg_val_loss = sum_val_loss / (batch_val + 1)
    avg_val_loss_avg_pred = sum_val_loss_avg_pred / (batch + 1)
    avg_val_loss_std = sum_val_loss_std / (batch_val + 1)

    logger.info('final weights of first 3 elements of batch: {}, {}, {}'.format(weights_val[0, :], weights_val[1, :],
                                                                                weights_val[2, :]))

    # ------------------------- computing and saving metrics (train set and validation set)----------------------------------------------------
    template = 'train loss: {}, train mse loss: {} - train mse loss (from avg pred): {} - train loss std (mse): {} - val loss: {} - val mse loss: {} - val mse loss (from avg pred): {}, val loss std (mse): {}'
    logger.info(template.format(avg_total_train_loss.numpy(),
                                avg_train_loss.numpy(),
                                avg_train_loss_avg_pred.numpy(),
                                avg_train_loss_std.numpy(),
                                avg_total_val_loss.numpy(),
                                avg_val_loss.numpy(),
                                avg_val_loss_avg_pred.numpy(),
                                avg_val_loss_std.numpy()))

    # saving loss and metrics information:
    train_loss_history.append(avg_total_train_loss.numpy())
    train_loss_mse_history.append(avg_train_loss.numpy())
    train_loss_mse_avg_pred_history.append(avg_train_loss_avg_pred.numpy())
    val_loss_history.append(avg_total_val_loss.numpy())
    val_loss_mse_history.append(avg_val_loss.numpy())
    val_loss_mse_avg_pred_history.append(avg_val_loss_avg_pred.numpy())

    # ------------- end of saving metrics information -------------------------------------------------------------------------------

    ckpt_save_path = ckpt_manager.save()

    logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  logger.info('total training time for {} epochs:{}'.format(EPOCHS, time.time() - start_training))

  keys = ['train loss', 'train mse loss', 'train mse loss (avg pred)' 'val loss', 'val mse loss', 'val mse loss (avg pred)']
  values = [train_loss_history, train_loss_mse_history, train_loss_mse_avg_pred_history, val_loss_history, val_loss_mse_history, val_loss_mse_avg_pred_history]
  csv_fname='smc_transformer_history_{}.csv'.format(num_train)
  saving_training_history(keys=keys, values=values,
                          output_path=output_path,
                          csv_fname=csv_fname,
                          logger=logger,
                          start_epoch=start_epoch)

  # # making predictions with the trained model and saving them on .npy files
  # saving_model_outputs(output_path=output_path,
  #                      predictions=predictions_val,
  #                      attn_weights=attn_weights_val,
  #                      pred_fname='smc_predictions.npy',
  #                      attn_weights_fname='smc_attn_weights.npy',
  #                      logger=logger)
  #
  # model_output_path = os.path.join(output_path, "model_outputs")
  # # saving weights on top of it.
  # weights_fn = model_output_path + '/' + 'smc_weights.npy'
  # np.save(weights_fn, weights_val)

  # ----------------------  compute statistics at the end of training ----------------------------------------------------------------------------

  logger.info("computing metrics at the end of training...")
  train_loss_mse, train_loss_mse_avg_pred, train_loss_std, val_loss_mse, val_loss_mse_avg_pred, val_loss_std = [], [], [], [], [], []
  for batch_train, (inp, tar) in enumerate(train_dataset):
    (predictions_train, _, weights_train, _), attn_weights_train = smc_transformer(
      inputs=inp,
      training=False,
      mask=create_look_ahead_mask(seq_len))
    _, train_loss_mse_batch, train_loss_mse_avg_pred_batch, train_loss_mse_std_batch = loss_function_regression(real=tar,
                                                                                 predictions=predictions_train,
                                                                                 weights=weights_train,
                                                                                 transformer=smc_transformer)
    train_loss_mse.append(train_loss_mse_batch.numpy())
    train_loss_mse_avg_pred.append(train_loss_mse_avg_pred_batch.numpy())
    train_loss_std.append(train_loss_mse_std_batch.numpy())

  for batch_val, (inp, tar) in enumerate(val_dataset):
    (predictions_val, _, weights_val, _), attn_weights_val = smc_transformer(
      inputs=inp,
      training=False,
      mask=create_look_ahead_mask(seq_len))
    _, val_loss_mse_batch, val_loss_mse_avg_pred_batch, val_loss_mse_std_batch = loss_function_regression(real=tar,
                                                                             predictions=predictions_val,
                                                                             weights=weights_val,
                                                                             transformer=smc_transformer)

    val_loss_mse.append(val_loss_mse_batch.numpy())
    val_loss_mse_avg_pred.append(val_loss_mse_avg_pred_batch.numpy())
    val_loss_std.append(val_loss_mse_std_batch.numpy())

  # computing as a metric the mean of losses & std losses over the number of batches
  mean_train_loss_mse = statistics.mean(train_loss_mse)
  mean_train_loss_mse_avg_pred = statistics.mean(train_loss_mse_avg_pred)
  mean_train_loss_std = statistics.mean(train_loss_std)
  mean_val_loss_mse = statistics.mean(val_loss_mse)
  mean_val_loss_mse_avg_pred = statistics.mean(val_loss_mse_avg_pred)
  mean_val_loss_std = statistics.mean(val_loss_std)

  logger.info(
    "average losses over batches: train mse loss:{} - train mse loss (avg pred): {}, train loss std (mse):{} - val mse loss:{} - val mse loss (avg pred): {} - val loss (mse) std: {}".format(
      mean_train_loss_mse,
      mean_train_loss_mse_avg_pred,
      mean_train_loss_std,
      mean_val_loss_mse,
      mean_val_loss_mse_avg_pred,
      mean_val_loss_std))

  logger.info('training of SMC Transformer for a time-series dataset done...')

  logger.info(
    ">>>--------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")