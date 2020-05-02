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

def train_LSTM(model, optimizer, EPOCHS, train_dataset_for_RNN, val_dataset_for_RNN, output_path, checkpoint_path, logger, num_train):
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


def train_SMC_transformer(smc_transformer, optimizer, EPOCHS, train_dataset, val_dataset, checkpoint_path, logger, num_train):

  # creating checkpoint manager
  ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                             optimizer=optimizer)
  smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer_{}".format(num_train))
  ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS) #TODO: change this?

  # if a checkpoint exists, restore the latest checkpoint.
  start_epoch = restoring_checkpoint(ckpt_manager=ckpt_manager, ckpt=ckpt, args_load_ckpt=True, logger=logger)
  if start_epoch is None:
    start_epoch = 0

  # check the pass forward.
  for input_example_batch, target_example_batch in train_dataset.take(1):
    example_batch_predictions, _, _ = smc_transformer(inputs=input_example_batch, targets=target_example_batch)
    print("predictions shape: {}".format(example_batch_predictions.shape))

  if start_epoch > 0:
    if start_epoch > EPOCHS:
      print("adding {} more epochs to existing training".format(EPOCHS))
      start_epoch = 0
    else:
      logger.info("starting training after checkpoint restoring from epoch {}".format(start_epoch))

  for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    logger.info('Epoch {}/{}'.format(epoch+1, EPOCHS))
    sum_train_loss, sum_val_loss = 0., 0.

    for batch, (inp, tar) in enumerate(train_dataset):
      train_loss_batch = train_step_SMC_T(inputs=inp,
                                          targets=tar,
                                          smc_transformer=smc_transformer,
                                          optimizer=optimizer)
      sum_train_loss += train_loss_batch

    train_loss = sum_train_loss / (batch+1)

    for batch, (inp, tar) in enumerate(val_dataset):
      predictions_val, _, _ = smc_transformer(inputs=inp, targets=tar) # shape (B,1,S,F_y)
      val_loss_batch = tf.keras.losses.MSE(tar, predictions_val) # (B,1,S)
      val_loss_batch = tf.reduce_mean(val_loss_batch, axis=-1) # (B,1)
      val_loss_batch = tf.reduce_mean(val_loss_batch, axis=-1) # (B)
      val_loss_batch = tf.reduce_mean(val_loss_batch, axis=-1)
      sum_val_loss+= val_loss_batch

    val_loss = sum_val_loss / (batch + 1)

    logger.info('train loss: {:5.3f} - val loss: {:5.3f}'.format(train_loss.numpy(), val_loss.numpy()))
    ckpt_manager.save()
    logger.info('Time taken for 1 epoch: {} secs'.format(time.time() - start))

  logger.info('total training time for {} epochs:{:10.1f}'.format(EPOCHS, time.time() - start))