import tensorflow as tf
import numpy as np
import statistics
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.loss_functions import loss_function_regression
import os


def compute_latest_statistics(smc_transformer, train_dataset, val_dataset, seq_len, output_path, logger):
  train_loss_mse, train_loss_mse_avg_pred, train_loss_std, val_loss_mse, val_loss_mse_avg_pred, val_loss_std = [], [], [], [], [], []
  predictions_validation_set = []
  for batch_train, (inp, tar) in enumerate(train_dataset):
    (predictions_train, _, weights_train, _), attn_weights_train = smc_transformer(
      inputs=inp,
      training=False,
      mask=create_look_ahead_mask(seq_len))
    _, train_loss_mse_batch, train_loss_mse_avg_pred_batch, train_loss_mse_std_batch = loss_function_regression(
      real=tar,
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
    predictions_validation_set.append(predictions_val)
    val_loss_mse.append(val_loss_mse_batch.numpy())
    val_loss_mse_avg_pred.append(val_loss_mse_avg_pred_batch.numpy())
    val_loss_std.append(val_loss_mse_std_batch.numpy())

  # saving predictions for all validation set:
  predictions_validation_set = tf.stack(predictions_validation_set, axis=0)
  predictions_val_path = output_path + "/" + "predictions_val_end_of_training.npy"
  np.save(predictions_val_path, predictions_validation_set)

  logger.info("saving predictions on validation set...")
  logger.info("predictions shape: {}".format(predictions_validation_set.shape))

  # computing as a metric the mean of losses & std losses over the number of batches
  mean_train_loss_mse = statistics.mean(train_loss_mse)
  mean_train_loss_std = statistics.mean(train_loss_std)
  mean_train_loss_mse_avg_pred = statistics.mean(train_loss_mse_avg_pred)
  mean_val_loss_mse = statistics.mean(val_loss_mse)
  mean_val_loss_mse_avg_pred = statistics.mean(val_loss_mse_avg_pred)
  mean_val_loss_std = statistics.mean(val_loss_std)

  logger.info(
    "train mse loss:{} - train mse loss (avg pred): {} - train loss std (mse):{} - val mse loss:{} - val mse loss (avg pred): {} - val loss (mse) std: {}".format(
      mean_train_loss_mse,
      mean_train_loss_mse_avg_pred,
      mean_train_loss_std,
      mean_val_loss_mse,
      mean_val_loss_mse_avg_pred,
      mean_val_loss_std))

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def evaluate_SMC_Transformer(smc_transformer, test_dataset, seq_len, task, stats, output_path, logger):
  for (test_data, y_test) in test_dataset:
    (predictions_test, _, weights_test, _), attn_weights_test = smc_transformer(
      inputs=test_data,
      training=False,
      mask=create_look_ahead_mask(seq_len))
    _, test_loss_mse, test_loss_mse_avg_pred, test_loss_mse_std = loss_function_regression(real=y_test,
                                                                                           predictions=predictions_test,
                                                                                           weights=weights_test,
                                                                                           transformer=smc_transformer)

  logger.info('test mse loss: {} - test mse loss (avg pred): {}, test loss std(mse) - {}'.format(test_loss_mse,
                                                                                                 test_loss_mse_avg_pred,
                                                                                                 test_loss_mse_std))
  # unnormalized predictions & target:
  if task == 'unistep-forcst':
    data_mean, data_std = stats
    predictions_unnormalized = predictions_test * data_std + data_mean
    targets_unnormalized = y_test * data_std + data_mean

  # save predictions & attention weights:
  logger.info("saving predictions for test set in .npy files...")
  eval_output_path = os.path.join(output_path, "eval_outputs")
  if not os.path.isdir(eval_output_path):
    os.makedirs(eval_output_path)

  pred_unistep_N_1_test = eval_output_path + '/' + 'pred_unistep_N_1_test.npy'
  attn_weights_unistep_N_1_test = eval_output_path + '/' + 'attn_weights_unistep_N_1_test.npy'
  targets_test = eval_output_path + '/' + 'targets_test.npy'
  weights_test_path = eval_output_path + '/' + 'weights_test.npy'
  if task == 'unistep-forcst':
    pred_unnorm = eval_output_path + '/' + 'pred_unistep_N_1_test_unnorm.npy'
    targets_unnorm = eval_output_path + '/' + 'targets_test_unnorm.npy'

  np.save(pred_unistep_N_1_test, predictions_test)
  np.save(attn_weights_unistep_N_1_test, attn_weights_test)
  np.save(targets_test, y_test)
  np.save(weights_test_path, weights_test)

  if task == 'unistep-forcst':
    np.save(pred_unnorm, predictions_unnormalized)
    np.save(targets_unnorm, targets_unnormalized)

  logger.info("predictions shape for test set:{}".format(predictions_test.shape))
  if task == 'unistep-forcst':
    logger.info("unormalized predictions shape for test set: {}".format(predictions_unnormalized.shape))