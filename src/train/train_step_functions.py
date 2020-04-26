# TODO: test the classification loss for a number of classes equal to 2.
# TODO: debug the mse_with_particles function for the regression case.
import tensorflow as tf
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.loss_functions import loss_function_classic_T_classif
from train.loss_functions import loss_function_classification
from train.loss_functions import loss_function_regression

# -------------------------------- TRAIN STEP FUNCTIONS ---------------------------------------------------------------------
train_step_signature = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

#@tf.function(input_signature=train_step_signature)
def train_step_classic_T(inputs, targets, transformer, optimizer):
  '''training step for the classic Transformer model'''
  # CAUTION. Unlike the SMC_Transformer, the inputs and targets need to be of shape (B,S,1).
  assert len(tf.shape(inputs)) == 3
  assert len(tf.shape(targets)) == 3

  seq_len = tf.shape(inputs)[-2]
  mask_transformer = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inputs=inputs, training=True, mask=mask_transformer)

    loss = tf.keras.losses.MSE(targets, predictions)
    # averaging loss over the seq and batch dims
    loss = tf.reduce_mean(loss, axis=-1) # (B,)
    loss = tf.reduce_mean(loss, axis=-1)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  return loss

# --------------SMC Transformer train_step-----------------------------------------------------------------------------------------------------
@tf.function(input_signature=train_step_signature)
def train_step_SMC_T(inputs,
                     smc_transformer,
                     optimizer,
                     targets=None,
                     perplexity_metric=None,
                     SMC_loss=True,
                     classic_loss=True):
  '''
  compute a gradient descent step using categorical crossentropy loss by updating the trainable parameters.
  :param input: input data > shape (B,S) for nlp and univariate time_series.
  multivariate case needs to be implemented still.
  :param target: target data (sequential one) > shape (B,S).
  :param SMC_loss: boolean to compute SMC_loss or not. Default is False.
  :param classic_loss: boolean to compute classic cross-entropy loss, or not. Default is True.
  :return:
  The updated loss, the training accuracy (from average predictions and from max predictions).
  '''

  if targets is None:
    tar_inp = inputs[:, :-1]
    tar_real = inputs[:, 1:]
  else:
    tar_inp = inputs
    tar_real = targets # (B,S,F)

  seq_len = tf.shape(tar_inp)[1]
  mask_transformer = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    (predictions, trajectories, weights, ind_matrix), attn_weights = smc_transformer(inputs=tar_inp,
                                                                                             training=True,
                                                                                             mask=mask_transformer)
    # predictions: shape (B,P,S,C) > sequence of log_probas for the classification task.
    # trajectories: shape (B,P,S,D) = [z0,z1,z2,...,zT]
    # weights: shape (B,P,1) = w_T: used in the computation of the loss.

    #train_inf_pred_batch, train_avg_pred_batch, train_max_pred_batch = predictions_metric

    if smc_transformer.task_type == 'classification':
      assert tf.shape(predictions)[-1] > 2
      loss = loss_function_classification(real=tar_real,
                                          predictions=predictions,
                                          weights=weights,
                                          transformer=smc_transformer,
                                          SMC_loss=SMC_loss,
                                          classic_loss=classic_loss)

    elif smc_transformer.task_type == 'regression':
      loss, loss_mse, loss_mse_from_avg_pred, loss_mse_std = loss_function_regression(real=tar_real,
                                      predictions=predictions,
                                      weights=weights,
                                      transformer=smc_transformer,
                                      SMC_loss=SMC_loss,
                                      classic_loss=classic_loss)
    else:
      raise ValueError('task_type argument in Transformer class is not supported.'
                       'Please choose between "classification" or "regression"')

    trainable_variables = list(smc_transformer.trainable_variables)
    trainable_variables_names = [t.name for t in trainable_variables]
    gradients = tape.gradient(loss, smc_transformer.trainable_variables)
    #var_and_grad_dict = dict(zip(trainable_variables_names, gradients))
    #print('dict of variables and associated gradients', var_and_grad_dict)

  optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))

  # if smc_transformer.task_type == 'classification':
  #   train_inf_batch = train_accuracy(tar_real, train_inf_pred_batch)  # accuracy from average_predictions for now.
  #   train_avg_acc_batch = train_accuracy(tar_real, train_avg_pred_batch)  # average over logits instead of after softmax (inference case).
  #   train_max_acc_batch = train_accuracy(tar_real, train_max_pred_batch)
  #   train_metrics = (train_inf_batch, train_avg_acc_batch, train_max_acc_batch)
  # else:
  train_metrics = (loss_mse, loss_mse_from_avg_pred, loss_mse_std)

  if perplexity_metric is not None:
    train_perplexity = perplexity_metric(tar_real, predictions)
  else:
    train_perplexity = None

  return loss, train_metrics, train_perplexity


@tf.function
def train_step_rnn_classif(inp, target, model, optimizer, accuracy_metric):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
      tf.keras.losses.sparse_categorical_crossentropy(target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  train_acc_batch = accuracy_metric(target, predictions)
  return loss, train_acc_batch













