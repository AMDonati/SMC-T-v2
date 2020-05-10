import tensorflow as tf
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask

# -------------------------------- TRAIN STEP FUNCTIONS ---------------------------------------------------------------------
train_step_signature = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

#@tf.function(input_signature=train_step_signature) #TODO: debug this problem
def train_step_classic_T(inputs, targets, transformer, optimizer):
  '''training step for the classic Transformer model'''

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
#@tf.function(input_signature=train_step_signature)
def train_step_SMC_T(inputs, targets, smc_transformer, optimizer):
  '''
  :param inputs:
  :param targets:
  :param smc_transformer:
  :param optimizer:
  :return:
  '''

  assert len(tf.shape(inputs)) == len(tf.shape(targets)) == 4

  with tf.GradientTape() as tape:
    (preds, preds_resampl), _,  _ = smc_transformer(inputs=inputs,
                                         targets=targets) # predictions: shape (B,P,S,F_y) with P=1 during training.

    targets_tiled = tf.tile(targets, multiples=[1, smc_transformer.cell.num_particles, 1, 1])
    classic_loss = tf.keras.losses.MSE(targets_tiled, preds_resampl) # (B,P,S)
    classic_loss = tf.reduce_mean(classic_loss) # mean over all dimensions.

    if smc_transformer.cell.noise:
      smc_loss, smc_loss_no_log = smc_transformer.compute_SMC_loss(predictions=preds_resampl, targets=targets_tiled)
      loss = smc_loss
      mse_metric_avg_pred = tf.keras.losses.MSE(targets, tf.reduce_mean(preds, axis=1, keepdims=True)) # (B,1,S)
      mse_metric_avg_pred = tf.reduce_mean(mse_metric_avg_pred)
    else:
      loss = classic_loss

    gradients = tape.gradient(loss, smc_transformer.trainable_variables)

    # To debug the loss.
    #trainable_variables = list(smc_transformer.trainable_variables)
    #trainable_variables_names = [t.name for t in trainable_variables]
    #var_and_grad_dict = dict(zip(trainable_variables_names, gradients))
    #print('dict of variables and associated gradients', var_and_grad_dict)

  optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))
  if smc_transformer.cell.noise:
    smc_transformer.cell.Sigma_obs.assign(tf.math.maximum(0,smc_transformer.cell.Sigma_obs))
    smc_transformer.cell.attention_smc.sigma_k.assign(tf.math.maximum(0, smc_transformer.cell.attention_smc.sigma_k))
    smc_transformer.cell.attention_smc.sigma_q.assign(tf.math.maximum(0, smc_transformer.cell.attention_smc.sigma_q))
    smc_transformer.cell.attention_smc.sigma_v.assign(tf.math.maximum(0, smc_transformer.cell.attention_smc.sigma_v))
    smc_transformer.cell.attention_smc.sigma_z.assign(tf.math.maximum(0, smc_transformer.cell.attention_smc.sigma_z))
    assert smc_transformer.cell.Sigma_obs >= 0
    assert smc_transformer.cell.attention_smc.sigma_k >=0
    assert smc_transformer.cell.attention_smc.sigma_q >=0
    assert smc_transformer.cell.attention_smc.sigma_v >=0
    assert smc_transformer.cell.attention_smc.sigma_z >=0

  if smc_transformer.cell.noise:
    return loss, smc_loss_no_log, mse_metric_avg_pred
  else:
    return loss, None, None

#
# @tf.function
# def train_step_rnn_classif(inp, target, model, optimizer, accuracy_metric):
#   with tf.GradientTape() as tape:
#     predictions = model(inp)
#     loss = tf.reduce_mean(
#       tf.keras.losses.sparse_categorical_crossentropy(target, predictions, from_logits=True))
#   grads = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(grads, model.trainable_variables))
#   train_acc_batch = accuracy_metric(target, predictions)
#   return loss, train_acc_batch













