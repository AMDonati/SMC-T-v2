#TODO: test the classification loss for a number of classes equal to 2.
#TODO: add a mask option in the loss for nlp datasets.

#TODO: debug the mse_with_particles function for the regression case.
import tensorflow as tf

### ----------------------- LOSS FUNCTIONS------------------------------------------------------------------------------

def loss_function_classic_T_classif(real, pred, data_type):
  # squeezing 'real' to have a shape of (B,S):
  real=tf.squeeze(real, axis=-1)
  if data_type=='nlp':
    mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss_ = loss_object(real, pred) # shape (B,S)
  if data_type=='nlp':
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
  # averaging over the sequences dimension:
  loss_=tf.reduce_mean(loss_, axis=-1) # (B,)
  return tf.reduce_mean(loss_)


def categorical_ce_with_particules(real, pred, sampling_weights, data_type):
  '''
  :param real: targets tensor > shape (B,S)
  :param pred: predictions (particules logits) > shape (B,P,S,V)
  :param sampling_weights: re-sampling weights for last timestep > shape (B,P)
  :return:
  '''
  # tiling the targets to have a shape (B,P,S)
  num_particles = tf.shape(pred)[1]

  if len(tf.shape(real)) < 3:
    real = tf.expand_dims(real, axis=1)
    real = tf.tile(real, multiples=[1, num_particles, 1])

  if data_type == 'nlp':
    mask = tf.math.logical_not(tf.math.equal(real, 0))

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss_ = loss_object(real, pred)  # shape (B,P,S)

  if data_type=='nlp':
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

  # mean over sequence elements
  loss_ = tf.reduce_mean(loss_, axis=-1)  # shape (B,P)
  # weighted sum over number of particles
  loss_ = tf.reduce_sum(sampling_weights * loss_, axis=-1)
  # mean over batch elements
  loss = tf.reduce_mean(loss_, axis=0)
  return loss


def binary_ce_with_particules(real, pred, sampling_weights, from_logits=True):
  #TODO: not working: to debug one of these days.
  '''
  DOES NOT WORK. USE THE Categorical_one instead, event for 2 classes.
  :param real: targets tensor > shape (B,S)
  :param pred: predictions (particules logits) > shape (B,P,S,1)
  :param sampling_weights: re-sampling weights for last timestep > shape (B,P)
  :return:
  '''
  # tiling the targets to have a shape (B,P,S)
  num_particles = tf.shape(pred)[1]
  if len(tf.shape(real)) < 3:
    real = tf.expand_dims(real, axis=1)
    real = tf.tile(real, multiples=[1, num_particles, 1])

  # One-hot encoding of real to have a shape (B,P,S,2)
  real = tf.cast(real, dtype=tf.int32)
  real = tf.one_hot(real, depth=2)

  loss_ = tf.keras.losses.binary_crossentropy(
    y_true=real,
    y_pred=pred,
    from_logits=from_logits,
    label_smoothing=0)  # shape (B,P,S)

  # mean over sequence elements
  loss_ = tf.reduce_mean(loss_, axis=-1)  # shape (B,P)

  # weighted sum over number of particles

  loss_ = tf.reduce_sum(sampling_weights * loss_, axis=-1)  # squeezing weights to have shape (B,P)

  # mean over batch elements
  loss = tf.reduce_mean(loss_, axis=0)

  return loss

def mse_with_particles(real, pred):
  '''
  :param real: shape (B,S,F)
  :param pred: shape (B,P,S,F)
  :param sampling_weights: shape (B,P)
  :return:
  the average mse scalar loss (with a weighted average over the dim number of particles)
  '''
  # tiling real over the particles dimension to have a tensor of shape (B,P,S,1)

  num_particles=tf.shape(pred)[1]
  real = tf.expand_dims(real, axis=1)
  real= tf.tile(real, multiples=[1, num_particles, 1, 1])
  mse = tf.keras.losses.MSE
  loss_tensor = mse(y_true=real, y_pred=pred)  # shape (B,P,S)

  loss_tensor = tf.reduce_mean(loss_tensor, axis=-1) # mean over seq dim > (B,P)

  loss = tf.reduce_mean(loss_tensor, axis=-1) # mean over particles dim (weights of 1/M because is resampling is done after propagation.) > (B,)
  loss_std = tf.math.reduce_std(loss_tensor, axis = -1)

  loss = tf.reduce_mean(loss, axis=-1) # mean over batch dims.
  loss_std = tf.reduce_mean(loss_std, axis=-1)

  return loss, loss_std


def loss_function_classification(real, predictions, weights, transformer, classic_loss=True, SMC_loss=True):
  '''
  :param real: targets > shape (B,P,S)
  :param predictions (log_probas) > shape (B,P,S,V)
  :param weights: re-sampling_weights for the last element > shape (B,P)
  :param classic_loss: boolean to compute the classic loss or not (default=True)
  :param SMC_loss: boolean to compute the SMC loss (default=True)
  :return:
  a scalar computing the SMC loss as defined in the paper.
  '''
  if classic_loss:
    loss_ce = categorical_ce_with_particules(real=real,
                                             pred=predictions,
                                             sampling_weights=weights,
                                             data_type=transformer.data_type)
  else:
    loss_ce = 0
  if SMC_loss:
    loss_smc = -transformer.compute_SMC_log_likelihood(sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
  else:
    loss_smc = 0
  loss = loss_ce + loss_smc
  return loss


def loss_function_binary(real, predictions, weights, transformer, classic_loss=True, SMC_loss=True):
  '''DOES NOT WORK...
  :param real: targets > shape (B,P,S,F)
  :param predictions (log_probas) > shape (B,P,S,C=F)
  :param weights: re-sampling_weights for the last element > shape (B,P)
  :param classic_loss: boolean to compute the classic loss or not (default=True)
  :param SMC_loss: boolean to compute the SMC loss (default=True)
  :return:
  a scalar computing the SMC loss as defined in the paper.
  '''
  if classic_loss:
    loss_ce = binary_ce_with_particules(real=real, pred=predictions, sampling_weights=weights)
  else:
    loss_ce = 0
  if SMC_loss:
    loss_smc = -transformer.compute_SMC_log_likelihood(real=real, sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
  else:
    loss_smc = 0
  loss = loss_ce + loss_smc
  return loss


def loss_function_regression(real, predictions, weights, transformer, classic_loss=True, SMC_loss=True):
  '''
  :param real: targets > shape (B,P,S) (B,P,S,F)
  :param predictions > shape (B,P,S,1) (B,P,S,F)
  :param weights: re-sampling_weights for the last element > shape (B,P)
  :param classic_loss: boolean to compute the classic loss or not (default=True)
  :param SMC_loss: boolean to compute the SMC loss (default=True)
  :return:
  a scalar computing the SMC loss as defined in the paper.
  '''
  if classic_loss:
    # TODO: if sigma of weights_computation is not equal to 1. change the mse by a custom SMC_log_likelihood.
    loss_mse, loss_mse_std = mse_with_particles(real=real, pred=predictions)
  else:
    loss_mse = 0
  if SMC_loss:
    # take minus the log_likelihood.
    loss_smc = -transformer.compute_SMC_log_likelihood(sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
  else:
    loss_smc = 0
  loss = loss_mse + loss_smc

  # compute mse from average prediction.
  avg_prediction = tf.reduce_mean(predictions, axis=1) # (B,S,F)
  loss_mse_from_avg_pred = tf.keras.losses.MSE(real, avg_prediction) # (B,S)
  loss_mse_from_avg_pred = tf.reduce_mean(loss_mse_from_avg_pred, axis=-1) # (B)
  loss_mse_from_avg_pred = tf.reduce_mean(loss_mse_from_avg_pred, axis=-1)

  return loss, loss_mse, loss_mse_from_avg_pred,  loss_mse_std

# -------- custom schedule for learning rate... -----------------------------------------------------------------
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def compute_accuracy_variance(predictions_val, tar, accuracy_metric):
  """
  :param predictions_val: particles of predictions > shape (B,P,S,V)
  :param tar: targets > shape (B,S)
  :param accuracy_metric: the tf.keras.metric object used to compute the accuracy.
  :return:
  """
  accuracies=[]
  num_particles=tf.shape(predictions_val)[1]

  for m in range(num_particles):
    pred_particle=predictions_val[:,m,:,:] # shape (B,S,V)
    acc_particle=accuracy_metric(tar, pred_particle)
    accuracies.append(acc_particle.numpy())

  accuracies.sort()
  variance_acc=accuracies[num_particles-1]-accuracies[0]

  return variance_acc

if __name__ == "__main__":
  #------------------------ testing of categorical ce with particules function......-----------------------------------------------------
  B = 8
  P = 5
  S = 10
  V = 50

  real = tf.ones(shape=(B,P,S))
  logits = tf.random.uniform(shape=(B,P,S,V))
  sampling_weights = tf.ones(shape=(B,P))
  loss=categorical_ce_with_particules(real, logits, sampling_weights, data_type='nlp')

  print('categorical ce loss for {} classes'.format(V), loss.numpy())

  # test in the binary case:
  V = 2
  logits = tf.random.uniform(shape=(B, P, S, V))
  sampling_weights = tf.ones(shape=(B, P))
  loss_binary = categorical_ce_with_particules(real, logits, sampling_weights, data_type='nlp')

  print('categorical ce loss - binary case', loss_binary.numpy())

  #-------------------- test of mse with particles loss -----------------------------------------------------------------------------------
  B = 8
  P = 1
  S = 10
  V = 1

  real = tf.random.uniform(shape=(B, P, S))
  logits = tf.random.uniform(shape=(B, P, S, V))
  sampling_weights = tf.ones(shape=(B, P, 1))

  loss_regression=mse_with_particles(real=real, pred=logits, sampling_weights=sampling_weights)

  print('regression loss', loss_regression.numpy())

  #------- test of compute accuracy_variance-------
  predictions_val=tf.ones(shape=(8,3,10,65))
  tar=tf.zeros((8,10))
  accuracy_metric=tf.keras.metrics.SparseCategoricalAccuracy()
  acc_variance=compute_accuracy_variance(predictions_val, tar, accuracy_metric)
  print('variance', acc_variance)



  # ------ old loss functions ------------

  # def mse_with_particles(real, pred, sampling_weights):
  #   '''
  #   :param real: shape (B,S,F)
  #   :param pred: shape (B,P,S,F)
  #   :param sampling_weights: shape (B,P)
  #   :return:
  #   the average mse scalar loss (with a weighted average over the dim number of particles)
  #   '''
  #   # tiling real over the particles dimension to have a tensor of shape (B,P,S,1)
  #
  #   num_particles = tf.shape(pred)[1]
  #   real = tf.expand_dims(real, axis=1)
  #   real = tf.tile(real, multiples=[1, num_particles, 1, 1])
  #   mse = tf.keras.losses.MSE
  #   loss = mse(y_true=real, y_pred=pred)  # shape (B,P,S)
  #
  #   # loss=tf.keras.metrics.Mean(loss)
  #
  #   # # mean over the sequence dimension.
  #   # loss = tf.reduce_mean(loss, axis=-1)  # shape (B,P,S)
  #   # # squeezing sampling_weights to have a shape (B,P)
  #   # if len(tf.shape(sampling_weights))==3:
  #   #   sampling_weights=tf.squeeze(sampling_weights, axis=-1)
  #   # # weighted average over the particle dimension.
  #   # loss = tf.reduce_sum(sampling_weights * loss)  # shape (B,)
  #   # # mean over the batch elements.
  #   # loss = tf.reduce_mean(loss)
  #   return loss

# #------ old function not working------
# def categorical_crossentropy(real, logits, sampling_weights):
#     '''formula: mean(over batch)[sum(w(m)*-sum(real*log pred))
#     -args:
#       -real: tensor of dim (B,P,S) or dim (B,S)
#       -pred (logits):tensor of dim (B,P,S,V) with V the vocabulary size.
#       -sampling_weights: tensor of dim (B,P)'''
#     num_particles = tf.shape(sampling_weights)[-1]
#     #if len(tf.shape(real)) < 3:
#       #real = tf.tile(real[:, tf.newaxis, :], multiples=[1, num_particles, 1])  # add particles dimension > dim (B,P,S)
#     pred = tf.reduce_max(logits, axis=-1)  # dim (B, P, S)
#     pred = tf.cast(pred, dtype=tf.float32)
#     real = tf.cast(real, dtype=tf.float32)
#     loss = -tf.reduce_sum(real * tf.math.log(pred), axis=-1)  # dim (B,P)
#     # weighted sum using sampling_weights.
#     loss = tf.reduce_sum(sampling_weights * loss, axis=-1)  # dim (B,)
#     # averaging over the batch
#     loss = tf.reduce_mean(loss, axis=0)
#     print(loss.shape)
#     return loss







