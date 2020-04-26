import tensorflow as tf
import math

#TODO: redo the mathematical computation of the loss from the log likelihood to check that the formulas implemented are correct.

#TODO: ask Florian if I need to add a @tf.function to this function. cf https://www.tensorflow.org/tutorials/generative/cvae as an example.
def compute_SMC_ll_one_layer(list_means):
  '''
  compute log p(z_j / z_(0:j-1), Y(0:j-1))
  :param epsilon: epsilon of the reparametrized noise (normal gaussian distrib.) > shape (B,P,S,D)
  :param sigma: cov variance matrix in the reparametrized noise > shape (D,D)
  :return:
  a tensor of shape (B,P,S) with the log-likelihood corresponding to one layer.
  '''
  mean_z = list_means[0]
  mean_k = list_means[1]
  mean_v = list_means[2]
  mean_q = list_means[3]

  mean_z = tf.transpose(mean_z, perm=[0, 3, 1, 2]) # shape (B,D,P,S)
  mean_k = tf.transpose(mean_k, perm=[0, 3, 1, 2])  # shape (B,D,P,S)
  mean_v = tf.transpose(mean_v, perm=[0, 3, 1, 2])  # shape (B,D,P,S)
  mean_q = tf.transpose(mean_q, perm=[0, 3, 1, 2])  # shape (B,D,P,S)

  epsilon_part= tf.reduce_sum(tf.multiply(mean_z, mean_z), axis=1) + tf.reduce_sum(tf.multiply(mean_k, mean_k), axis=1) + tf.reduce_sum(tf.multiply(mean_v, mean_v), axis=1) + tf.reduce_sum(tf.multiply(mean_q, mean_q), axis=1)
  #det_part=tf.linalg.logdet(sigma) # 2*math.pi removed for easier debugging.
  #ll_one_layer = det_part + epsilon_part
  ll_one_layer = epsilon_part
  return ll_one_layer # (B,P,S)

def compute_SMC_log_likelihood(list_epsilon, list_sigma, sampling_weights):
  '''
  compute the SMC_log_likelihood for the multi_layer case by looping over layers.
  :param list_epsilon: list of epsilons containing the epsilon of every layer.
  :param list_sigma: list of sigmas containing the (same) sigma for each layer.
  :param sampling_weights: shape (B,P,1): w_T of the last_timestep from the SMC Cell/Layer.
  :return:
  A scalar computing log p (z_j / z_(0:j-1), Y_(0:j-1))
  '''
  ll_all_layers=[]
  for epsilon, sigma in zip(list_epsilon, list_sigma):
    ll_one_layer=compute_SMC_ll_one_layer(epsilon, sigma) # shape (B,P,S)
    ll_all_layers.append(ll_one_layer)

  # stacking each loss by layer on a tensor of shape (B,L,P,S)
  total_ll=tf.stack(ll_all_layers, axis=1) # shape (B,L,P,S)

  # multiply by -1/2 and suming over layer dimension:
  total_ll=tf.reduce_sum(tf.scalar_mul(-1/2, total_ll), axis=1) # shape (B,P,S)

  # # mean over the seq_len dim
  # total_ll=tf.reduce_mean(total_ll, axis=-1) # shape (B,P)
  #
  # # weighted sum over particles dim using sampling_weights:
  # if len(tf.shape(sampling_weights)) == 3:
  #   sampling_weights = tf.squeeze(sampling_weights, axis=-1)
  # SMC_loss = tf.reduce_sum(sampling_weights * total_ll, axis=-1)  # dim (B,)
  #
  # # mean over batch dim:
  # SMC_loss=tf.reduce_mean(SMC_loss, axis=0)

  return SMC_loss

if __name__ == "__main__":
  B=8
  P=5
  S=3
  L=4
  D=12
  sigma_scalar=1

  #--------------------------test of compute_ll_one_layer---------------------------------------------------
  sigma_tensor = tf.constant(sigma_scalar, shape=(D,), dtype=tf.float32)
  sigma = tf.Variable(tf.linalg.diag(sigma_tensor), dtype=tf.float32)

  # test for epsilon = zeros tensor
  epsilon = tf.zeros(shape=(B, P, S, D))
  SMC_loss_tensor=compute_SMC_ll_one_layer(epsilon=epsilon, sigma=sigma) # ok, test passed. return a zero tensor :-) when sigma=1 &
  #epsilon is a zero tensor.

  #--------------------------test of compute_log_likelihood------------------------------------------------

  #list_epsilon=[tf.random.normal(shape=(B,P,S,D)) for _ in range(L)]

  # test with all epsilon as zero tensor...
  list_epsilon=[tf.zeros(shape=(B,P,S,D)) for _ in range(L)]
  list_sigma=[sigma for _ in range(L)]

  sampling_weights=tf.random.uniform(shape=(B,P,1))
  # normalization:
  sampling_weights=sampling_weights/tf.expand_dims(tf.reduce_sum(sampling_weights, axis=1), axis=1)

  SMC_loss=compute_SMC_log_likelihood(list_epsilon=list_epsilon, list_sigma=list_sigma, sampling_weights=sampling_weights)

  print(SMC_loss.numpy()) # ok, test passed. return 0 when sigma is the identity matrix and epsilon is a zero tensor.