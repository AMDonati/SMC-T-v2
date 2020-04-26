import tensorflow as tf
from train.train_SMC_Transformer_dummy_dataset import categorical_crossentropy

# --- test of categorical_crossentropy function---

if __name__ == "__main__":

  real = tf.ones(shape=(8, 10, 20))
  pred = tf.random.uniform(shape=(8, 10, 20, 800), maxval=1, dtype=tf.float32)
  sampling_weights = tf.random.uniform(shape=(8, 10), maxval=1, dtype=tf.float32)

  loss=categorical_crossentropy(real, pred, sampling_weights)

  print(loss)

# -- test of the update covariance_matrix function.

def compute_direct_update_cov_matrix(list_stddev, list_sigmas):
  '''
  for each layer, update the sigma of the reparametrized noise with the optimal sigma of the current training step.
  '''
  #num_layers=len(list_stddev)
  seq_length=tf.shape(list_stddev[0])[2]
  for l, stddev, sigma in enumerate(zip(list_stddev, list_sigmas)):
    sigma_optimal_l=[]
    for t in range(seq_length):
      stddev_t=stddev[:,:,t,:]
      # permute P and D dims:
      stddev_t=tf.transpose(stddev_t, perm=[0,2,1]) # dim (B,D,P)
      # do double dot product to have on the particle and batch dims to have at the end a tensor of dim (D,D)
      sigma_optimal_l_t=tf.tensordot(stddev_t, tf.linalg.matrix_transpose(stddev_t), axes=[[0,2],[0,1]]) # dim (D,D)
      sigma_optimal_l.append(sigma_optimal_l_t)
    sigma_optimal_l=tf.stack(sigma_optimal_l, axis=0) # dim (S,D,D)
    # sum over all the layers
    sigma_optimal_l=tf.reduce_sum(sigma_optimal_l, axis=0) # dim (D,D)
    # multiply by 1/seq_len
    sigma_optimal_l=tf.math.scalar_mul(1/seq_length, sigma_optimal_l)

    sigma.assign(sigma_optimal_l)
    #self.decoder.dec_layers[l].mha1.sigma.assign(sigma_optimal_l) # real line for function inside the transformer.

  return list_sigmas

