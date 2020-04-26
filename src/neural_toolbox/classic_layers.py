import tensorflow as tf

def softmax_layer(logits, labels, num_labels, mask):
  logits = tf.reshape(logits, [-1, num_labels])
  labels = tf.reshape(labels, [-1])
  mask = tf.cast(mask, dtype=tf.float32)
  one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
  loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
  loss *= tf.reshape(mask, [-1])
  loss = tf.reduce_sum(loss)
  total_size = tf.reduce_sum(mask)
  total_size += 1e-12  # to avoid division by 0 for all-0 weights
  loss /= total_size
  # predict not mask we could filtered it in the prediction part.
  probabilities = tf.math.softmax(logits, axis=-1)
  predict = tf.math.argmax(probabilities, axis=-1)
  return loss, predict

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', name='FFN1_after_mha_dff'),  # (B, S, dff)
      tf.keras.layers.Dense(d_model, name='FFN2_after_mha_dmodel')  # (B, P, S, d_model)
  ])

