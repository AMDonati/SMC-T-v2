import tensorflow as tf

def compute_categorical_cross_entropy(targets, preds, num_particles, attention_mask=None):
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    # resampling_weights = smc_transformer.cell.list_weights[-1] #TODO: add comparison with best particle.
    # best_particles = tf.math.argmax(resampling_weights)
    ce_metric_avg_pred = ce(y_true=targets, y_pred=tf.reduce_mean(preds, axis=1, keepdims=True))  # (B,1,S)
    if attention_mask is not None:
        attn_mask = tf.tile(tf.expand_dims(attention_mask, axis=1), multiples=[1, num_particles, 1])
        attn_mask = tf.cast(attn_mask, dtype=tf.float32)
        ce_metric_avg_pred = ce_metric_avg_pred * attn_mask
    ce_metric_avg_pred = tf.reduce_mean(ce_metric_avg_pred)
    return ce_metric_avg_pred
