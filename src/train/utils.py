import tensorflow as tf

def compute_categorical_cross_entropy(targets, preds, attention_mask=None):
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    # resampling_weights = smc_transformer.cell.list_weights[-1] #TODO: add comparison with best particle.
    # best_particles = tf.math.argmax(resampling_weights)
    ce_metric_avg_pred = ce(y_true=targets, y_pred=tf.reduce_mean(preds, axis=1, keepdims=True))  # (B,1,S)
    if attention_mask is not None:
        attention_mask /= tf.reduce_mean(attention_mask)
        ce_metric_avg_pred *= attention_mask
    ce_metric_avg_pred = tf.reduce_mean(ce_metric_avg_pred)
    return ce_metric_avg_pred
