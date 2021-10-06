import tensorflow as tf
from src.models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from src.train.utils import compute_categorical_cross_entropy

# -------------------------------- TRAIN STEP FUNCTIONS ---------------------------------------------------------------------
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


# @tf.function(input_signature=train_step_signature) #TODO: debug this problem
def train_step_classic_T(inputs, targets, transformer, optimizer):
    '''training step for the classic Transformer model'''
    seq_len = tf.shape(inputs)[-2]
    mask_transformer = create_look_ahead_mask(seq_len)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inputs=inputs, training=True, mask=mask_transformer)
        loss = tf.keras.losses.MSE(targets, predictions)
        loss = tf.reduce_mean(loss)  # averaging loss over the seq and batch dims.
        gradients = tape.gradient(loss, transformer.trainable_variables)

    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss


# --------------SMC Transformer train_step-----------------------------------------------------------------------------------------------------
# @tf.function(input_signature=train_step_signature)
def train_step_SMC_T(inputs, targets, smc_transformer, optimizer, it, attention_mask=None):
    '''
    :param it:
    :param inputs:
    :param targets:
    :param smc_transformer:
    :param optimizer:
    :return:
    '''

    #assert len(tf.shape(inputs)) == len(tf.shape(targets)) == 4

    with tf.GradientTape() as tape:
        (preds, preds_resampl), _, _ = smc_transformer(inputs=inputs,
                                                       targets=targets,
                                                       attention_mask=attention_mask)  # predictions: shape (B,P,S,F_y) with P=1 during training.

        smc_loss, classic_loss = smc_transformer.compute_SMC_loss(predictions=preds_resampl, targets=targets, attention_mask=attention_mask)
        loss = smc_loss
        ce_metric_avg_pred = compute_categorical_cross_entropy(targets=targets, preds=preds, num_particles=smc_transformer.cell.num_particles, attention_mask=attention_mask)

        gradients = tape.gradient(loss, smc_transformer.trainable_variables)

        # To debug the loss.
        # trainable_variables = list(smc_transformer.trainable_variables)
        # trainable_variables_names = [t.name for t in trainable_variables]
        # var_and_grad_dict = dict(zip(trainable_variables_names, gradients))
        #print('dict of variables and associated gradients', var_and_grad_dict)

    optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))

    if smc_transformer.cell.noise:
        return loss, ce_metric_avg_pred
    else:
        return loss, None

