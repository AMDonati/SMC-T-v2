import tensorflow as tf
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
import os
import numpy as np
from models.Baselines.Transformer_without_enc import Transformer
from utils.utils_train import create_logger


def MC_Dropout_Transformer(transformer, test_dataset, seq_len, task, stats, num_samples, output_path, logger, inference=True):
  list_predictions = []
  # setting the right dropout rate:
  #transformer.decoder.rate = dropout_rate
  #for layer in (transformer.decoder.dec_layers):
    #layer.rate = dropout_rate

  for i in range(num_samples):
    if inference:
      predictions_test, _ = transformer(inputs=test_dataset,
                                        training=True,
                                        mask=create_look_ahead_mask(seq_len))
    else:
      for (test_data, y_test) in test_dataset:
        X_test = test_data[:,:-1,:]
        predictions_test, _ = transformer(inputs=X_test,
                                        training=True,
                                        mask=create_look_ahead_mask(seq_len))
      # predictions_test shape (B,S,1)
    list_predictions.append(predictions_test)

  predictions_test_MC_Dropout = tf.stack(list_predictions, axis=1) # shape (B, N, S, 1)
  if logger is not None:
    logger.info("saving predictions from MC Dropout on the Baseline Transformer...")

  if output_path is not None:
    eval_output_path = os.path.join(output_path, "eval_outputs_Baseline_T_MC_Dropout")
    if not os.path.isdir(eval_output_path):
      os.makedirs(eval_output_path)
    predictions_MC_Dropout_path = os.path.join(eval_output_path, 'predictions_MC_Dropout.npy')
    np.save(predictions_MC_Dropout_path, predictions_test_MC_Dropout)
    targets_path = os.path.join(eval_output_path, 'targets_test.npy')
    np.save(targets_path, y_test)

  # unormalization of the predictions:
  if task =='unistep-forcst':
    data_mean, data_std = stats
    predictions_unnorm = predictions_test * data_std + data_mean
    targets_unnorm = y_test * data_std + data_mean
    # saving predictions and targets unnorm:
    if output_path is not None:
      predictions_unnorm_path = os.path.join(eval_output_path, 'predictions_unnorm_MC_Dropout.npy')
      targets_unnorm_path = os.path.join(eval_output_path, 'targets_unnorm.npy')
      np.save(predictions_unnorm_path, predictions_unnorm)
      np.save(targets_unnorm_path, targets_unnorm)

  return predictions_test_MC_Dropout

if __name__ == "__main__":
  # create a test dataset
  B = 4000
  S = 25
  num_feat = 3
  num_layers = 1
  d_model = 2
  dff= 8
  num_heads = 1
  target_vocab_size = 1
  maximum_position_encoding_baseline = 50
  data_type = 'time_series_multi'
  rate = 0.1
  seq_len = 24
  num_samples = 5000
  task = 'synthetic'
  stats = None
  output_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/temp'
  logger= create_logger(out_file_log=output_path + 'training_log.log')

  # create the test dataset:
  X_test = tf.random.uniform(shape=(B, S, num_feat))
  y_test = tf.random.uniform(shape=(B, S - 1, 1))
  test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  test_dataset = test_dataset.batch(B, drop_remainder=True)

  # create a transformer model.
  transformer = Transformer(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            target_vocab_size=target_vocab_size,
                            maximum_position_encoding=maximum_position_encoding_baseline,
                            data_type=data_type,
                            rate=rate)
  # MC Dropout on Baseline Transformer:
  MC_Dropout_Transformer(transformer=transformer,
                         test_dataset=test_dataset,
                         seq_len=seq_len,
                         task=task,
                         stats=stats,
                         num_samples=num_samples,
                         output_path=output_path,
                         logger=logger)