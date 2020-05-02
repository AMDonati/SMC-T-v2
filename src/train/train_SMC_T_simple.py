import tensorflow as tf
import numpy as np
import os, argparse
from preprocessing.time_series.df_to_dataset import split_input_target, data_to_dataset_4D, split_synthetic_dataset
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from train.train_functions import train_SMC_transformer
from utils.utils_train import create_logger, restoring_checkpoint
from train.loss_functions import CustomSchedule

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("-d_model", type=int, required=True, default=2, help="depth of attention parameters")
  parser.add_argument("-bs", type=int, required=True, default=128, help="batch size")
  parser.add_argument("-ep", type=int, default=20, help="number of epochs")
  parser.add_argument("-data_path", type=str, default="../../data", help="path for saving data")
  parser.add_argument("-output_path", type=str, default="../../output", help="path for output folder")

  args = parser.parse_args()

  # -------------------------------- Upload synthetic dataset ----------------------------------------------------------------------------------

  BUFFER_SIZE = 500
  BATCH_SIZE = args.bs
  TRAIN_SPLIT = 0.7

  data_path = os.path.join(args.data_path, 'synthetic_dataset_1_feat.npy')
  input_data = np.load(data_path)

  train_data, val_data, test_data = split_synthetic_dataset(x_data=input_data,
                                                            TRAIN_SPLIT=TRAIN_SPLIT,
                                                            cv=False)

  val_data_path = os.path.join(args.data_path,'val_data_synthetic_1_feat.npy')
  train_data_path = os.path.join(args.data_path, 'train_data_synthetic_1_feat.npy')
  test_data_path = os.path.join(args.data_path, 'test_data_synthetic_1_feat.npy')

  np.save(val_data_path, val_data)
  np.save(train_data_path, train_data)
  np.save(test_data_path, test_data)

  train_dataset, val_dataset, test_dataset = data_to_dataset_4D(train_data=train_data,
                                                                val_data=val_data,
                                                                test_data=test_data,
                                                                split_fn=split_input_target,
                                                                BUFFER_SIZE=BUFFER_SIZE,
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                target_feature=None,
                                                                cv=False)

  # ------------------------------ Define hyperparameters ------------------------------------------------------------------------------------------

  d_model = args.d_model
  EPOCHS = args.ep
  for (_, tar) in train_dataset.take(1):
    output_size = tf.shape(tar)[-1]
  seq_len = train_data.shape[1] - 1 # 24.

  # define optimizer
  learning_rate = CustomSchedule(d_model)
  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
  output_path = args.output_path
  out_file = 'Recurrent_T_depth_{}_bs_{}'.format(d_model, BATCH_SIZE)

  output_path = os.path.join(output_path, out_file)
  if not os.path.isdir(output_path):
    os.makedirs(output_path)

  # -------------------- create logger and checkpoint saver ----------------------------------------------------------------------------------------------------

  out_file_log = output_path + '/' + 'training_log.log'
  logger = create_logger(out_file_log=out_file_log)
  #  creating the checkpoint manager:
  checkpoint_path = os.path.join(output_path, "checkpoints")
  if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

  # ------ Training of the recurrent Transformer ---------------------------------------------------------------------------------------------------
  logger.info('hparams...')
  logger.info('d_model: {} - batch size {}'.format(d_model, BATCH_SIZE))

  smc_transformer = SMC_Transformer(d_model=d_model,
                                    output_size=output_size,
                                    seq_len=seq_len)

  train_SMC_transformer(smc_transformer=smc_transformer,
                        optimizer=optimizer,
                        EPOCHS=EPOCHS,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        checkpoint_path=checkpoint_path,
                        logger=logger,
                        num_train=1)

  # computing loss on test_dataset:
  for (inp, tar) in test_dataset:
    predictions_test, _, _ = smc_transformer(inputs=inp, targets=tar)
    test_loss = tf.keras.losses.MSE(tar, predictions_test)
    test_loss = tf.reduce_mean(test_loss, axis=-1)
    test_loss = tf.reduce_mean(test_loss, axis=-1)
    test_loss = tf.reduce_mean(test_loss, axis=-1)

  logger.info("test loss at the end of training: {}".format(test_loss))