import tensorflow as tf
import numpy as np
import os, argparse
from preprocessing.time_series.df_to_dataset_synthetic import split_synthetic_dataset, data_to_dataset_3D, \
  split_input_target
from models.Baselines.Transformer_without_enc import Transformer
from models.Baselines.SMC_on_classic_transformer import SMC_on_Transformer
from train.loss_functions import CustomSchedule
from train.train_functions import train_baseline_transformer
from utils.utils_train import create_logger, restoring_checkpoint
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask


if __name__ == '__main__':

  #trick for boolean parser args.
  def str2bool(v):
    if isinstance(v, bool):
      return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
    else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

  parser = argparse.ArgumentParser()

  parser.add_argument("-d_model", type=int, required=True, help="depth of attention parameters")
  parser.add_argument("-bs", type=int, default=128, help="batch size")
  parser.add_argument("-ep", type=int, default=20, help="number of epochs")
  parser.add_argument("-dff", type=int, default=32, help="dimension of feed-forward network")
  parser.add_argument("-pe", type=int, default=50, help="maximum positional encoding")
  parser.add_argument("-full_model", type=str2bool, default=False, help="full_model = ffn & layernorm")
  parser.add_argument("-data_path", type=str, required=True, help="path for saving data")
  parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
  parser.add_argument("-launch_smc", type=str2bool, default=False, help="launching SMC algo after training")

  args = parser.parse_args()

  # ------------------- Upload synthetic dataset ----------------------------------------------------------------------------------
  BUFFER_SIZE = 500
  BATCH_SIZE = args.bs
  TRAIN_SPLIT = 0.7

  data_path = os.path.join(args.data_path, 'synthetic_dataset_1_feat.npy')
  input_data = np.load(data_path)
  train_data, val_data, test_data = split_synthetic_dataset(x_data=input_data, TRAIN_SPLIT=TRAIN_SPLIT, cv=False)

  val_data_path = os.path.join(args.data_path, 'val_data_synthetic_1_feat.npy')
  train_data_path = os.path.join(args.data_path, 'train_data_synthetic_1_feat.npy')
  test_data_path = os.path.join(args.data_path, 'test_data_synthetic_1_feat.npy')

  np.save(val_data_path, val_data)
  np.save(train_data_path, train_data)
  np.save(test_data_path, test_data)

  train_dataset, val_dataset, test_dataset = data_to_dataset_3D(train_data=train_data,
                                                                val_data=val_data,
                                                                test_data=test_data,
                                                                split_fn=split_input_target,
                                                                BUFFER_SIZE=BUFFER_SIZE,
                                                                BATCH_SIZE=BATCH_SIZE,
                                                                target_feature=None,
                                                                cv=False)

  # -------------------- define hyperparameters -----------------------------------------------------------------------------------
  d_model = args.d_model
  dff = args.dff
  maximum_position_encoding = args.pe
  EPOCHS = args.ep
  target_vocab_size = 1

  # define optimizer
  learning_rate = CustomSchedule(d_model)
  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
  output_path = args.output_path
  out_file = 'Classic_T_depth_{}_dff_{}_pe_{}_bs_{}_fullmodel_{}'.format(d_model, dff, maximum_position_encoding, BATCH_SIZE, args.full_model)
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

  # --------------------- Training of the Transformer --------------------------------------------------------------------------------------------------------------
  logger.info('hparams...')
  logger.info('d_model:{} - dff:{} - positional encoding: {} - learning rate: {}'.format(d_model, dff, maximum_position_encoding, learning_rate))
  logger.info('Transformer with one head and one layer')

  transformer = Transformer(num_layers=1, d_model=d_model, num_heads=1, dff=dff, target_vocab_size=target_vocab_size,
                            maximum_position_encoding=maximum_position_encoding, rate=0, full_model=args.full_model)

  train_baseline_transformer(transformer=transformer,
                             optimizer=optimizer,
                             EPOCHS=EPOCHS,
                             train_dataset=train_dataset,
                             val_dataset=val_dataset,
                             output_path=output_path,
                             checkpoint_path=checkpoint_path,
                             logger=logger,
                             num_train=1)

  # computing loss on test dataset
  for (inp, tar) in test_dataset:
    seq_len = tf.shape(inp)[-2]
    predictions_test, _ = transformer(inputs=inp,
                                      training=False,
                                      mask=create_look_ahead_mask(seq_len)) # (B,S,F)
    loss_test = tf.keras.losses.MSE(tar, predictions_test) # (B,S)
    loss_test = tf.reduce_mean(loss_test)

  logger.info("test loss at the end of training: {}".format(loss_test))


  # # ------------------- SMC algo on pre-trained Transformer -------------------------------------------------------------------------------------------------------------
  # if args.launch_smc:
  #   # load checkpoint:
  #   ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
  #   ckpt_path = os.path.join(checkpoint_path, "transformer_baseline_1")
  #   ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=EPOCHS)
  #   restoring_checkpoint(ckpt_manager=ckpt_manager, ckpt=ckpt, logger=logger)
  #
  #   logger.info("<--------------------------------------------------------------------------------------->")
  #   logger.info('starting SMC algo on trained Transformer...')
  #
  #   list_sigma = [0.05, 0.1, 0.5]
  #   list_params = ['k', 'q', 'v', 'z']
  #
  #   list_sigma_obs = [0.5, 0.7, 0.9]
  #   list_num_particles = [10]
  #
  #   for sigma, sigma_obs in zip(list_sigma, list_sigma_obs):
  #     dict_sigmas = dict(zip(list_params, [sigma for _ in range(4)]))
  #     logger.info('SMC params: sigma_k_q_v_z: {} - sigma_obs:{}'.format(sigma, sigma_obs))
  #
  #     for num_particles in list_num_particles:
  #       logger.info('SMC for number of particles: {}'.format(num_particles))
  #       loss_smc_train, loss_smc_val = 0, 0
  #
  #       for (batch, (inp, tar)) in enumerate(train_dataset):
  #         seq_len = tf.shape(inp)[-2]
  #         # check loss for normal passforward procedure
  #         if batch == 0:
  #           predictions, _ = transformer(inputs=inp, training=False, mask=create_look_ahead_mask(seq_len))
  #           loss_normal = tf.keras.losses.MSE(tar, predictions)
  #           loss_normal = tf.reduce_mean(loss_normal, axis=-1)
  #           loss_normal = tf.reduce_mean(loss_normal, axis=-1)
  #           logger.info('normal loss on first batch...{}'.format(loss_normal.numpy()))
  #         inp_p = tf.expand_dims(inp, axis=1)
  #         inp_p = tf.tile(inp_p, multiples=[1, num_particles, 1, 1]) # (B,P,S,D)
  #         tar_p = tf.expand_dims(tar, axis=1)
  #         tar_p = tf.tile(tar_p, multiples=[1, num_particles, 1, 1])
  #         list_inputs = [tf.expand_dims(inp_p[:,:,t,:], axis=-2) for t in range(seq_len)]
  #         list_targets = [tf.expand_dims(tar_p[:,:,t,:], axis=-2) for t in range(seq_len)]
  #         predictions_smc, K, V = SMC_on_Transformer(transformer=transformer,
  #                                                dict_sigmas=dict_sigmas,
  #                                                sigma_obs=sigma_obs,
  #                                                list_inputs=list_inputs,
  #                                                list_targets=list_targets) # (B,P,S,F)
  #         mean_prediction = tf.reduce_mean(predictions_smc, axis=1) # (B,S,F)
  #         loss_smc_train_batch = tf.keras.losses.MSE(tar, mean_prediction)
  #         loss_smc_train_batch = tf.reduce_mean(loss_smc_train_batch, axis=-1)
  #         loss_smc_train_batch = tf.reduce_mean(loss_smc_train_batch, axis=-1)
  #         loss_smc_train += loss_smc_train_batch
  #         # # taking best pred:
  #         # loss_smc_p = tf.keras.losses.MSE(tar_p, predictions_smc)
  #         # min_smc_loss = tf.reduce_min(loss_smc_p, axis=1)
  #         # min_smc_loss = tf.reduce_mean(min_smc_loss, axis=-1)
  #         # min_smc_loss = tf.reduce_mean(min_smc_loss, axis=-1)
  #         logger.info('smc train loss at batch {}: {}'.format(batch+1, loss_smc_train_batch))
  #
  #
  #       logger.info('<----------------------------------------------------->')
  #
  #       transformer.stop_SMC_algo() # putting self.noise = False in the self_attention mechanism.
  #
  #       for (batch, (inp, tar)) in enumerate(val_dataset):
  #         seq_len = tf.shape(inp)[-2]
  #         inp_p = tf.expand_dims(inp, axis=1)
  #         inp_p = tf.tile(inp_p, multiples=[1, num_particles, 1, 1])  # (B,P,S,D)
  #         tar_p = tf.expand_dims(tar, axis=1)
  #         tar_p = tf.tile(tar_p, multiples=[1, num_particles, 1, 1])
  #         list_inputs = [tf.expand_dims(inp_p[:, :, t, :], axis=-2) for t in range(seq_len)]
  #         list_targets = [tf.expand_dims(tar_p[:, :, t, :], axis=-2) for t in range(seq_len)]
  #         predictions_smc, K, V = SMC_on_Transformer(transformer=transformer,
  #                                                    dict_sigmas=dict_sigmas,
  #                                                    sigma_obs=sigma_obs,
  #                                                    list_inputs=list_inputs,
  #                                                    list_targets=list_targets)  # (B,P,S,F)
  #         mean_prediction = tf.reduce_mean(predictions_smc, axis=1)  # (B,S,F)
  #         loss_smc_val_batch = tf.keras.losses.MSE(tar, mean_prediction)
  #         loss_smc_val_batch = tf.reduce_mean(loss_smc_val_batch, axis=-1)
  #         loss_smc_val_batch = tf.reduce_mean(loss_smc_val_batch, axis=-1)
  #         loss_smc_val += loss_smc_val_batch
  #         logger.info('smc val loss at batch {}: {}'.format(batch + 1, loss_smc_val_batch))
  #         # # taking best pred:
  #         # loss_smc_p = tf.keras.losses.MSE(tar_p, predictions_smc)
  #         # min_smc_loss = tf.reduce_min(loss_smc_p, axis=1)
  #         # min_smc_loss = tf.reduce_mean(min_smc_loss, axis=-1)
  #         # min_smc_loss = tf.reduce_mean(min_smc_loss, axis=-1)
  #         logger.info('smc train loss at batch {}: {}'.format(batch + 1, loss_smc_train_batch))
  #
  #       transformer.stop_SMC_algo()
  #
  #       logger.info("<--------------------------------------------------------------------------->")
  #     logger.info("<----------------------------------------------------------------------------------------------------->")
