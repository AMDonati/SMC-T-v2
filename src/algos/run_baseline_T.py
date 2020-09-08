from utils.utils_train import CustomSchedule
import tensorflow as tf
import os
from models.Baselines.Transformer_without_enc import Transformer
from train.train_functions import train_baseline_transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from algos.run_rnn import Algo
from utils.utils_train import restoring_checkpoint
import datetime

class BaselineTAlgo(Algo):
    def __init__(self, dataset, args):
        super(BaselineTAlgo, self).__init__(dataset=dataset, args=args)
        self.lr = CustomSchedule(args.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self.create_logger()
        self.ckpt_path = self.create_ckpt_path()
        self.save_hparams(args)
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=3)
        self.transformer = Transformer(num_layers=1, d_model=args.d_model, num_heads=1, dff=args.dff,
                                       target_vocab_size=self.output_size,
                                       maximum_position_encoding=args.pe, rate=0, full_model=args.full_model)
        self.start_epoch = 0
        self._load_ckpt(arg=args)

    def _create_out_folder(self, args):
        out_file = 'Classic_T_depth_{}_dff_{}_pe_{}_bs_{}_fullmodel_{}'.format(args.d_model, args.dff, args.pe,
                                                                               self.bs, args.full_model)
        datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        output_folder = os.path.join(args.output_path, out_file, datetime_folder)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return output_folder

    def _load_ckpt(self, args, num_train=1):
        # creating checkpoint manager
        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)
        baseline_T_ckpt_path = os.path.join(self.ckpt_path, "baseline_transformer_{}".format(num_train))
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, baseline_T_ckpt_path, max_to_keep=self.EPOCHS)
        # if a checkpoint exists, restore the latest checkpoint.
        start_epoch = restoring_checkpoint(ckpt_manager=self.ckpt_manager, ckpt=ckpt, args_load_ckpt=True, logger=self.logger)
        if start_epoch is not None:
            self.start_epoch = start_epoch

    def train(self):
        self.logger.info('hparams...')
        self.logger.info(
            'd_model:{} - dff:{} - positional encoding: {} - learning rate: {}'.format(self.transformer.decoder.d_model,
                                                                                       self.transformer.decoder.dff,
                                                                                       self.transformer.decoder.maximum_position_encoding,
        self.lr))
        self.logger.info('Transformer with one head and one layer')
        train_baseline_transformer(transformer=self.transformer,
                                   optimizer=self.optimizer,
                                   EPOCHS=self.EPOCHS,
                                   train_dataset=self.train_dataset,
                                   val_dataset=self.val_dataset,
                                   output_path=self.out_folder,
                                   ckpt_manager=self.ckpt_manager,
                                   logger=self.logger,
                                   start_epoch=self.start_epoch,
                                   num_train=1)

    def test(self):
        for (inp, tar) in self.test_dataset:
            seq_len = tf.shape(inp)[-2]
            predictions_test, _ = self.transformer(inputs=inp,
                                              training=False,
                                              mask=create_look_ahead_mask(seq_len))  # (B,S,F)
            loss_test = tf.keras.losses.MSE(tar, predictions_test)  # (B,S)
            loss_test = tf.reduce_mean(loss_test)
        self.logger.info("test loss at the end of training: {}".format(loss_test))

