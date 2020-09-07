from utils.utils_train import create_logger, CustomSchedule
import tensorflow as tf
import os
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.Baselines.Transformer_without_enc import Transformer
from train.train_functions import train_SMC_transformer, train_LSTM, train_baseline_transformer
from models.Baselines.RNNs import build_LSTM_for_regression
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask


class Algo:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.bs = args.bs
        self.EPOCHS = args.ep
        self.output_path = args.output_path
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        self.out_folder = args.output_path

    def train(self):
        pass

    def test(self):
        pass

    def create_logger(self):
        out_file_log = os.path.join(self.out_folder, 'training_log.log')
        logger = create_logger(out_file_log=out_file_log)
        return logger

    def create_ckpt_path(self):
        checkpoint_path = os.path.join(self.out_folder, "checkpoints")
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        return checkpoint_path

    def load_datasets(self, num_dim=4, target_feature=None, cv=False):
        train_data, val_data, test_data = self.dataset.get_datasets()
        self.seq_len = train_data.shape[1] - 1
        self.logger.info('num samples in training dataset:{}'.format(train_data.shape[0]))
        train_dataset, val_dataset, test_dataset = self.dataset.data_to_dataset(train_data=train_data,
                                                                                val_data=val_data,
                                                                                test_data=test_data,
                                                                                target_feature=target_feature, cv=cv,
                                                                                num_dim=num_dim)
        for (inp, tar) in train_dataset.take(1):
            self.output_size = tf.shape(tar)[-1].numpy()
            self.num_features = tf.shape(inp)[-1].numpy()

        return train_dataset, val_dataset, test_dataset


class SMCTAlgo(Algo):
    def __init__(self, dataset, args):
        super(SMCTAlgo, self).__init__(dataset=dataset, args=args)
        self.lr = CustomSchedule(args.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self.create_logger()
        self.ckpt_path = self.create_ckpt_path()
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=4)
        self.smc_transformer = SMC_Transformer(d_model=args.d_model,
                                               output_size=self.output_size,
                                               seq_len=self.seq_len,
                                               full_model=args.full_model,
                                               dff=args.dff,
                                               attn_window=args.attn_w)
        self._init_SMC_T(args=args)

    def _create_out_folder(self, args):
        # TODO:add date & time.
        out_file = '{}_Recurrent_T_depth_{}_bs_{}_fullmodel_{}_dff_{}_attn_w_{}'.format(args.dataset, args.d_model,
                                                                                        self.bs, args.full_model,
                                                                                        args.dff, args.attn_w)
        if args.smc:
            out_file = out_file + '__p_{}'.format(args.particles)
            out_file = out_file + '_SigmaObs_{}'.format(args.sigma_obs)
            out_file = out_file + '_sigmas_{}'.format(args.sigmas)

        out_folder = os.path.join(self.output_path, out_file)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        return out_folder

    def _init_SMC_T(self, args):
        if args.smc:
            self.logger.info("SMC Transformer for {} particles".format(args.particles))
            if args.sigmas is not None:
                dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [args.sigmas for _ in range(4)]))
            else:
                dict_sigmas = None
            self.smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas, sigma_obs=args.sigma_obs,
                                                         num_particles=args.particles)  # TODO: check if dict_sigmas is saved.
            assert self.smc_transformer.cell.noise == self.smc_transformer.cell.attention_smc.noise == True
            self.logger.info("Sigma_obs init: {}".format(self.smc_transformer.cell.Sigma_obs))

    def train(self):
        self.logger.info('hparams...')
        self.logger.info(
            'd_model: {} - batch size {} - full model? {} - dff: {} -attn window: {}'.format(
                self.smc_transformer.d_model, self.bs,
                self.smc_transformer.full_model, self.smc_transformer.dff,
                self.smc_transformer.cell.attention_smc.attn_window))
        train_SMC_transformer(smc_transformer=self.smc_transformer,
                              optimizer=self.optimizer,
                              EPOCHS=self.EPOCHS,
                              train_dataset=self.train_dataset,
                              val_dataset=self.val_dataset,
                              checkpoint_path=self.ckpt_path,
                              logger=self.logger,
                              num_train=1)

    def test(self):
        self.logger.info("computing test mse metric at the end of training...")
        # computing loss on test_dataset:
        for (inp, tar) in self.test_dataset:
            (preds_test, preds_test_resampl), _, _ = self.smc_transformer(inputs=inp,
                                                                          targets=tar)  # predictions test are the ones not resampled.
            test_metric_avg_pred = tf.keras.losses.MSE(tar,
                                                       tf.reduce_mean(preds_test, axis=1, keepdims=True))  # (B,1,S)
            test_metric_avg_pred = tf.reduce_mean(test_metric_avg_pred)

        self.logger.info("test mse metric from avg particle: {}".format(test_metric_avg_pred))


class RNNAlgo(Algo):
    def __init__(self, dataset, args):
        super(RNNAlgo, self).__init__(dataset=dataset, args=args)
        self.lr = args.lr
        self.optimizer = tf.keras.optimizers.Adam(self.lr,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
        self.p_drop = args.p_drop
        self.rnn_drop = args.rnn_drop
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self.create_logger()
        self.ckpt_path = self.create_ckpt_path()
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=3)
        self.lstm = build_LSTM_for_regression(shape_input_1=self.seq_len,
                                              shape_input_2=self.num_features,
                                              shape_output=self.output_size,
                                              rnn_units=args.rnn_units,
                                              dropout_rate=args.p_drop,
                                              rnn_drop_rate=args.rnn_drop,
                                              training=True)
        self.cv = args.cv

    def _create_out_folder(self, args):
        output_path = args.output_path
        out_file = '{}_LSTM_units_{}_pdrop_{}_rnndrop_{}_lr_{}_bs_{}_cv_{}'.format(args.dataset, args.rnn_units,
                                                                                   args.p_drop,
                                                                                   args.rnn_drop, self.lr,
                                                                                   self.bs, args.cv)
        output_folder = os.path.join(output_path, out_file)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return output_folder

    def train(self):
        if not self.cv:
            train_LSTM(model=self.lstm,
                       optimizer=self.optimizer,
                       EPOCHS=self.EPOCHS,
                       train_dataset=self.train_dataset,
                       val_dataset=self.val_dataset,
                       checkpoint_path=self.checkpoint_path,
                       output_path=self.output_path,
                       logger=self.logger,
                       num_train=1)
        else:
            raise NotImplementedError("cross-validation training not Implemented.")

    def test(self):
        for inp, tar in self.test_dataset:
            test_preds = self.lstm(inp)
            test_loss = tf.keras.losses.MSE(test_preds, tar)
            test_loss = tf.reduce_mean(test_loss)
        self.logger.info("test loss: {}".format(test_loss))


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
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=3)
        self.transformer = Transformer(num_layers=1, d_model=args.d_model, num_heads=1, dff=args.dff,
                                       target_vocab_size=self.output_size,
                                       maximum_position_encoding=args.pe, rate=0, full_model=args.full_model)

    def _create_out_folder(self, args):
        out_file = 'Classic_T_depth_{}_dff_{}_pe_{}_bs_{}_fullmodel_{}'.format(args.d_model, args.dff, args.pe,
                                                                               self.bs, args.full_model)
        output_folder = os.path.join(args.output_path, out_file)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return output_folder

    def train(self):
        self.logger.info('hparams...')
        self.logger.info(
            'd_model:{} - dff:{} - positional encoding: {} - learning rate: {}'.format(self.transformer.d_model,
                                                                                       self.transformer.dff,
                                                                                       self.transformer.decoder.maximum_position_encoding,
        self.lr))
        self.logger.info('Transformer with one head and one layer')
        train_baseline_transformer(transformer=self.transformer,
                                   optimizer=self.optimizer,
                                   EPOCHS=self.EPOCHS,
                                   train_dataset=self.train_dataset,
                                   val_dataset=self.val_dataset,
                                   output_path=self.output_path,
                                   checkpoint_path=self.checkpoint_path,
                                   logger=self.logger,
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


algos = {"smc_t": SMCTAlgo, "lstm": RNNAlgo, "baseline_t": BaselineTAlgo}
