import tensorflow as tf
import os
from src.train.train_functions import train_LSTM
from src.models.Baselines.RNNs import build_LSTM_for_regression
from src.algos.generic import Algo
import numpy as np
import datetime

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
        self.save_hparams(args)
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=3)
        self.lstm = build_LSTM_for_regression(shape_input_1=self.seq_len,
                                              shape_input_2=self.num_features,
                                              shape_output=self.output_size,
                                              rnn_units=args.rnn_units,
                                              dropout_rate=args.p_drop,
                                              rnn_drop_rate=args.rnn_drop,
                                              training=True)
        assert self.past_len < self.seq_len, "past_len should be inferior to the sequence length of the dataset"
        self.distribution = True if self.p_drop > 0 else False
        self.future_len = args.future_len if args.future_len is not None else (self.seq_len - self.past_len)
        self._load_ckpt()

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            output_path = args.output_path
            #out_file = '{}_LSTM_units_{}_pdrop_{}_rnndrop_{}_lr_{}_bs_{}_cv_{}'.format(args.dataset, args.rnn_units,
                                                                                       #args.p_drop,
                                                                                       #args.rnn_drop, self.lr,
                                                                                       #self.bs, args.cv)
            out_file = '{}_d{}_p{}'.format(args.algo, args.rnn_units, args.p_drop)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            output_folder = os.path.join(output_path, out_file, datetime_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            return output_folder

    def _load_ckpt(self, num_train=1):
        ckpt_path = os.path.join(self.ckpt_path, "RNN_Baseline_{}".format(num_train))
        if os.path.isdir(ckpt_path):
            latest = tf.train.latest_checkpoint(ckpt_path)
            self.logger.info("loading latest checkpoint from {}".format(latest))
            self.lstm.load_weights(latest)

    def _MC_Dropout(self, inp_model, save_path=None):
        '''
        :param LSTM_hparams: shape_input_1, shape_input_2, shape_ouput, num_units, dropout_rate
        :param inp_model: array of shape (B,S,F)
        :param mc_samples:
        :return:
        '''
        list_predictions = []
        for i in range(self.mc_samples):
            predictions_test = self.lstm(inputs=inp_model)  # (B,S,1)
            list_predictions.append(predictions_test)
        predictions_test_MC_Dropout = tf.stack(list_predictions, axis=1)  # shape (B, N, S, F)
        print('MC Dropout unistep done')
        if save_path is not None:
            np.save(save_path, predictions_test_MC_Dropout.numpy())
        return predictions_test_MC_Dropout

    def stochastic_forward_pass_multistep(self, inp_model, future_inp_features=None, save_path=None):
        '''
            :param LSTM_hparams: shape_input_1, shape_input_2, shape_ouput, num_units, dropout_rate
            :param inp_model: array of shape (B,S,F)
            :param mc_samples:
            :return:
        '''
        list_predictions = []
        for i in range(self.mc_samples):
            inp = inp_model
            for t in range(self.future_len + 1):
                preds_test = self.lstm(inputs=inp)  # (B,S,1)
                last_pred = preds_test[:, -1, :]
                if t < self.future_len:
                    if future_inp_features is not None:
                        last_pred = tf.concat([last_pred, future_inp_features[:, t, :]], axis=-1)
                    last_pred = tf.expand_dims(last_pred, axis=-2)
                    inp = tf.concat([inp, last_pred], axis=1)
            list_predictions.append(preds_test[:,self.past_len:, :])
        preds_test_MC_Dropout = tf.stack(list_predictions, axis=1)
        print('mc dropout LSTM multistep done')
        if save_path is not None:
            np.save(save_path, preds_test_MC_Dropout.numpy())
        return preds_test_MC_Dropout

    def train(self):
        if not self.cv:
            train_LSTM(model=self.lstm,
                       optimizer=self.optimizer,
                       EPOCHS=self.EPOCHS,
                       train_dataset=self.train_dataset,
                       val_dataset=self.val_dataset,
                       checkpoint_path=self.ckpt_path,
                       output_path=self.out_folder,
                       logger=self.logger,
                       num_train=1)
        else:
            for num_train, (train_dataset, val_dataset) in enumerate(zip(self.train_dataset, self.val_dataset)):
                train_LSTM(model=self.lstm,
                           optimizer=self.optimizer,
                           EPOCHS=self.EPOCHS,
                           train_dataset=train_dataset,
                           val_dataset=val_dataset,
                           checkpoint_path=self.ckpt_path,
                           output_path=self.out_folder,
                           logger=self.logger,
                           num_train=num_train + 1)
                self.logger.info(
                    "training of a LSTM for train/val split number {} done...".format(num_train + 1))
                self.logger.info('-' * 60)

    def compute_test_loss(self, save_particles=True):
        TEST_LOSS, PREDS_TEST = [], []
        for inp, tar in self.test_dataset:
            test_preds = self.lstm(inp)
            test_loss = tf.keras.losses.MSE(test_preds, tar)
            test_loss = tf.reduce_mean(test_loss)
            TEST_LOSS.append(test_loss)
            PREDS_TEST.append(test_preds)
        PREDS_TEST = tf.stack(PREDS_TEST, axis=0)
        return np.mean(TEST_LOSS), PREDS_TEST
