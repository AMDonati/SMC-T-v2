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
        self._load_ckpt()

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            output_path = args.output_path
            out_file = '{}_LSTM_units_{}_pdrop_{}_rnndrop_{}_lr_{}_bs_{}_cv_{}'.format(args.dataset, args.rnn_units,
                                                                                       args.p_drop,
                                                                                       args.rnn_drop, self.lr,
                                                                                       self.bs, args.cv)
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

    def _MC_Dropout_LSTM(self, inp_model, save_path=None):
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
            np.save(save_path, predictions_test_MC_Dropout)
        return predictions_test_MC_Dropout

    def _MC_Dropout_LSTM_multistep(self, inp_model, len_future=None, save_path=None):
        '''
            :param LSTM_hparams: shape_input_1, shape_input_2, shape_ouput, num_units, dropout_rate
            :param inp_model: array of shape (B,S,F)
            :param mc_samples:
            :return:
        '''
        if len_future is None:
            len_future = self.seq_len - self.past_len
        list_predictions = []
        for i in range(self.mc_samples):
            inp = inp_model
            for t in range(len_future + 1):
                preds_test = self.lstm(inputs=inp)  # (B,S,1)
                last_pred = tf.expand_dims(preds_test[:, -1, :], axis=-2)
                inp = tf.concat([inp, last_pred], axis=1)
            list_predictions.append(preds_test)
        preds_test_MC_Dropout = tf.stack(list_predictions, axis=1)
        print('mc dropout LSTM multistep done')
        if save_path is not None:
            np.save(save_path, preds_test_MC_Dropout)
        return preds_test_MC_Dropout

    def launch_inference(self, **kwargs):
        mc_dropout_unistep_path, mc_dropout_multistep_path = self._get_inference_paths()
        if self.test_predictive_distribution is None:
            self.get_predictive_distribution()
        np.save(mc_dropout_unistep_path, self.test_predictive_distribution)
        print("mc dropout samples unistep shape", self.test_predictive_distribution.shape)

        if kwargs["multistep"]:
            for (inputs, _) in self.test_dataset:
                mc_samples_multi = self._MC_Dropout_LSTM_multistep(inp_model=inputs[:, :self.past_len, :], save_path=mc_dropout_multistep_path)
                print("mc dropout samples multistep shape", mc_samples_multi.shape)

    def get_predictive_distribution(self):
        if self.distribution:
            for (inp, _) in self.test_dataset:
                mc_samples_uni = self._MC_Dropout_LSTM(inp_model=inp, save_path=None)
                print("shape of predictive distribution", mc_samples_uni.shape)
            self.test_predictive_distribution = mc_samples_uni

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
        for inp, tar in self.test_dataset:
            test_preds = self.lstm(inp)
            test_loss = tf.keras.losses.MSE(test_preds, tar)
            test_loss = tf.reduce_mean(test_loss)
        return test_loss, test_preds
