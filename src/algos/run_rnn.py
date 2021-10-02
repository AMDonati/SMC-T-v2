import tensorflow as tf
import os
from src.train.train_functions import train_LSTM
from src.models.Baselines.RNNs import build_LSTM_for_classification
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
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=2) # num_dim = 2 for classification lstm.

        self.lstm = build_LSTM_for_classification(seq_len=None, emb_size=args.d_model,
                                                  shape_output=self.output_size, rnn_units=args.rnn_units,
                                                  dropout_rate=args.p_drop,
                                                  rnn_drop_rate=args.rnn_drop,
                                                  training=True)
        assert self.past_len < self.seq_len, "past_len should be inferior to the sequence length of the dataset"
        self.distribution = True if self.p_drop > 0 else False
        self.future_len = args.future_len if args.future_len is not None else (self.seq_len - self.past_len)
        self._load_ckpt()

    def _create_out_folder(self, args):
        if args.save_path is not None:
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            output_folder = os.path.join(args.save_path, datetime_folder)
        else:
            output_path = args.output_path
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

    def inference_multistep(self, inputs, targets=None, attention_mask=None, past_len=4, future_len=5):
        '''
            :param LSTM_hparams: shape_input_1, shape_input_2, shape_ouput, num_units, dropout_rate
            :param inp_model: array of shape (B,S,F)
            :param mc_samples:
            :return:
        '''
        list_predictions = []
        for i in range(self.mc_samples):
            inp = inputs
            for t in range(future_len + 1):
                preds_test = self.lstm(inputs=inp)  # (B,S,1)
                last_pred = preds_test[:, -1, :]
                last_pred = tf.random.categorical(logits=last_pred, num_samples=1, dtype=tf.int32)
                inp = tf.concat([inp, last_pred], axis=1)
            list_predictions.append(inp)
        preds_test_MC_Dropout = tf.stack(list_predictions, axis=1) # shape (B, mc_samples, seq_len)
        return preds_test_MC_Dropout

    def train(self):
        train_LSTM(model=self.lstm,
                       optimizer=self.optimizer,
                       EPOCHS=self.EPOCHS,
                       train_dataset=self.train_dataset,
                       val_dataset=self.val_dataset,
                       checkpoint_path=self.ckpt_path,
                       output_path=self.out_folder,
                       logger=self.logger,
                       num_train=1)

