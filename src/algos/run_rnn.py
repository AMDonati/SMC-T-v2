from utils.utils_train import create_logger, restoring_checkpoint
import tensorflow as tf
import os
from train.train_functions import train_LSTM
from models.Baselines.RNNs import build_LSTM_for_regression
from algos.generic import Algo


# class Algo:
#     def __init__(self, dataset, args):
#         self.dataset = dataset
#         self.bs = args.bs
#         self.EPOCHS = args.ep
#         self.output_path = args.output_path
#         self.save_path = args.save_path
#         if not os.path.isdir(self.output_path):
#             os.makedirs(self.output_path)
#         self.out_folder = args.output_path
#
#     def train(self):
#         pass
#
#     def test(self):
#         pass
#
#     def create_logger(self):
#         out_file_log = os.path.join(self.out_folder, 'training_log.log')
#         logger = create_logger(out_file_log=out_file_log)
#         return logger
#
#     def create_ckpt_path(self):
#         checkpoint_path = os.path.join(self.out_folder, "checkpoints")
#         if not os.path.isdir(checkpoint_path):
#             os.makedirs(checkpoint_path)
#         return checkpoint_path
#
#     def load_datasets(self, num_dim=4, target_feature=None, cv=False):
#         train_data, val_data, test_data = self.dataset.get_datasets()
#         self.seq_len = train_data.shape[1] - 1
#         self.logger.info('num samples in training dataset:{}'.format(train_data.shape[0]))
#         train_dataset, val_dataset, test_dataset = self.dataset.data_to_dataset(train_data=train_data,
#                                                                                 val_data=val_data,
#                                                                                 test_data=test_data,
#                                                                                 target_feature=target_feature, cv=cv,
#                                                                                 num_dim=num_dim)
#         for (inp, tar) in train_dataset.take(1):
#             self.output_size = tf.shape(tar)[-1].numpy()
#             self.num_features = tf.shape(inp)[-1].numpy()
#
#         return train_dataset, val_dataset, test_dataset


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
                       checkpoint_path=self.ckpt_path,
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


#algos = {"smc_t": SMCTAlgo, "lstm": RNNAlgo, "baseline_t": BaselineTAlgo}
