import os
from src.algos.generic import Algo
import numpy as np
import datetime
from statsmodels.tsa.arima.model import ARIMA
from src.utils.utils_train import write_to_csv
import tensorflow as tf

class ARIMAAlgo(Algo):
    def __init__(self, dataset, args):
        super(ARIMAAlgo, self).__init__(dataset=dataset, args=args)
        self.p = args.p_model
        self.d = args.d
        self.q = args.q
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self.create_logger()
        self.train_data, self.val_data, self.test_data = self.load_datasets()
        self.distribution = True
        assert self.past_len < self.seq_len, "past_len should be inferior to the sequence length of the dataset"
        self.future_len = args.future_len if args.future_len is not None else (self.seq_len - self.past_len)

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            output_path = args.output_path
            out_file = '{}_p{}d{}q{}'.format(args.algo, args.p_model, args.d, args.q)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            output_folder = os.path.join(output_path, out_file, datetime_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            return output_folder

    def gather_alltimesteps(self, data):
        data = np.reshape(data, newshape=(data.shape[0]*data.shape[1], data.shape[-1]))
        return data

    def load_datasets(self, num_dim=3):
        if not self.cv:
            train_data, val_data, test_data = self.dataset.get_datasets()
            self.seq_len = train_data.shape[1] - 1
            self.num_features = train_data.shape[-1]
            self.logger.info('num samples in training dataset: {}'.format(train_data.shape[0]))
            self.logger.info('number of timeteps: {}'.format(self.seq_len))
            self.logger.info('number of features: {}'.format(self.num_features))
            train_data = self.gather_alltimesteps(train_data)
            val_data = self.gather_alltimesteps(val_data)
            test_data = self.gather_alltimesteps(test_data)
            return train_data, val_data, test_data
        else:
            list_train_data_seq, list_val_data_seq, list_test_data_seq = self.dataset.get_data_splits_for_crossvalidation()
            list_train_data, list_val_data, list_test_data = [], [], []
            for (train_data, val_data, test_data) in zip(list_train_data_seq, list_val_data_seq, list_test_data_seq):
                list_train_data.append(self.gather_alltimesteps(train_data))
                list_val_data.append(self.gather_alltimesteps(val_data))
                list_test_data.append(self.gather_alltimesteps(test_data))
            return list_train_data, list_val_data, list_test_data

    def get_predictive_distribution(self, save_path=None):  # TODO: add option to save_distrib.
        unistep_samples = []
        predictions = self.model_test.get_prediction()
        if self.distribution:
            mean_preds = predictions.predicted_mean
            std_preds = predictions.se_mean
            for mean, std in zip(mean_preds, std_preds):
                samples = predictions.dist(loc=mean, scale=std).rvs(self.mc_samples)
                unistep_samples.append(samples)
        unistep_samples = np.stack(unistep_samples, axis=0)
        self.test_predictive_distribution = tf.expand_dims(tf.constant(unistep_samples, dtype=tf.float32), axis=-1) # (num_samples, mc_samples, 1)

    def compute_mse_predictive_distribution(self, alpha):
        test_data = tf.constant(self.test_data, dtype=tf.float32)
        test_data = tf.expand_dims(test_data, axis=1)
        test_data_tiled = tf.tile(test_data, multiples=[1, self.mc_samples, 1])
        mse = tf.keras.losses.MSE(self.test_predictive_distribution, alpha * test_data_tiled)
        mse = tf.reduce_mean(mse)
        return mse

    def train(self, **kwargs):
        # fit model
        model = ARIMA(self.train_data, order=(self.p, self.d, self.q))
        model_fit = model.fit()
        # summary of fit model
        print(model_fit.summary())
        self.logger.info("train loss:{}".format(model_fit.mse))
        # forecast on validation set:
        model_val = model_fit.apply(self.val_data)
        self.logger.info("val loss:{}".format(model_val.mse))
        model_test = model_fit.apply(self.test_data)
        self.model_fit = model_fit
        self.model_val = model_val
        self.model_test = model_test

    def test(self, **kwargs):
        test_metrics = {}
        self.logger.info("test loss:{}".format(self.model_test.mse))
        test_metrics["test_loss"] = self.model_test.mse
        if self.dataset.name == "synthetic":
            self.logger.info("computing mean square error of predictive distribution...")
            self.get_predictive_distribution()
            mse_predictive_distrib = self.compute_mse_predictive_distribution(alpha=kwargs["alpha"])
            test_metrics["mse_distrib"] = mse_predictive_distrib
            self.logger.info("mse_distrib: {}".format(mse_predictive_distrib))
        write_to_csv(dic=test_metrics, output_dir=os.path.join(self.out_folder, "test_metrics_unistep.csv"))

    def test_cv(self, **kwargs):
        pass