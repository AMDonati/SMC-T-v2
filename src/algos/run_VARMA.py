import os
from src.algos.generic import Algo
import numpy as np
import datetime
import statsmodels.api as sm
from src.utils.utils_train import write_to_csv

class VARMAAlgo(Algo):
    def __init__(self, dataset, args):
        super(VARMAAlgo, self).__init__(dataset=dataset, args=args)
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
            out_file = '{}_d{}q{}'.format(args.algo, args.d, args.q)
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

    def train(self, **kwargs):
        # fit model
        model = ARIMA(self.train_data, order=(self.seq_len, self.d, self.q))
        model_fit = model.fit()
        # summary of fit model
        print(model_fit.summary())
        self.logger.info("train loss:{}".format(model_fit.mse))
        # forecast on validation set:
        val_forecast = model_fit.apply(self.val_data)
        self.logger.info("val loss:{}".format(val_forecast.mse))
        test_metrics = {}
        test_model = model_fit.apply(self.test_data)
        self.logger.info("test loss:{}".format(test_model.mse))
        test_metrics["test_loss"] = test_model.mse
        write_to_csv(dic=test_metrics, output_dir=os.path.join(self.out_folder, "test_metrics_unistep.csv"))

    def test(self, **kwargs):
        pass

    def test_cv(self, **kwargs):
        pass