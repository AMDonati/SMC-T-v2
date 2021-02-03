import os
from src.algos.generic import Algo
import numpy as np
import datetime
import statsmodels.api as sm
from src.utils.utils_train import write_to_csv
from src.preprocessing.utils import split_array_per_sequences, split_dataset_into_seq, get_array_per_step


class VARMAAlgo(Algo):
    def __init__(self, dataset, args, max_size_test=3000):
        super(VARMAAlgo, self).__init__(dataset=dataset, args=args)
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self.create_logger()
        self.max_samples = args.max_samples
        self.max_size_test = max_size_test
        self.train_data, self.val_data, self.test_data = self.load_datasets()
        self.distribution = True
        assert self.past_len < self.seq_len, "past_len should be inferior to the sequence length of the dataset"
        self.future_len = args.future_len if args.future_len is not None else (self.seq_len - self.past_len)
        self.p = args.p_model if args.p_model is not None else self.seq_len
        self.q = args.q
        self.exog = args.exog

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            output_path = args.output_path
            out_file = '{}_p{}q{}_ep{}'.format(args.algo, args.p_model, args.q, args.ep)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            output_folder = os.path.join(output_path, out_file, datetime_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            return output_folder

    def split_dataset(self, data, TRAIN_SPLIT=0.7, VAL_SPLIT=0.5, normalize=True):
        if self.dataset.name == "weather":
            data = get_array_per_step(data, step=24)
        if normalize:
            data_mean = data.mean(axis=0)
            data_std = data.std(axis=0)
            data = (data - data_mean) / data_std
        num_samples = len(data)
        train_split = int(num_samples * TRAIN_SPLIT)
        val_split = train_split + int((num_samples - train_split) * VAL_SPLIT)
        train_data = data[:train_split]
        val_data = data[train_split:val_split]
        test_data = data[val_split:]
        return train_data, val_data, test_data

    def load_datasets(self, num_dim=3):
        self.seq_len = self.dataset.get_datasets()[0].shape[1] - 1
        self.num_features = self.dataset.get_datasets()[0].shape[-1]
        data = self.dataset.data_arr
        train_data, val_data, test_data = self.split_dataset(data)
        if self.max_samples is not None:
            train_data = train_data[:self.max_samples]
        if test_data.shape[0] > self.max_size_test:
            test_data = test_data[:self.max_size_test]
            print("reducing test dataset to {} samples...", self.max_size_test)
        self.logger.info('num samples in training dataset: {}'.format(train_data.shape[0]))
        self.logger.info('number of timeteps: {}'.format(self.seq_len))
        self.logger.info('number of features: {}'.format(len(self.dataset.target_features)))
        self.logger.info('seq len (past dependency): {}'.format(self.seq_len))
        return train_data, val_data, test_data

    def get_endog_exog_data(self, data):
        if len(self.dataset.target_features) < data.shape[-1]:
            endog = data[:, self.dataset.target_features]
            exog = data[:, self.dataset.target_features:] if self.exog else None
        else:
            endog = data
            exog = None
        assert endog.shape[-1] == len(self.dataset.target_features), "error in endogeneous data"
        return endog, exog

    def train(self, **kwargs):
        # fit model
        endog, exog = self.get_endog_exog_data(self.train_data)
        model = sm.tsa.VARMAX(endog, order=(self.p, self.q), trend='n',
                              exog=exog)
        model_fit = model.fit(maxiter=self.EPOCHS, disp=False)
        print(model_fit.summary())
        self.logger.info("train loss:{}".format(model_fit.mse))
        # forecast on validation set:
        val_endog, val_exog = self.get_endog_exog_data(self.val_data)
        model_val = model_fit.apply(endog=val_endog, exog=val_exog)
        self.logger.info("val loss:{}".format(model_val.mse))
        test_endog, test_exog = self.get_endog_exog_data(self.test_data)
        model_test = model_fit.apply(endog=test_endog, exog=test_exog)
        self.model_fit = model_fit
        self.model_val = model_val
        self.model_test = model_test

    def compute_predictive_interval(self, test_preds, past_len=0, save_path=None):
        CI = test_preds.conf_int()
        CI = CI[past_len:]
        lower_bounds = CI[:, :len(self.dataset.target_features)]  # (B*S,F) #TODO: check conf_int function.
        upper_bounds = CI[:, len(self.dataset.target_features):]  # (B*S, F)
        #print("upper bounds", upper_bounds)
        #print("lower bounds", lower_bounds)
        piw = upper_bounds - lower_bounds
        MPIW = np.mean(piw)
        MPIW_per_timestep = np.mean(piw, axis=-1)
        return lower_bounds, upper_bounds, MPIW, MPIW_per_timestep

    def compute_PICP_MPIW(self, test_preds, past_len=0, save_path=None):
        lower_bounds, upper_bounds, MPIW, MPIW_per_timestep = self.compute_predictive_interval(test_preds)
        #print("MPIW per timestep", MPIW_per_timestep)
        test_endog, _ = self.get_endog_exog_data(self.test_data)
        bool_low = np.greater_equal(test_endog, lower_bounds)  # (B*S,F)
        bool_up = np.greater_equal(upper_bounds, test_endog)  # (B*S,F)
        prod_bool = np.multiply(bool_low, bool_up)
        PICP = np.mean(prod_bool)
        PICP_per_timestep = np.mean(prod_bool, axis=-1)

        return (PICP, PICP_per_timestep), (MPIW, MPIW_per_timestep), (None, None), None

    def compute_PICP_MPIW_multistep(self, test_data_in_seq, save_path=None):
        inside_pi, inside_pi_2 = 0, 0
        MPIW, MPIW_per_timestep, PICP_per_timestep = [], [], []
        for i in range(test_data_in_seq.shape[0]): # loop over samples.
            sample = test_data_in_seq[i]  # (S,F)
            model = self.model_fit.apply(sample)
            preds = model.get_prediction(dynamic=self.past_len)
            #print("preds mean", preds.predicted_mean)
            lower_bounds, upper_bounds, mpiw, mpiw_per_timestep = self.compute_predictive_interval(preds, past_len=self.past_len,
                                                                                                   save_path=save_path)
            bool_low = np.greater_equal(test_data_in_seq[i, self.past_len:], lower_bounds)  # (S,F)
            bool_up = np.greater_equal(upper_bounds, test_data_in_seq[i, self.past_len:])  # (S,F)
            prod_bool = np.multiply(bool_low, bool_up)
            num_inside_pi = np.sum(prod_bool)
            #print("INSIDE PI", inside_pi_2)
            #print("mpiw", mpiw)
            #print("mpiw per timestep", mpiw_per_timestep)
            inside_pi += num_inside_pi
            PICP_per_timestep.append(np.mean(prod_bool, axis=-1))
            MPIW.append(mpiw)
            MPIW_per_timestep.append(mpiw_per_timestep)

        MPIW = np.mean(MPIW)
        MPIW_per_timestep = np.mean(np.stack(MPIW_per_timestep, axis=0), axis=0)
        PICP = inside_pi / (test_data_in_seq.shape[0] * upper_bounds.shape[0] * test_data_in_seq.shape[-1])
        PICP_per_timestep = np.mean(np.stack(PICP_per_timestep, axis=0), axis=0)

        return (PICP, PICP_per_timestep), (MPIW, MPIW_per_timestep), (None, None), None

    def test_unistep(self, **kwargs):
        test_metrics_unistep = {}
        self.logger.info("test loss:{}".format(self.model_test.mse))
        test_metrics_unistep["test_loss"] = self.model_test.mse
        test_preds = self.model_test.get_prediction()
        test_endog, _ = self.get_endog_exog_data(self.test_data)
        mse_2 = np.mean(np.square(test_endog - test_preds.predicted_mean))
        test_metrics_unistep["test_loss_2"] = mse_2
        if self.dataset.name != "synthetic":
            (PICP, PICP_per_timestep), (MPIW, MPIW_per_timestep), _, _ = self.compute_PICP_MPIW(test_preds)
            test_metrics_unistep["PICP"] = np.round(PICP, 4)
            test_metrics_unistep["MPIW"] = np.round(MPIW, 4)
            np.save(os.path.join(self.inference_path, "PICP_per_timestep.npy"), PICP_per_timestep)
            np.save(os.path.join(self.inference_path, "MPIW_per_timestep.npy"), MPIW_per_timestep)
        self.logger.info("---------------------TEST METRICS UNISTEP -----------------------------------")
        self.logger.info(test_metrics_unistep)
        self.logger.info('-' * 60)
        write_to_csv(dic=test_metrics_unistep, output_dir=os.path.join(self.out_folder, "test_metrics_unistep.csv"))

    def test_multistep(self):
        test_metrics_multistep = {}
        test_data, _ = self.get_endog_exog_data(self.test_data)
        test_data_in_seq = split_dataset_into_seq(test_data, start_index=0, end_index=None, history_size=self.seq_len, step=1)
        (PICP, PICP_per_timestep), (MPIW, MPIW_per_timestep), _, _ = self.compute_PICP_MPIW_multistep(test_data_in_seq=test_data_in_seq)
        test_metrics_multistep["PICP_multistep"] = np.round(PICP, 4)
        test_metrics_multistep["MPIW_multistep"] = np.round(MPIW, 4)
        test_metrics_multistep["PICP_per_timestep_multistep"] = np.round(PICP_per_timestep, 4)
        np.save(os.path.join(self.inference_path, "PICP_per_timestep_multistep.npy"), PICP_per_timestep)
        np.save(os.path.join(self.inference_path, "MPIW_per_timestep_multistep.npy"), MPIW_per_timestep)
        test_metrics_multistep["MPIW_per_timestep_multistep"] = np.round(MPIW_per_timestep, 4)
        self.logger.info("----------------------TEST METRICS MULTISTEP --------------------------")
        for (k, v) in test_metrics_multistep.items():
            self.logger.info(k + ": {}".format(v))
        self.logger.info('-' * 60)
        write_to_csv(dic=test_metrics_multistep, output_dir=os.path.join(self.out_folder, "test_metrics_multistep.csv"))

    def test(self, **kwargs):
        _, _ = self._get_inference_paths()
        self.test_unistep()
        if kwargs["multistep"] and self.dataset.name != "synthetic":
            self.test_multistep()


    def test_cv(self, **kwargs):
        pass
