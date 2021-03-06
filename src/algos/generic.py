from src.utils.utils_train import create_logger
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from src.utils.utils_train import write_to_csv, create_config_file
import json
import math


class Algo:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.bs = args.bs
        self.EPOCHS = args.ep
        self.cv = args.cv
        self.output_path = args.output_path
        self.save_path = args.save_path
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        self.out_folder = args.output_path
        self.start_epoch = 0
        self.mc_samples = args.mc_samples
        self.past_len = args.past_len
        self.test_predictive_distribution = None
        self.test_predictive_distribution_multistep = None
        self.distribution = False
        self.lambda_QD = args.lambda_QD

    def stochastic_forward_pass_multistep(self, inp_model, future_inp_features, save_path):
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

    def save_hparams(self, args):
        dict_hparams = vars(args)
        dict_hparams = {key: str(value) for key, value in dict_hparams.items()}
        config_path = os.path.join(self.out_folder, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(dict_hparams, fp, sort_keys=True, indent=4)
        # create_config_file(os.path.join(self.out_folder, "config.ini"), args)

    def load_datasets(self, num_dim=4):
        if not self.cv:
            train_data, val_data, test_data = self.dataset.get_datasets()
            self.logger.info('num samples in training dataset: {}'.format(train_data.shape[0]))
            self.logger.info('number of timeteps: {}'.format(train_data.shape[1] - 1))
            self.logger.info('number of features: {}'.format(train_data.shape[-1]))
            train_dataset, val_dataset, test_dataset = self.dataset.data_to_dataset(train_data=train_data,
                                                                                    val_data=val_data,
                                                                                    test_data=test_data,
                                                                                    num_dim=num_dim)
            self.dataset.check_dataset(train_dataset)
            self.dataset.check_dataset(val_dataset)
            self.dataset.check_dataset(test_dataset)
            for (inp, tar) in train_dataset.take(1):
                self.output_size = tf.shape(tar)[-1].numpy()
                self.logger.info("number of target features: {}".format(self.output_size))
                self.num_features = tf.shape(inp)[-1].numpy()
                self.seq_len = tf.shape(inp)[-2].numpy()
        else:
            self.logger.info("loading datasets for performing cross-validation...")
            train_dataset, val_dataset, test_dataset = self.dataset.get_datasets_for_crossvalidation(num_dim=num_dim)
            for (inp, tar) in train_dataset[0].take(1):
                self.output_size = tf.shape(tar)[-1].numpy()
                self.logger.info("number of target features: {}".format(self.output_size))
                self.num_features = tf.shape(inp)[-1].numpy()
                self.seq_len = tf.shape(inp)[-2].numpy()
        return train_dataset, val_dataset, test_dataset

    def _get_inference_paths(self):
        # create inference folder
        self.inference_path = os.path.join(self.out_folder, "inference_results")
        if not os.path.isdir(self.inference_path):
            os.makedirs(self.inference_path)
        distrib_unistep_path = os.path.join(self.inference_path, 'distrib_unistep.npy')
        distrib_multistep_path = os.path.join(self.inference_path, 'distrib_multistep.npy')
        return distrib_unistep_path, distrib_multistep_path

    def get_predictive_distribution(self, save_path):
        if self.distribution:
            for (inp, _) in self.test_dataset:
                mc_samples_uni = self._MC_Dropout(inp_model=inp, save_path=save_path)
                print("shape of predictive distribution", mc_samples_uni.shape)
            self.test_predictive_distribution = mc_samples_uni

    def get_predictive_distribution_multistep(self, save_path):
        if self.distribution:
            for (inp, _) in self.test_dataset:
                if self.output_size < self.num_features:
                    future_input_features = inp[:, self.past_len:, len(self.dataset.target_features):]
                else:
                    future_input_features = None
                mc_samples_multi = self.stochastic_forward_pass_multistep(inp_model=inp[:, :self.past_len, :],
                                                                          save_path=save_path,
                                                                          future_inp_features=future_input_features)
                print("shape of predictive distribution multistep", mc_samples_multi.shape)
            self.test_predictive_distribution_multistep = mc_samples_multi

    def compute_mse_predictive_distribution(self, alpha):
        for (inp, _) in self.test_dataset:
            if len(tf.shape(inp)) == 3:
                inp = tf.expand_dims(inp, axis=1)
            test_data_tiled = tf.tile(inp, multiples=[1, self.mc_samples, 1, 1])
            mse = tf.keras.losses.MSE(self.test_predictive_distribution, alpha * test_data_tiled)
            mse = tf.reduce_mean(mse)
        return mse

    def compute_predictive_interval(self, predictive_distribution, std_multiplier=1.96, save_path=None):
        assert predictive_distribution is not None, "error in predictive intervals computation"
        mean_distrib = tf.reduce_mean(predictive_distribution, axis=1)
        std_distrib = tf.math.reduce_std(predictive_distribution, axis=1)
        lower_bounds = mean_distrib - std_multiplier * std_distrib  # shape(B,S,F)
        upper_bounds = mean_distrib + std_multiplier * std_distrib
        piw = upper_bounds - lower_bounds  # (B,S,F)
        MPIW = tf.reduce_mean(piw)
        MPIW_per_timestep = tf.reduce_mean(piw, axis=[0, 2])
        if save_path is not None:
            np.save(os.path.join(save_path, "MPIW_per_timestep.npy"), MPIW_per_timestep)
        return lower_bounds, upper_bounds, MPIW, MPIW_per_timestep.numpy()

    def compute_PICP_MPIW(self, predictive_distribution, past_len=0, save_path=None):
        inside_pi, MPIW_captured = 0, 0
        PICP_per_timestep, MPIW_captured_per_timestep = [], []
        lower_bounds, upper_bounds, MPIW, MPIW_per_timestep = self.compute_predictive_interval(predictive_distribution,
                                                                                               save_path=save_path)
        seq_len = lower_bounds.shape[1]
        num_samples = lower_bounds.shape[0]
        num_features = lower_bounds.shape[-1]
        assert num_features == self.output_size
        for (inp, _) in self.test_dataset:
            if len(tf.shape(inp)) == 4:
                inp = tf.squeeze(inp, axis=1)
            inp = inp[:, past_len:, :len(self.dataset.target_features)]  # taking only the target features.
            for t in range(seq_len):
                item = inp[:, t, :]  # shape (B,F)
                low_b = lower_bounds[:, t, :]  # (B,F)
                upper_b = upper_bounds[:, t, :]
                bool_low = tf.math.greater_equal(item, low_b)  # (B,F)
                bool_up = tf.math.greater_equal(upper_b, item)  # (B,F)
                prod_bool = tf.math.multiply(tf.cast(bool_up, dtype=tf.float32),
                                             tf.cast(bool_low, dtype=tf.float32))  # (B,F)
                # compute of PICP and MPIW captured.
                num_inside_pi = tf.reduce_sum(prod_bool)
                mpiw_capt = tf.reduce_sum(prod_bool * (upper_b - low_b))
                PICP_per_timestep.append(num_inside_pi.numpy() / (num_samples * num_features))
                MPIW_captured_per_timestep.append(mpiw_capt.numpy() / num_inside_pi)
                inside_pi += num_inside_pi
                MPIW_captured += mpiw_capt

        PICP = inside_pi.numpy() / (seq_len * num_samples * num_features)
        MPIW_captured = MPIW_captured.numpy() / inside_pi.numpy()
        PICP_per_timestep = np.stack(PICP_per_timestep)  # shape S.
        MPIW_captured_per_timestep = np.stack(MPIW_captured_per_timestep)  # shape S.
        N = (num_features * num_samples * seq_len)
        loss_QD = self.compute_loss_QD(mpiw_captured=MPIW_captured, PICP=PICP, N=N, lambd=self.lambda_QD)
        if save_path is not None:
            np.save(os.path.join(save_path, "PICP_per_timestep_mean.npy"), PICP_per_timestep)
            np.save(os.path.join(save_path, "MPIW_capt_per_timestep.npy"), MPIW_captured_per_timestep)
        return (PICP, PICP_per_timestep), (MPIW, MPIW_per_timestep), (
            MPIW_captured, MPIW_captured_per_timestep), loss_QD

    def compute_loss_QD(self, mpiw_captured, PICP, N, lambd=15., alpha=0.05):
        picp_term = max(0., (1 - alpha) - PICP)
        ratio = N / (alpha * (1 - alpha))
        loss_QD = mpiw_captured + lambd * ratio * math.pow(picp_term, 2)
        return loss_QD

    def plot_preds_targets(self, predictions_test):
        for (inputs, targets) in self.test_dataset:
            if len(tf.shape(inputs)) == 4:
                inputs = tf.squeeze(inputs, axis=1)
                targets = tf.squeeze(targets, axis=1)
            index = np.random.randint(inputs.shape[0])
            inp, tar = inputs[index], targets[index]
            mean_pred = predictions_test[index, :, 0].numpy()
            # tar = tar[:, 0].numpy()
            inp = inp[:, 0].numpy()
        if self.test_predictive_distribution is not None:
            sample = self.test_predictive_distribution[index, :, :, 0].numpy()  # (mc_samples, seq_len, F)
            mean_pred = np.mean(sample, axis=0)
        x = np.linspace(1, self.seq_len, self.seq_len)
        plt.plot(x, mean_pred, 'red', lw=2, label='predictions for sample: {}'.format(index))
        # plt.plot(x, tar, 'blue', lw=2, label='targets for sample: {}'.format(index))
        plt.plot(x, inp, 'cyan', lw=2, label='ground-truth for sample: {}'.format(index))
        if self.test_predictive_distribution is not None:
            for i in range(sample.shape[0]):
                plt.scatter(x, sample[i], c='orange')
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(self.out_folder, "plot_test_preds_targets_sample{}".format(index)))
        plt.close()

    def test_cv(self, **kwargs):
        test_metrics_cv = {}
        for num_train in range(5):
            self.logger.info("-" * 20 + "Testing for test split number {}".format(num_train + 1) + "-" * 20)
            self._load_ckpt(num_train=num_train + 1)
            if num_train == 0:
                test_metrics = self.test(**kwargs, save_particles=True, plot=True, save_metrics=False,
                                         save_distrib=True)
            else:
                test_metrics = self.test(**kwargs, save_particles=False, plot=False, save_metrics=False,
                                         save_distrib=False)
            self.logger.info("-" * 60)
            for key, metric in test_metrics.items():
                test_metrics_cv[key + '{}'.format(num_train + 1)] = metric
        test_losses, test_mse = [], []
        for key, val in test_metrics_cv.items():
            if "test_loss" in key:
                test_losses.append(val)
            else:
                test_mse.append(val)
        test_metrics_cv["test_loss_mean"] = np.mean(test_losses)
        test_metrics_cv["test_loss_std"] = np.std(test_losses)
        if len(test_mse) > 0:
            test_metrics_cv["mse_mean"] = np.mean(test_mse)
            test_metrics_cv["mse_std"] = np.std(test_mse)
        write_to_csv(output_dir=os.path.join(self.out_folder, "test_metrics.csv"), dic=test_metrics_cv)
        return test_metrics_cv

    def test(self, **kwargs):
        test_metrics_unistep = {}
        test_metrics_multistep = {}
        test_loss, predictions_test = self.compute_test_loss(kwargs["save_particles"])
        test_metrics_unistep["test_loss"] = test_loss.numpy()
        self.logger.info("test loss at the end of training: {}".format(test_loss.numpy()))
        if self.distribution:
            if kwargs["save_distrib"]:
                distrib_unistep_path, distrib_multistep_path = self._get_inference_paths()
            else:
                distrib_unistep_path, distrib_multistep_path = None, None
            self.get_predictive_distribution(save_path=distrib_unistep_path)
            if self.dataset.name == "synthetic":
                self.logger.info("computing mean square error of predictive distribution...")
                mse = self.compute_mse_predictive_distribution(alpha=kwargs["alpha"])
                if self.dataset.model == 2:
                    mse_2 = self.compute_mse_predictive_distribution(alpha=kwargs["beta"])
                    mse = kwargs["p"] * mse + (1 - kwargs["p"]) * mse_2
                self.logger.info("mse predictive distribution: {}".format(mse.numpy()))
                test_metrics_unistep["mse"] = mse.numpy()
            else:
                self.logger.info("computing MPIW on test set...")
                (PICP, PICP_per_timestep), (MPIW, MPIW_per_timestep), (
                    MPIW_capt, MPIW_capt_per_timestep), loss_QD = self.compute_PICP_MPIW(
                    predictive_distribution=self.test_predictive_distribution)
                test_metrics_unistep["PICP"] = round(PICP, 4)
                test_metrics_unistep["MPIW"] = np.round(MPIW.numpy(), 4)
                test_metrics_unistep["MPIW_capt"] = round(MPIW_capt, 4)
                test_metrics_unistep["loss_QD"] = round(loss_QD, 4)
                self.logger.info("---------------------TEST METRICS UNISTEP -----------------------------------")
                self.logger.info(test_metrics_unistep)
                self.logger.info('-' * 60)
                if kwargs["multistep"]:
                    self.logger.info("computing multistep uncertainty metrics on test set...")
                    self.get_predictive_distribution_multistep(save_path=distrib_multistep_path)
                    (PICP_m, PICP_m_per_timestep), (MPIW_m, MPIW_m_per_timestep), (
                        MPIW_m_capt, MPIW_m_capt_per_timestep), loss_QD_m = self.compute_PICP_MPIW(
                        predictive_distribution=self.test_predictive_distribution_multistep, past_len=self.past_len,
                        save_path=self.inference_path)
                    test_metrics_multistep["PICP_multistep"] = round(PICP_m,4)
                    test_metrics_multistep["MPIW_multistep"] = np.round(MPIW_m.numpy(), 4)
                    test_metrics_multistep["loss_QD_multistep"] = round(loss_QD_m, 4)
                    test_metrics_multistep["MPIW_capt_multistep"] = round(MPIW_m_capt, 4)
                    test_metrics_multistep["PICP_per_timestep_multistep"] = np.round(PICP_m_per_timestep, 4)
                    test_metrics_multistep["MPIW_per_timestep_multistep"] = np.round(MPIW_m_per_timestep, 4)
                    test_metrics_multistep["MPIW_capt_per_timestep_multistep"] = np.round(MPIW_m_capt_per_timestep, 4)
                    self.logger.info("----------------------TEST METRICS MULTISTEP --------------------------")
                    for (k, v) in test_metrics_multistep.items():
                        self.logger.info(k + ": {}".format(v))
                    self.logger.info('-' * 60)
        # plot targets versus preds for test samples:
        if kwargs["plot"]:
            for _ in range(4):
                self.plot_preds_targets(predictions_test=predictions_test)
        if kwargs["save_metrics"]:
            write_to_csv(dic=test_metrics_unistep, output_dir=os.path.join(self.out_folder, "test_metrics_unistep.csv"))
            if test_metrics_multistep:
                write_to_csv(dic=test_metrics_multistep,
                             output_dir=os.path.join(self.out_folder, "test_metrics_multistep.csv"))
