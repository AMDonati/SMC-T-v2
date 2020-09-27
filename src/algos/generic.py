from src.utils.utils_train import create_logger
import os
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np


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

    def train(self):
        pass

    def test(self):
        pass

    def launch_inference(self, **kwargs):
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

    def load_datasets(self, num_dim=4, target_feature=None):
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
            train_dataset, val_dataset, test_dataset = self.dataset.get_datasets_for_crossvalidation(num_dim=num_dim,
                                                                                                     target_feature=target_feature)
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
        mc_dropout_unistep_path = os.path.join(self.inference_path, 'mc_dropout_samples_test_data_unistep.npy')
        mc_dropout_multistep_path = os.path.join(self.inference_path, 'mc_dropout_samples_test_data_multistep.npy')
        return mc_dropout_unistep_path, mc_dropout_multistep_path

    def compute_mse_predictive_distribution(self, alpha):
        for (inp, _) in self.test_dataset:
            if len(tf.shape(inp)) == 3:
                inp = tf.expand_dims(inp, axis=1)
            test_data_tiled = tf.tile(inp, multiples=[1, self.mc_samples, 1, 1])
            mse = tf.keras.losses.MSE(self.test_predictive_distribution, alpha * test_data_tiled)
            mse = tf.reduce_mean(mse)
        return mse

    def compute_predictive_interval(self, factor=1.96):
        assert self.test_predictive_distribution is not None, "error in predictive intervals computation"
        mean_distrib = tf.reduce_mean(self.test_predictive_distribution, axis=1)
        std_distrib = tf.math.reduce_std(self.test_predictive_distribution, axis=1)
        lower_bounds = mean_distrib - factor * std_distrib  # shape(B,S,F)
        upper_bounds = mean_distrib + factor * std_distrib
        #return tf.cast(lower_bounds, dtype=tf.float64), tf.cast(upper_bounds, dtype=tf.float64)
        return lower_bounds, upper_bounds

    def compute_MPIW(self):
        inside_pi = 0
        lower_bounds, upper_bounds = self.compute_predictive_interval()
        seq_len = lower_bounds.shape[1]
        num_samples = lower_bounds.shape[0]
        for (inp, _) in self.test_dataset:
            if len(tf.shape(inp)) == 4:
                inp = tf.squeeze(inp, axis=1)
            for index in range(num_samples):
                for t in range(seq_len):
                    item = inp[index, t, :]  # shape (F)
                    low_b = lower_bounds[index, t]
                    upper_b = upper_bounds[index, t]
                    bool_low = tf.math.reduce_all(tf.math.greater_equal(item, low_b))
                    bool_up = tf.math.reduce_all(tf.math.greater_equal(upper_b, item))
                    if bool_low and bool_up:
                        inside_pi += 1
        mpiw = inside_pi / (seq_len * num_samples)
        return mpiw

    def plot_preds_targets(self, predictions_test):
        for (inputs, targets) in self.test_dataset:
            if len(tf.shape(inputs)) == 4:
                inputs = tf.squeeze(inputs, axis=1)
                targets = tf.squeeze(targets, axis=1)
            index = np.random.randint(inputs.shape[0])
            inp, tar = inputs[index], targets[index]
            mean_pred = predictions_test[index, :, 0].numpy()
            #tar = tar[:, 0].numpy()
            inp = inp[:, 0].numpy()
        if self.test_predictive_distribution is not None:
            sample = self.test_predictive_distribution[index, :, :, 0].numpy()  # (mc_samples, seq_len, F)
            mean_pred = np.mean(sample, axis=0)
        x = np.linspace(1, self.seq_len, self.seq_len)
        plt.plot(x, mean_pred, 'red', lw=2, label='predictions for sample: {}'.format(index))
        #plt.plot(x, tar, 'blue', lw=2, label='targets for sample: {}'.format(index))
        plt.plot(x, inp, 'cyan', lw=2, label='ground-truth for sample: {}'.format(index))
        if self.test_predictive_distribution is not None:
            for i in range(sample.shape[0]):
                plt.scatter(x, sample[i], c='orange')
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(self.out_folder, "plot_test_preds_targets_sample{}".format(index)))
        plt.close()
