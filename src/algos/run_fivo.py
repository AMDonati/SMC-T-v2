import tensorflow as tf
import os
import numpy as np
from src.algos.generic import Algo
import numpy as np
import matplotlib.pyplot as plt

class FIVOAlgo(Algo):
    def __init__(self, dataset, args):
        super(FIVOAlgo, self).__init__(dataset=dataset, args=args)
        self.dataset.BATCH_SIZE = 800 if args.split_fivo == "train" else 100
        self.out_folder = args.save_path
        self.logger = self.create_logger()
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=3)
        self.standardize = args.standardize
        self.split = args.split_fivo

    def _load_samples(self, split="test"):
        samples = np.load(os.path.join(self.save_path, "samples_unistep_{}.npy".format(split)))
        samples = tf.constant(samples, dtype=tf.float32)
        samples = tf.squeeze(samples, axis=1) # (S,B,N,F)
        samples = tf.transpose(samples, perm=[1,2,0,3]) # (B,N,S,F)
        mean_samples = tf.reduce_mean(samples, axis=1)  # (B,S,F)
        prefix = np.load(os.path.join(self.save_path, "prefix_{}.npy".format(split))) # (S,B,F)
        prefix = tf.constant(prefix, dtype=tf.float32)
        prefix = tf.transpose(prefix, perm=[1,0,2]) # (B,S,F)
        return samples, mean_samples, prefix

    def get_FIVO_data(self, split="test"):
        train_data, val_data, test_data = self.dataset.get_datasets()
        inputs, targets, _, train_mean = self.dataset.prepare_dataset_for_FIVO(train_data, val_data, test_data,
                                                                               split=split, standardize=self.standardize)
        targets = tf.transpose(targets, perm=[1, 0, 2])
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        return inputs, targets, train_mean

    def _compute_mse_from_samples_mean(self, split="test"):
        samples, mean_samples, prefix = self._load_samples(split=split)
        mean_samples = tf.reduce_mean(samples, axis=1) # (B,S,F)
        inputs, targets, train_mean = self.get_FIVO_data(split=split)
        loss_test = tf.keras.losses.MSE(targets, mean_samples)  # (B,S)
        loss_test = tf.reduce_mean(loss_test)
        self.logger.info("{} loss for FIVO: {}".format(split, loss_test))
        self.logger.info("{} mean preds:{}".format(split, tf.squeeze(mean_samples[0])))
        self.logger.info("{} true targets:{}".format(split, tf.squeeze(targets[0])))
        self.logger.info("{} prefix: {}".format(split, prefix[0]))
        return loss_test

    def plot_preds_targets(self, split="test"):
        samples, mean_samples, _ = self._load_samples(split=split)
        _, targets, train_mean = self.get_FIVO_data(split=split)
        index = np.random.randint(mean_samples.shape[0])
        mean_sample = tf.squeeze(mean_samples[index]).numpy()
        sample = tf.squeeze(samples[index]).numpy()
        target = tf.squeeze(targets[index]).numpy()
        x = np.linspace(1,24,24)
        plt.plot(x, mean_sample, 'red', lw=2, label='predictions for sample: {}'.format(index))
        plt.plot(x, target, 'blue', lw=2, label='predictions for sample: {}'.format(index))
        if sample.shape[0] > 1:
            for i in range(sample.shape[0]):
                plt.scatter(x, sample[i], c='orange')
        plt.show()


    def test(self, split="test"):
        loss = self._compute_mse_from_samples_mean(split=self.split)
        for _ in range(5):
            self.plot_preds_targets(split=split)