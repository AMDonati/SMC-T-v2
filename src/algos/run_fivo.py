import tensorflow as tf
import os
import numpy as np
from src.algos.run_rnn import Algo
import numpy as np
import matplotlib.pyplot as plt

class FIVOAlgo(Algo):
    def __init__(self, dataset, args):
        super(FIVOAlgo, self).__init__(dataset=dataset, args=args)
        self.out_folder = args.save_path
        self.logger = self.create_logger()
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=3)

    def _load_samples(self):
        samples = np.load(os.path.join(self.save_path, "samples_unistep.npy"))
        samples = tf.constant(samples, dtype=tf.float32)
        samples = tf.squeeze(samples, axis=1) # (S,B,N,F)
        samples = tf.transpose(samples, perm=[1,2,0,3]) # (B,N,S,F)
        mean_samples = tf.reduce_mean(samples, axis=1)  # (B,S,F)
        prefix = np.load(os.path.join(self.save_path, "prefix.npy")) # (S,B,F)
        prefix = tf.constant(prefix, dtype=tf.float32)
        prefix = tf.transpose(prefix, perm=[1,0,2]) # (B,S,F)
        return samples, mean_samples, prefix

    def _compute_mse_from_samples_mean(self):
        samples, mean_samples, prefix = self._load_samples()
        mean_samples = tf.reduce_mean(samples, axis=1) # (B,S,F)
        for (_, tar) in self.test_dataset:
            loss_test = tf.keras.losses.MSE(tar, mean_samples)  # (B,S)
            loss_test = tf.reduce_mean(loss_test)
        self.logger.info("Test loss for FIVO: {}".format(loss_test))
        self.logger.info("mean preds:{}".format(tf.squeeze(mean_samples[0])))
        self.logger.info("true targets:{}".format(tf.squeeze(tar[0])))
        self.logger.info("prefix: {}".format(prefix[0]))
        return loss_test

    def plot_preds_targets(self):
        samples, mean_samples, _ = self._load_samples()
        index = np.random.randint(mean_samples.shape[0])
        mean_sample = tf.squeeze(mean_samples[index]).numpy()
        sample = tf.squeeze(samples[index]).numpy()
        for (_, tar) in self.test_dataset:
            target = tf.squeeze(tar[index]).numpy()
        x = np.linspace(1,24,24)
        plt.plot(x, mean_sample, 'red', lw=2, label='predictions for sample: {}'.format(index))
        plt.plot(x, target, 'blue', lw=2, label='predictions for sample: {}'.format(index))
        for i in range(sample.shape[0]):
            plt.scatter(x, sample[i], c='orange')
        plt.show()


    def test(self):
        loss_test = self._compute_mse_from_samples_mean
        for _ in range(10):
            self.plot_preds_targets()