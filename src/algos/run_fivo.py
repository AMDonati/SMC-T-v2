import tensorflow as tf
import os
import numpy as np
from src.algos.run_rnn import Algo

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
        return samples

    def _compute_mse_from_samples_mean(self):
        samples = self._load_samples()
        mean_samples = tf.reduce_mean(samples, axis=1) # (B,S,F)
        for (_, tar) in self.test_dataset:
            loss_test = tf.keras.losses.MSE(tar, mean_samples)  # (B,S)
            loss_test = tf.reduce_mean(loss_test)
        self.logger.info("Test loss for FIVO: {}".format(loss_test))
        self.logger.info("mean preds:{}".format(tf.squeeze(mean_samples[0])))
        self.logger.info("true targets:{}".format(tf.squeeze(tar[0])))
        return loss_test

    def test(self):
        loss_test = self._compute_mse_from_samples_mean()