from src.utils.utils_train import create_logger
import os
import tensorflow as tf
import json

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
            self.logger.info('number of timeteps: {}'.format(train_data.shape[1]))
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
            train_dataset, val_dataset, test_dataset = self.dataset.get_datasets_for_crossvalidation(num_dim=num_dim, target_feature=target_feature)
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