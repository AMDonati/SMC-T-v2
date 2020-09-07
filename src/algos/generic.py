from utils.utils_train import create_logger
import os
import tensorflow as tf

class Algo:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.bs = args.bs
        self.EPOCHS = args.ep
        self.output_path = args.output_path
        self.save_path = args.save_path
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        self.out_folder = args.output_path

    def train(self):
        pass

    def test(self):
        pass

    def launch_inference(self): #TODO: add a kwargs here.
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

    def load_datasets(self, num_dim=4, target_feature=None, cv=False):
        train_data, val_data, test_data = self.dataset.get_datasets()
        self.seq_len = train_data.shape[1] - 1
        self.logger.info('num samples in training dataset:{}'.format(train_data.shape[0]))
        train_dataset, val_dataset, test_dataset = self.dataset.data_to_dataset(train_data=train_data,
                                                                                val_data=val_data,
                                                                                test_data=test_data,
                                                                                target_feature=target_feature, cv=cv,
                                                                                num_dim=num_dim)
        for (inp, tar) in train_dataset.take(1):
            self.output_size = tf.shape(tar)[-1].numpy()
            self.num_features = tf.shape(inp)[-1].numpy()

        return train_dataset, val_dataset, test_dataset