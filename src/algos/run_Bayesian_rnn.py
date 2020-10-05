from src.algos.generic import Algo
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.Baselines.BayesianLSTM import BayesianLSTMModel
import datetime
from src.utils.utils_train import saving_training_history


class BayesianRNNAlgo(Algo):
    def __init__(self, dataset, args):
        super(BayesianRNNAlgo, self).__init__(dataset=dataset, args=args)
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_torch_datasets(args=args)
        self.lr = args.lr
        self.EPOCHS = args.ep
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bayesian_lstm = BayesianLSTMModel(input_size=self.num_features, rnn_units=args.rnn_units,
                                               output_size=self.output_size, prior_sigma_1=args.prior_sigma_1,
                                               prior_sigma_2=args.prior_sigma_2, prior_pi=args.prior_pi,
                                               posterior_rho_init=args.posterior_rho).to(self.device)
        self.optimizer = optim.Adam(self.bayesian_lstm.parameters(), lr=self.lr)
        self.sample_nbr = args.particles
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self.create_logger()
        self.ckpt_path = self.create_ckpt_path()
        _, _ = self._load_ckpt()
        self.save_hparams(args=args)
        self.distribution = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.future_len = args.future_len if args.future_len is not None else (self.seq_len - self.past_len)

    def data_to_dataset(self, train_data, val_data, test_data, args):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.dataset.get_features_labels(train_data=train_data,
                                                                                                val_data=val_data,
                                                                                                test_data=test_data)
        X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=test_data.shape[0])
        self.seq_len = X_train.shape[1]
        self.output_size = y_train.shape[-1]
        self.num_features = X_train.shape[-1]
        self.num_train_samples = X_train.shape[0]
        return dataloader_train, dataloader_val, dataloader_test

    def create_torch_datasets(self, args):
        if not args.cv:
            train_data, val_data, test_data = self.dataset.get_datasets()
            dataloader_train, dataloader_val, dataloader_test = self.data_to_dataset(train_data=train_data,
                                                                                     val_data=val_data,
                                                                                     test_data=test_data, args=args)
            return dataloader_train, dataloader_val, dataloader_test
        else:
            train_datasets, val_datasets, test_datasets = self.get_dataset_for_crossvalidation(args=args)
            return train_datasets, val_datasets, test_datasets

    def get_dataset_for_crossvalidation(self, args):
        list_train_data, list_val_data, list_test_data = self.dataset.get_data_splits_for_crossvalidation()
        train_datasets, val_datasets, test_datasets = [], [], []
        for train_data, val_data, test_data in zip(list_train_data, list_val_data, list_test_data):
            train_dataset, val_dataset, test_dataset = self.data_to_dataset(train_data=train_data, val_data=val_data,
                                                                            test_data=test_data, args=args)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        return train_datasets, val_datasets, test_datasets[0]

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            out_file = '{}_BayesianLSTM_units_{}_bs_{}_lr_{}_nbr_{}_sigma1_{}_sigma2_{}_pi_{}_rho_{}_cv_{}'.format(
                args.dataset,
                args.rnn_units,
                self.bs, self.lr,
                self.sample_nbr,
                args.prior_sigma_1,
                args.prior_sigma_2,
                args.prior_pi,
                args.posterior_rho,
                args.cv)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            output_folder = os.path.join(args.output_path, out_file, datetime_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            return output_folder

    def get_mse(self, dataset):
        self.bayesian_lstm.eval()
        losses = []
        with torch.no_grad():
            for (X, y) in dataset:
                # forward pass on val_loss.
                y = y.to(self.device)
                preds = self.bayesian_lstm(X)
                loss_batch = self.criterion(preds, y)
                losses.append(loss_batch.cpu().numpy())
        mse = np.mean(losses)
        return mse, preds

    def save_ckpt(self, EPOCH, loss, num_train=1):
        ckpt_path = os.path.join(self.ckpt_path, "Bayesian_LSTM_{}".format(num_train))
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': self.bayesian_lstm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(ckpt_path, 'model.pt'))

    def _load_ckpt(self, num_train=1):
        ckpt_path = os.path.join(self.ckpt_path, "Bayesian_LSTM_{}".format(num_train))
        if os.path.isdir(ckpt_path):
            checkpoint = torch.load(os.path.join(ckpt_path, 'model.pt'))
            self.bayesian_lstm.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            self.start_epoch = epoch
            self.logger.info("loading checkpoint for epoch {} from {}".format(epoch, ckpt_path))
        else:
            epoch = 0
            loss = None
        return epoch, loss

    def compute_test_loss(self, save_particles=True):
        test_loss, preds = self.get_mse(dataset=self.test_dataset)
        return torch.tensor(test_loss).float(), preds.cpu().detach().numpy()

    def get_predictive_distribution(self, save_path):
        self.bayesian_lstm.eval()
        with torch.no_grad():
            for (X_test, y_test) in self.test_dataset:
                preds = [self.bayesian_lstm(X_test) for _ in range(self.mc_samples)]
        preds = torch.stack(preds, dim=1).cpu()  # (B,N,S,F)
        if save_path is not None:
            np.save(save_path, preds)
        self.test_predictive_distribution = preds

    def stochastic_forward_pass_multistep(self, inp_model, future_inp_features, save_path):
        list_predictions = []
        with torch.no_grad():
            for i in range(self.mc_samples):
                inp = inp_model
                for t in range(self.future_len + 1):
                    preds_test = self.bayesian_lstm(inp)  # (B,S,F)
                    last_pred = preds_test[:, -1, :]
                    if t < self.future_len:
                        if future_inp_features is not None:
                            future_inp_features = future_inp_features.to(self.device)
                            last_pred = torch.cat([last_pred, future_inp_features[:, t, :]], dim=-1)
                            last_pred = torch.unsqueeze(last_pred, dim=-2)
                            inp = torch.cat([inp, last_pred], dim=1)
                list_predictions.append(preds_test[:, self.past_len:, :])
            preds_test_MC = torch.stack(list_predictions, dim=1).cpu() # (B,N,len_future,F)
        print("{} multistep predictions sampled from Bayesian LSTM".format(self.mc_samples))
        if save_path is not None:
            np.save(save_path, preds_test_MC)
        return preds_test_MC

    def compute_predictive_interval(self, predictive_distribution, std_multiplier=1.96, save_path=None):
        means = predictive_distribution.mean(axis=1)
        stds = predictive_distribution.std(axis=1)
        ci_upper = means + (std_multiplier * stds)  # (B,S,F)
        ci_lower = means - (std_multiplier * stds)  # (B,S,F)
        mpiw = (ci_upper - ci_lower).mean()
        mpiw_per_timestep = (ci_upper - ci_lower).mean(dim=[0,2]).cpu().numpy()
        if save_path is not None:
            np.save(os.path.join(save_path, "MPIW_per_timestep.npy"), mpiw_per_timestep)
        return ci_lower, ci_upper, mpiw, mpiw_per_timestep

    def compute_PICP_MPIW(self, predictive_distribution, past_len=0, save_path=None):
        inside_pi = 0
        lower_bounds, upper_bounds, MPIW, MPIW_per_timestep = self.compute_predictive_interval(predictive_distribution, save_path=save_path)
        seq_len = lower_bounds.size(1)
        num_samples = lower_bounds.size(0)
        PICP_per_timestep = []
        for (inp, _) in self.test_dataset:
            inp = inp[:, past_len:, :len(self.dataset.target_features)]  # taking only the target features.
            for t in range(seq_len):
                item = inp[:, t, :]  # shape (B,F)
                low_b = lower_bounds[:, t, :]  # (B,F)
                upper_b = upper_bounds[:, t, :]
                bool_low = (low_b <= item).all(dim=-1)
                bool_up = (upper_b >= item).all(dim=-1)
                ic_acc = bool_low * bool_up
                ic_acc = ic_acc.float().sum()
                inside_pi += ic_acc
                PICP_per_timestep.append(ic_acc.numpy() / num_samples)
        PICP_per_timestep = np.stack(PICP_per_timestep)
        if save_path is not None:
            np.save(os.path.join(save_path, "PICP_per_timestep.npy"), PICP_per_timestep)
        PICP = inside_pi / (seq_len * num_samples)
        return PICP, MPIW, PICP_per_timestep, MPIW_per_timestep

    def compute_mse_predictive_distribution(self, alpha):
        self.bayesian_lstm.eval()
        with torch.no_grad():
            for (X_test, _) in self.test_dataset:
                X_test = torch.unsqueeze(X_test, dim=1)
                X_test_tiled = X_test.repeat(repeats=[1, self.mc_samples, 1, 1]).to(self.device)
                mse = self.criterion(self.test_predictive_distribution.to(self.device), alpha * X_test_tiled).cpu()
        return mse

    def _train(self, train_dataset, val_dataset, num_train=1):
        print("starting training ...")
        train_mse_history, val_mse_history = [], []
        for epoch in range(self.EPOCHS):
            for i, (datapoints, labels) in enumerate(train_dataset):
                datapoints, labels = datapoints.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.bayesian_lstm.sample_elbo(inputs=datapoints,
                                                      labels=labels,
                                                      criterion=self.criterion,
                                                      sample_nbr=self.sample_nbr,
                                                      complexity_cost_weight=1 / self.num_train_samples)
                loss.backward()
                self.optimizer.step()

            train_mse, _ = self.get_mse(dataset=train_dataset)
            train_mse_history.append(train_mse)
            val_mse, _ = self.get_mse(dataset=val_dataset)
            val_mse_history.append(val_mse)

            self.logger.info("Epoch: {}/{}".format(str(epoch + 1), self.EPOCHS))
            self.logger.info("Train-Loss: {:.4f}".format(loss))
            self.logger.info("Train-mse: {:.4f}".format(train_mse))
            self.logger.info("Val-mse: {:.4f}".format(val_mse))
            self.save_ckpt(EPOCH=epoch, loss=loss, num_train=num_train)

        # storing history of losses and accuracies in a csv file
        keys = ['train mse', 'val mse']
        values = [train_mse_history, val_mse_history]
        csv_fname = 'bayesian_lstm_history_{}.csv'.format(num_train)
        saving_training_history(keys=keys,
                                values=values,
                                output_path=self.out_folder,
                                csv_fname=csv_fname,
                                logger=self.logger,
                                start_epoch=self.start_epoch)

    def train(self):
        if not self.cv:
            self._train(train_dataset=self.train_dataset, val_dataset=self.val_dataset)
            self.logger.info("Training for a Bayesian LSTM done...")
            self.logger.info('-' * 60)
        else:
            for num_train, (train_dataset, val_dataset) in enumerate(zip(self.train_dataset, self.val_dataset)):
                start_epoch, _ = self._load_ckpt(num_train=num_train + 1)
                self._train(train_dataset=train_dataset, val_dataset=val_dataset, num_train=num_train + 1)
                self.logger.info(
                    "training of a Bayesian LSTM for train/val split number {} done...".format(num_train + 1))
            self.logger.info('-' * 60)

    def plot_preds_targets(self, predictions_test):
        pass
