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
        self.save_hparams(args=args)
        self.distribution = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_torch_datasets(self, args):
        train_data, val_data, test_data = self.dataset.get_datasets()
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

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            out_file = '{}_BayesianLSTM_units_{}_bs_{}_lr_{}_sigma1_{}_sigma2_{}_pi_{}_rho_{}'.format(args.dataset,
                                                                                                      args.rnn_units,
                                                                                                      self.bs, self.lr,
                                                                                                      args.prior_sigma_1,
                                                                                                      args.prior_sigma_2,
                                                                                                      args.prior_pi,
                                                                                                      args.posterior_rho)
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
                preds = self.bayesian_lstm(X)
                loss_batch = self.criterion(preds, y)
                losses.append(loss_batch.cpu().numpy())
        mse = np.mean(losses)
        return mse, preds

    def compute_test_loss(self, save_particles=True):
        test_loss, preds = self.get_mse(dataset=self.test_dataset)
        return torch.tensor(test_loss).float(), preds.cpu().detach().numpy()

    def get_predictive_distribution(self, save_path):
        self.bayesian_lstm.eval()
        with torch.no_grad():
            for (X_test, y_test) in self.test_dataset:
                preds = [self.bayesian_lstm(X_test) for _ in range(self.mc_samples)]
        preds = torch.stack(preds, dim=1).cpu() # (B,N,S,F)
        np.save(save_path, preds)
        self.test_predictive_distribution = preds

    def compute_mse_predictive_distribution(self, alpha):
        self.bayesian_lstm.eval()
        with torch.no_grad():
            for (X_test, _) in self.test_dataset:
                X_test = torch.unsqueeze(X_test, dim=1)
                X_test_tiled = X_test.repeat(repeats=[1, self.mc_samples, 1,1])
                mse = self.criterion(self.test_predictive_distribution, alpha*X_test_tiled).cpu()
        return mse

    def train(self, num_train=1):
        iteration = 0
        print('testing forward pass...')
        with torch.no_grad():
            for (X_test, y_test) in self.test_dataset:
                preds_test = self.bayesian_lstm(X_test)
                print('preds test shape', preds_test.cpu().shape)
                print('preds test example', preds_test.cpu().numpy()[0, :, 0])

        train_mse_history, val_mse_history = [], []
        for epoch in range(self.EPOCHS):
            for i, (datapoints, labels) in enumerate(self.train_dataset):
                datapoints, labels = datapoints.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.bayesian_lstm.sample_elbo(inputs=datapoints,
                                                      labels=labels,
                                                      criterion=self.criterion,
                                                      sample_nbr=self.sample_nbr,
                                                      complexity_cost_weight=1 / self.num_train_samples)
                loss.backward()
                self.optimizer.step()

                iteration += 1
            train_mse, _ = self.get_mse(dataset=self.train_dataset)
            train_mse_history.append(train_mse)
            val_mse, _ = self.get_mse(dataset=self.val_dataset)
            val_mse_history.append(val_mse)

            self.logger.info("Epoch: {}/{}".format(str(epoch+1), self.EPOCHS))
            self.logger.info("Train-Loss: {:.4f}".format(loss))
            self.logger.info("Train-mse: {:.4f}".format(train_mse))
            self.logger.info("Val-mse: {:.4f}".format(val_mse))

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


    def test_cv(self, **kwargs):
        pass
    def plot_preds_targets(self, predictions_test):
        pass
    def compute_PICP_MPIW(self, predictive_distribution, past_len=0):
        pass
    def get_predictive_distribution_multistep(self, save_path):
        pass