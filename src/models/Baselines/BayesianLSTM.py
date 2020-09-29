from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator
import torch.nn as nn
import torch


@variational_estimator
class BayesianLSTMModel(nn.Module):
    def __init__(self, input_size, rnn_units, output_size, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1.0, posterior_rho_init=-6.0):
        super(BayesianLSTMModel, self).__init__()
        self.lstm_1 = BayesianLSTM(input_size, rnn_units, prior_sigma_1=prior_sigma_1, prior_sigma_2=prior_sigma_2, prior_pi=prior_pi, posterior_rho_init=posterior_rho_init)
        self.linear = nn.Linear(rnn_units, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        x_, _ = self.lstm_1(x)
        x_ = self.linear(x_)
        return x_