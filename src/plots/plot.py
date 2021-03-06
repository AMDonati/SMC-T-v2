import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from src.data_provider.datasets import Dataset, CovidDataset
import numpy as np
import os
import json
import tensorflow as tf
#TODO: change shift to 0.15 and/or change marker size.
#TODO: CHANGE color of Bayesian LSTM.

class Plot:
    def __init__(self, dataset, distribs_path, captions=None, shift_x=0.15, output_path=None,
                 colors=['salmon', 'limegreen', 'seagreen', 'steelblue'], marker='o', markersize=4, alpha_plot=0.8,
                 linewidth=0):
        self.output_path = output_path
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        self.dataset = dataset
        self.distribs_path = distribs_path
        train_data, val_data, test_data = self.dataset.get_datasets()
        self.test_data = test_data[:, :-1, :]  # 24 first timesteps.
        _, _, self.test_dataset = self.dataset.data_to_dataset(train_data=train_data,
                                                                                val_data=val_data,
                                                                                test_data=test_data,
                                                                                num_dim=3)
        # plot params:
        self.get_captions(captions)
        self.color_smc = colors[0]
        self.color_lstm = colors[1]
        self.color_transf = colors[2]
        self.color_bayes = colors[3]
        self.shift_x = shift_x
        self.marker = marker
        self.markersize = markersize
        self.alpha_plot = alpha_plot
        self.linewidth = linewidth

    def get_captions(self, captions):
        if captions is not None:
            with open(os.path.join(captions)) as json_file:
                dict_captions = json.load(json_file)
            self.smc_caption = dict_captions["smc"]
            self.lstm_caption = dict_captions["lstm"]
            self.transf_caption = dict_captions["transf"]
            self.bayes_captions = dict_captions["bayes"]
        else:
            self.smc_caption = "SMC-Transformer - M=30"
            self.lstm_caption = "MC-Dropout LSTM - p=0.1"
            self.transf_caption = "MC-Dropout Transformer - p=0.1"
            self.bayes_captions = "Bayesian LSTM - M=10"

    def plot(self, num_plots=5):
        pass


class CIPlot(Plot):

    def __init__(self, dataset, distribs_path, captions=None, alpha=0.8, variance=0.5, shift_x=0.1, output_path=None,
                 colors=['salmon', 'limegreen', 'seagreen', 'steelblue'], marker='o', markersize=4, alpha_plot=0.8,
                 linewidth=0):
        super(CIPlot, self).__init__(dataset=dataset, distribs_path=distribs_path, captions=captions, shift_x=shift_x,
                                     output_path=output_path, colors=colors, marker=marker, markersize=markersize,
                                     alpha_plot=alpha_plot, linewidth=linewidth)
        self.smc_distrib = self.get_empirical_distribs("smc")
        self.lstm_distrib = self.get_empirical_distribs("lstm")
        self.transf_distrib = self.get_empirical_distribs("transf")
        self.bayes_distrib = self.get_empirical_distribs("bayes")
        self.alpha = alpha
        self.variance = variance

    def get_empirical_distribs(self, prefix):
        distrib = self.dataset.get_data_from_folder(self.distribs_path[prefix])
        print("{} distrib shape".format(prefix), distrib.shape)
        return distrib

    def get_true_CI(self, index, tsp):
        for inp, _ in self.test_dataset:
            inp = inp.numpy()
            mean = np.squeeze(inp[index, tsp])
            lower_bound = mean - 1.96 * np.sqrt(self.variance)
            upper_bound = mean + 1.96 * np.sqrt(self.variance)
            yy = np.linspace(lower_bound, upper_bound, 100)
        return yy

    def get_pred_CI(self, distrib, index, tsp):
        distrib_i_t = distrib[index, :, tsp]
        mean = np.squeeze(np.mean(distrib_i_t, axis=0))
        std = np.squeeze(np.std(distrib_i_t, axis=0))
        lower_b = mean - 1.96 * std
        upper_b = mean + 1.96 * std
        zz = np.linspace(lower_b, upper_b, 100)
        return zz

    def plot_true_mean(self, idx_test):
        for inp, _ in self.test_dataset:
            inp = inp.numpy()
            plt.plot(inp[idx_test, :, 0], color='grey', label='True mean', linewidth=3)

    def plot_predicted_mean(self, distrib, index, color):
        mean = np.squeeze(np.mean(distrib[index], axis=0))
        plt.plot(mean, color=color, linewidth=2)

    def plot_full_distrib(self, x, distrib, index, color):
        sample = distrib[index]
        for i in range(sample.shape[0]):
            plt.scatter(x, sample[i], c=color)

    def _plot(self, idx_test, plot_means=False):
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 1, 1)
        self.plot_true_mean(idx_test)
        if plot_means:
            self.plot_predicted_mean(self.smc_distrib, index=idx_test, color=self.color_smc)
            self.plot_predicted_mean(self.lstm_distrib, index=idx_test, color=self.color_lstm)
            self.plot_predicted_mean(self.transf_distrib, index=idx_test, color=self.color_transf)
            self.plot_predicted_mean(self.bayes_distrib, index=idx_test, color=self.color_bayes)
        for tsp in range(self.test_data.shape[1]):
            yy = self.get_true_CI(index=idx_test, tsp=tsp)
            smc = self.get_pred_CI(distrib=self.smc_distrib, index=idx_test, tsp=tsp)
            lstm = self.get_pred_CI(distrib=self.lstm_distrib, index=idx_test, tsp=tsp)
            transf = self.get_pred_CI(distrib=self.transf_distrib, index=idx_test, tsp=tsp)
            bayes = self.get_pred_CI(distrib=self.bayes_distrib, index=idx_test, tsp=tsp)
            xx = tsp * np.ones(100)
            if tsp == 0:
                plt.plot(xx, yy, color='grey', label='True 95% confidence interval', linewidth=3)
                plt.plot(xx - self.shift_x, smc, color=self.color_smc, label=self.smc_caption, marker=self.marker,
                         markersize=self.markersize, alpha=self.alpha_plot, linewidth=self.linewidth)
                plt.plot(xx + 1 * self.shift_x, lstm, color=self.color_lstm, label=self.lstm_caption,
                         marker=self.marker,
                         markersize=self.markersize, alpha=self.alpha_plot, linewidth=self.linewidth)
                plt.plot(xx + 2 * self.shift_x, transf, color=self.color_transf, label=self.transf_caption,
                         marker=self.marker,
                         markersize=self.markersize, alpha=self.alpha_plot, linewidth=self.linewidth)
                plt.plot(xx + 3 * self.shift_x, bayes, color=self.color_bayes, label=self.bayes_captions,
                         marker=self.marker,
                         markersize=self.markersize, alpha=self.alpha_plot, linewidth=self.linewidth)
            else:
                plt.plot(xx, yy, color='grey', linewidth=3)
                plt.plot(xx - self.shift_x, smc, color=self.color_smc, marker=self.marker,
                         markersize=self.markersize, alpha=self.alpha_plot, linewidth=self.linewidth)
                plt.plot(xx + 1 * self.shift_x, lstm, color=self.color_lstm, marker=self.marker,
                         markersize=self.markersize, alpha=self.alpha_plot, linewidth=self.linewidth)
                plt.plot(xx + 2 * self.shift_x, transf, color=self.color_transf,
                         marker=self.marker,
                         markersize=self.markersize, alpha=self.alpha_plot, linewidth=self.linewidth)
                plt.plot(xx + 3 * self.shift_x, bayes, color=self.color_bayes,
                         marker=self.marker,
                         markersize=self.markersize, alpha=self.alpha_plot, linewidth=self.linewidth)
        plt.ylabel('Signal values', fontsize=14)
        plt.xlabel('Time steps', fontsize=14)
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14, labelright=True)
        plt.grid('on')
        plt.legend(markerscale=3, fontsize=14, frameon=False)
        if self.output_path is not None:
            plt.savefig(os.path.join(self.output_path, "ci_plot_idx_test_{}.pdf".format(idx_test)), bbox_inches="tight")
        else:
            plt.show()

    def _plot_full_distribs(self, plot_means=False):
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 1, 1)
        idx_test = np.random.randint(0, self.test_data.shape[0])
        self.plot_true_mean(idx_test)
        if plot_means:
            self.plot_predicted_mean(self.smc_distrib, index=idx_test, color=self.color_smc)
            self.plot_predicted_mean(self.lstm_distrib, index=idx_test, color=self.color_lstm)
            self.plot_predicted_mean(self.transf_distrib, index=idx_test, color=self.color_transf)
            self.plot_predicted_mean(self.bayes_distrib, index=idx_test, color=self.color_bayes)
        for tsp in range(self.test_data.shape[1]):
            yy = self.get_true_CI(index=idx_test, tsp=tsp)
            xx = tsp * np.ones(100)
            if tsp == 0:
                plt.plot(xx, yy, color='grey', label='True 95% confidence interval', linewidth=3)
            else:
                plt.plot(xx, yy, color='grey', linewidth=3)
        # x = np.linspace(1,self.test_data.shape[1], self.test_data.shape[1])
        x = np.linspace(1, 24, 24)
        self.plot_full_distrib(x=x, distrib=self.smc_distrib, index=idx_test, color=self.color_smc)
        self.plot_full_distrib(x=x, distrib=self.lstm_distrib, index=idx_test, color=self.color_lstm)
        self.plot_full_distrib(x=x, distrib=self.bayes_distrib, index=idx_test, color=self.color_bayes)
        self.plot_full_distrib(x=x, distrib=self.transf_distrib, index=idx_test, color=self.color_transf)
        plt.ylabel('Signal values', fontsize=16)
        plt.xlabel('Time steps', fontsize=16)
        plt.grid('on')
        plt.legend(markerscale=3, fontsize=14, frameon=False)
        if self.output_path is not None:
            plt.savefig(os.path.join(self.output_path, "ci_plot_idx_test_{}".format(idx_test)))
        else:
            plt.show()


    def plot(self, num_plots=1):
        idx_test = 69
        # idx_test = np.random.randint(0, self.test_data.shape[0])
        for _ in range(num_plots):
            self._plot(idx_test=idx_test)
            #self._plot_full_distribs()


class PICP_MPIW_Plot(Plot):
    def __init__(self, dataset, distribs_path, captions=None, shift_x=0.2, output_path=None,
                 colors=['firebrick', 'limegreen', 'seagreen', 'blueviolet'], marker='o', markersize=4, alpha_plot=0.8,
                 linewidth=0):
        super(PICP_MPIW_Plot, self).__init__(dataset=dataset, distribs_path=distribs_path, captions=captions,
                                             shift_x=shift_x,
                                             output_path=output_path, colors=colors, marker=marker,
                                             markersize=markersize,
                                             alpha_plot=alpha_plot, linewidth=linewidth)

        self.smc_results = self.upload_npy_files("smc")
        self.lstm_results = self.upload_npy_files("lstm")
        self.transf_results = self.upload_npy_files("transf")
        self.bayes_results = self.upload_npy_files("bayes")
        self.seq_len = self.smc_results.shape[0]

    def upload_npy_files(self, prefix):
        metric = np.load(self.distribs_path[prefix])
        print("{} metric shape".format(prefix), metric.shape)
        return metric

    def plot(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        x = np.linspace(1, self.seq_len, self.seq_len)
        ax.bar(x, self.smc_results, width=0.2, label=self.smc_caption)
        ax.plot(x, self.smc_results, color="red")
        ax.set_xticks(x)
        ax.legend()
        plt.show()


if __name__ == '__main__':
    # load dataset
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-dataset", type=str, default='synthetic', help='dataset selection')
    parser.add_argument("-dataset_model", type=int, default=1, help="model 1 or 2 for the synthetic dataset.")
    parser.add_argument("-data_path", type=str, default="../../data/synthetic_model_1_pytorch4vm",
                        help="path for uploading the dataset")
    parser.add_argument("-output_path", type=str, default="../../output/plots/ci_plots/synthetic_model_1",
                        help="path for saving the plot")
    parser.add_argument("-smc", type=str, default="../../output/plots/ci_plots/synthetic_model_1/smc_t",
                        help="path for the smc distrib npy file.")
    parser.add_argument("-lstm", type=str, default="../../output/plots/ci_plots/synthetic_model_1/lstm",
                        help="path for the lstm distrib npy file.")
    parser.add_argument("-transf", default="../../output/plots/ci_plots/synthetic_model_1/baseline_t", type=str,
                        help="path for the transformer distrib npy file.")
    parser.add_argument("-bayes", type=str, default="../../output/plots/ci_plots/synthetic_model_1/bayesian_lstm",
                        help="path for the bayesian lstm distrib npy file.")
    parser.add_argument("-captions", type=str, help="path for the captions json file.")
    parser.add_argument("-alpha", type=float, default=0.8, help="alpha value in synthetic models 1 & 2.")
    parser.add_argument("-beta", type=float, default=0.54, help="beta value for synthetic model 2")
    parser.add_argument('-p', type=float, default=0.7, help="p value for synthetic model 2.")
    parser.add_argument('-variance', type=float, default=0.5, help="variance value for synthetic model.")

    args = parser.parse_args()

    dataset = Dataset(data_path=args.data_path, model=args.dataset_model, name=args.dataset)

    distribs_path = {"smc": args.smc, "lstm": args.lstm, "transf": args.transf, "bayes": args.bayes}

    ci_plot = CIPlot(dataset=dataset, distribs_path=distribs_path, captions=args.captions, alpha=args.alpha,
                     variance=args.variance, output_path=args.output_path)

    ci_plot.plot()
