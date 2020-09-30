from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from src.data_provider.datasets import Dataset, CovidDataset

def plot_confidence_intervals(test_data, list_smc_t_preds, list_smct_sigmas, list_captions=None, list_lstm_preds=None, alpha=0.8, var=0.5, num_samples=1000, shift_x=0.2, save_path=None):
    '''
    :param test_data: numpy array for targets: shape (B,S,1).
    :param list_smc_t_preds: preds particules shape (B,P,S,1)
    :param list_smct_sigmas:
    :param dict_lstm_preds: mc preds shape (B,N,S,1)
    :param alpha:
    :param var: true variance of the synthetic model.
    :param num_samples:
    :param shift_x:
    :return:
    '''
    if list_lstm_preds is not None:
        lstm_means = [np.mean(item, axis=1) for item in list_lstm_preds]
        lstm_std = [np.std(item, axis=1) for item in list_lstm_preds]
    test_data = test_data[:,np.newaxis, :-1, :]
    true_mean = alpha * test_data
    plt.figure(figsize=(15, 10))
    list_smct_intervals, list_smct_samples, list_zz_transf = [], [], []
    for j in range(1):
        plt.subplot(1, 1, j + 1)
        idx_test = np.random.randint(0, test_data.shape[0] - 1)
        plt.plot(true_mean[idx_test, 0, :-1, 0], color='darkcyan', label='True mean', linewidth=3)
        for tsp in range(test_data.shape[2] - 1):
            lower_b = true_mean[idx_test, 0, tsp, 0] - 1.96 * np.sqrt(var)
            upper_b = true_mean[idx_test, 0, tsp, 0] + 1.96 * np.sqrt(var)
            yy = np.linspace(lower_b, upper_b, 10)
            for preds, var in zip(list_smc_t_preds, list_smct_sigmas):
                sample, interval = [], np.zeros(shape=(2, 4, test_data.shape[2] - 1))
                Gauss = np.random.normal(0, 1, num_samples)
                idx = np.sum(np.random.multinomial(1, np.ones(10) / 10, num_samples) * np.arange(10), axis=1)
                for i in range(num_samples):
                    samp = preds[idx_test, idx[i], tsp + 1, 0] + np.sqrt(var) * Gauss[i]
                    sample.append(samp)
                    if samp > lower_b:
                        if samp < upper_b:
                            interval[j, 0, tsp] = interval[j, 0, tsp] + 1
                mean_samp = np.mean(sample)
                std_samp = np.std(sample)
                zz_transf = np.linspace(mean_samp - 1.96 * std_samp, mean_samp + 1.96 * std_samp, 10)
                list_zz_transf.append(zz_transf)
                list_smct_samples.append(sample)
                list_smct_intervals.append(interval)

            # compute predictive interval for lstm preds:
            lstm_ints = []
            for mean, std in zip(lstm_means, lstm_std):
                zz = np.linspace(mean[idx_test, tsp + 1, 0] - 1.96 *
                             std[idx_test, tsp + 1, 0],
                             mean[idx_test, tsp + 1, 0] + 1.96 *
                             std[idx_test, tsp + 1, 0], 10)
                lstm_ints.append(zz)
            xx = tsp * np.ones(10)
            xx_samp = tsp * np.ones(num_samples)
            colors_smc = ['salmon','firebrick']
            colors_smc = colors_smc[:len(list_smct_samples)]
            colors_lstm = ['limegreen', 'seagreen', 'darkgreen', 'blueviolet', 'indigo', 'mediumorchid']
            colors_lstm = colors_lstm[:len(lstm_ints)]
            if tsp == 0:
                plt.plot(xx, yy, color='darkcyan', label='True 95% confidence interval', linewidth=3)
                for col_s, samples in zip(colors_smc, list_smct_samples):
                    plt.plot(xx_samp - shift_x, samples, color=col_s,
                         label='Samples from estimated distribution  - SMC-Transformer (d = 16, M = 30)', marker='o',
                         markersize=4, alpha=0.8, linewidth=0)
                for index, (col_l, int) in enumerate(zip(colors_lstm, lstm_ints)):
                    plt.plot(xx + (index+1) * shift_x, int, color=col_l, label='Samples - LSTM (d = 32, drop = 0.2)', marker='o',
                         markersize=4, alpha=0.8, linewidth=0)
            else:
                plt.plot(xx, yy, color='darkcyan', linewidth=3)
                for col_s, samples in zip(colors_smc, list_smct_samples):
                    plt.plot(xx_samp - shift_x, samples, color=col_s,
                             marker='o',
                             markersize=4, alpha=0.8, linewidth=0)
                for index, (col_l, int) in enumerate(zip(colors_lstm, lstm_ints)):
                    plt.plot(xx + (index + 1) * shift_x, int, color=col_l,
                             marker='o',
                             markersize=4, alpha=0.8, linewidth=0)

        plt.ylabel('Signal values', fontsize=16)
        plt.xlabel('Time steps', fontsize=16)
        plt.grid('on')
        plt.legend(markerscale=3, fontsize=14, frameon=False)
        plt.show()

if __name__ == '__main__':
    # load dataset
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-dataset", type=str, default='synthetic', help='dataset selection')
    parser.add_argument("-dataset_model", type=int, default=1, help="model 1 or 2 for the synthetic dataset.")
    parser.add_argument("-data_path", type=str, default="../../data/synthetic_model_1", help="path for uploading the dataset")
    parser.add_argument("-alpha", type=float, default=0.8, help="alpha value in synthetic models 1 & 2.")
    parser.add_argument("-beta", type=float, default=0.54, help="beta value for synthetic model 2")
    parser.add_argument('-p', type=float, default=0.7, help="p value for synthetic model 2.")

    args = parser.parse_args()

    dataset = Dataset(data_path=args.data_path, model=args.dataset_model, name=args.dataset)
    _, _, test_data = dataset.get_datasets()
    preds_path_1 ="../../output/exp_synthetic_model_1/synthetic_Recurrent_T_depth_8_bs_32_fullmodel_True_dff_8_attn_w_None__p_10_SigmaObs_0.5_sigmas_0.5/20200914-112003/particles_preds_test.npy"
    smc_preds_1 = np.load(preds_path_1)
    print(smc_preds_1.shape)
    sigma = 0.53

    lstm_preds_path_1 = '../../output/exp_synthetic_model_1/lstm_results/synthetic_LSTM_units_32_pdrop_0.1_rnndrop_0.0_lr_0.001_bs_32_cv_0/20200916-061351/inference_results/mc_dropout_samples_test_data_unistep.npy'

    lstm_preds_1 = np.load(lstm_preds_path_1)
    lstm_preds_1 = lstm_preds_1[:,:,:,np.newaxis]
    print(lstm_preds_1.shape)

    plot_confidence_intervals(test_data=test_data, list_smc_t_preds=[smc_preds_1], list_smct_sigmas=[sigma], list_lstm_preds=[lstm_preds_1])