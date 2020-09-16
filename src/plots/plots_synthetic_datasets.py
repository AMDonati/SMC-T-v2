from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_confidence_intervals(test_data, dict_smc_t_preds, list_smct_sigmas, dict_lstm_preds, alpha=0.9, var=0.5, num_samples=1000, shift_x=0.2):
    true_mean = alpha * test_data
    plt.figure(figsize=(15, 10))
    dict_smct_intervals = dict(list(dict_smc_t_preds.keys()), [np.zeros((2, 4, test_data.shape[2] - 1))]*len(dict_smc_t_preds))
    dict_smct_samples = dict(list(dict_smc_t_preds.keys()), []*len(dict_smc_t_preds))
    dict_zz_transf = dict.fromkeys(dict_smc_t_preds.keys())
    for j in range(1):
        plt.subplot(1, 1, j + 1)
        idx_test = np.random.randint(0, test_data.shape[0] - 1)
        plt.plot(true_mean[idx_test, 0, :-1, 0], color='darkcyan', label='True mean', linewidth=3)
        for tsp in range(test_data.shape[2] - 1):
            lower_b = true_mean[idx_test, 0, tsp, 0] - 1.96 * np.sqrt(var)
            upper_b = true_mean[idx_test, 0, tsp, 0] + 1.96 * np.sqrt(var)
            yy = np.linspace(lower_b, upper_b, 10)
            for key, preds, interval, sample, var in zip(dict_smc_t_preds.keys(), dict_smc_t_preds.values, dict_smct_intervals.values, dict_smct_samples.values, list_smct_sigmas):
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
                dict_zz_transf[key] = zz_transf
            xx = tsp * np.ones(10)
            xx_samp = tsp * np.ones(num_samples)
            if tsp == 0:
                plt.plot(xx, yy, color='darkcyan', label='True 95% confidence interval', linewidth=3)
                # plt.plot(xx_samp-shift_x,lst_samp,color = 'salmon', label = 'Samples from estimated distribution  - SMC-Transformer (d = 8, M = 10)', marker = 'o', markersize = 3, alpha = 0.8,linewidth = 0)
                plt.plot(xx_samp - shift_x, lst_samp_2, color='firebrick',
                         label='Samples from estimated distribution  - SMC-Transformer (d = 16, M = 30)', marker='o',
                         markersize=4, alpha=0.8, linewidth=0)
            else:
                plt.plot(xx, yy, color='darkcyan', linewidth=3)
                plt.plot(xx_samp - shift_x, lst_samp_2, color='firebrick', marker='o', markersize=4, alpha=0.8,
                         linewidth=0)
        plt.ylabel('Signal values', fontsize=16)
        plt.xlabel('Time steps', fontsize=16)
        plt.grid('on')
        plt.legend(markerscale=3, fontsize=14, frameon=False)
