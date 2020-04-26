import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import pandas as pd

if __name__ == "__main__":

  eval_output_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/exp_162_grad_not_zero_azure/time_series_multi_unistep-forcst_heads_1_depth_3_dff_12_pos-enc_50_pdrop_0.1_b_1048_cs_True__particles_25_noise_True_sigma_0.1_smc-pos-enc_None/eval_outputs'


  output_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/exp_reg_162_loss_modified_grad_not0/time_series_' \
                'multi_unistep-forcst_heads_4_depth_12_dff_48_pos-enc_50_pdrop_0.1_b_1048_cs_True__particles_25_noise_True_sigma_0.05_smc-pos-enc_None'

  eval_output_path = output_path + '/eval_outputs'
  #predictions_N_1_uni_step_path = eval_output_path + '/' + 'pred_unistep_N_1_test.npy'
  predictions_test_path = eval_output_path + '/' + 'pred_unistep_N_1_test_unnorm.npy'
  #targets_test_path = eval_output_path + '/' + 'targets_test.npy'
  targets_test_path = eval_output_path + '/' + 'targets_test_unnorm.npy'


  #Load data
  particles = np.load(predictions_test_path)
  targets = np.load(targets_test_path)
  print('particles', particles.shape)
  print('targets', targets.shape)

  #Display one test sample with the associated predictions

  id_sample = 87
  plt.figure(figsize=(12, 8))
  plt.plot(targets[id_sample][0:23], lw=3, color = 'plum', label='True observations')
  plt.plot(particles[id_sample][0][1:24], lw=1, alpha = 0.5, color = 'darkcyan', linestyle = 'dashed', label='Samples from the predictive probability')
  for j in range(24):
      plt.plot(particles[id_sample][j+1][1:24], lw=1, color = 'darkcyan', linestyle = 'dashed', alpha = 0.5)
  plt.legend(fontsize=14)
  plt.title('Test sample', fontsize=16)
  plt.xlabel('Time steps (hours)', fontsize=14)
  plt.ylabel('Pressure (hPa)', fontsize=14)
  plt.tight_layout()
  fig_path = eval_output_path + '/particles_vs_targets_plot_sample-id_{}.png'.format(id_sample)
  plt.savefig(fig_path)
  #plt.show()

  #Boxplot to display error over 100 samples

  predictions = np.zeros((100,24,1))
  for i in range(100):
      moy = np.mean(particles[i],axis=0)
      predictions[i] = moy
  predictions.shape

  hours = []
  errs  = []
  for j in range(12):
      errs = np.append(errs,targets[:,j]-predictions[:,j+1])
      for ell in range(100):
          hours = np.append(hours,j+1)
  errs.shape
  d = {'hours': hours, 'errors': errs}
  df = pd.DataFrame(d)

  plt.figure(figsize=(12, 6))
  sns.boxplot(x="hours", y="errors", data=d, whis=np.inf)
  sns.swarmplot(x="hours", y="errors", data=d)
  plt.title('Error distribution over test samples', fontsize=16)
  plt.xlabel('Time steps', fontsize=14)
  plt.ylabel('Errors of predicted targets', fontsize=14)
  plt.tight_layout()

  plt.show()


