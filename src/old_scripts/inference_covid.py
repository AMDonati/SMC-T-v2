import numpy as np
import tensorflow as tf
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
import os
from utils.utils_train import restoring_checkpoint, CustomSchedule
from eval.inference_functions import split_input_target, EM_after_training, inference_onestep, inference_multistep, get_distrib_all_timesteps
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../../data/test/covid.npy",
                        help="npy file with input data") #TODO: see here we use a Dataset instead.
    args = parser.parse_args()
    # ---------- Load Test Data -----------------------------------------------------------------------------------------------------------------

    test_data = np.load(args.data_path)

    # ---------- Load SMC Transformer with learned params ------------------------------------------------------------------------------------
    out_path = "../../output/covid_SMC_T/covid_smc_t_10_p"
    list_sigmas = [0.5045, 0.4787, 0.4313, 0.5879] #TODO: take it from a save dict.
    dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], list_sigmas))
    Sigma_obs = 0.0368
    num_particles = 10

    # out_path = "../../output/covid_SMC_T/covid_smc_t_60_p"
    # list_sigmas = [0.51636, 0.49633, 0.46728, 0.48299]
    # dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], list_sigmas))
    # Sigma_obs = 0.03406
    # num_particles = 60

    d_model = 8 #TODO: take this from a config file.
    dff = 16
    N = 1

    smc_transformer = SMC_Transformer(d_model=d_model,
                                      output_size=1,
                                      seq_len=60,
                                      full_model=True,
                                      dff=dff)

    # get checkpoint path for SMC_Transformer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    checkpoint_path = os.path.join(out_path, "checkpoints")
    smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer_1")
    smc_T_ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                                     optimizer=optimizer)
    smc_T_ckpt_manager = tf.train.CheckpointManager(smc_T_ckpt, smc_T_ckpt_path, max_to_keep=50)
    num_epochs_smc_T = restoring_checkpoint(ckpt_manager=smc_T_ckpt_manager, ckpt=smc_T_ckpt,
                                            args_load_ckpt=True, logger=None)

    smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas, sigma_obs=Sigma_obs, num_particles=num_particles)

    # ------------------------------------- check test loss ----------------------------------------------------------------------------------

    # computing loss on test_dataset:
    inputs, targets = split_input_target(test_data[:,np.newaxis,:,:])
    (preds_test, preds_test_resampl), _, _ = smc_transformer(inputs=inputs,
                                                            targets=targets)  # predictions test are the ones not resampled.
    test_metric_avg_pred = tf.keras.losses.MSE(targets, tf.reduce_mean(preds_test, axis=1, keepdims=True))  # (B,1,S)
    test_metric_avg_pred = tf.reduce_mean(test_metric_avg_pred, axis=[1,2])
    top_k, top_i = tf.math.top_k(test_metric_avg_pred, k=15)
    print('indices with lowest loss', top_i)
    test_metric_avg_pred = tf.reduce_mean(test_metric_avg_pred)
    print('test loss', test_metric_avg_pred)

    # ------ sigmas estimation post-training --------------------------------------------------------------------------------------------------
    indexes = [72,2,76,77,7,88,82,78,75,74,73,70,67,66,53,41,33,10] #TODO: put this as a parser arg.
    #indexes = [11]
    for index in indexes:
        test_sample = test_data[index]
        print('test_sample', test_sample)
        test_sample = tf.convert_to_tensor(test_sample)
        test_sample = tf.reshape(test_sample, shape=(1, 1, test_sample.shape[-2], test_sample.shape[-1]))
        inputs, targets = split_input_target(test_sample[:, :, :41, :])
        smc_transformer.seq_len = 40
        save_path_EM = os.path.join(out_path, 'sigmas_after_EM_{}.npy'.format(index))
        Sigma_obs, sigmas = EM_after_training(smc_transformer=smc_transformer, inputs=inputs, targets=targets, save_path=save_path_EM)

        # ---------------------------- launching inference ----------------------------------------------------------------------------------------
        save_path_means = os.path.join(out_path, 'mean_preds_noresampling_sample_{}_59_timesteps.npy'.format(index, N))
        save_path_means_multi = os.path.join(out_path, 'mean_preds_sample_{}_N_{}_multi.npy'.format(index, N))
        save_path_preds_multi = os.path.join(out_path, 'particules_sample_{}_N_{}_multi.npy'.format(index, N))
        save_path_distrib = os.path.join(out_path, 'distrib_future_timesteps_noresampling_sample_{}_59_timesteps.npy'.format(index, N))
        save_path_distrib_multi = os.path.join(out_path, 'distrib_future_timesteps_sample_{}_N_{}_multi.npy'.format(index, N))

        smc_transformer.seq_len = 60
        preds_NP, mean_preds = inference_onestep(smc_transformer=smc_transformer,
                                                 test_sample=test_sample,
                                                 save_path=save_path_means,
                                                 past_len=1)


        #preds_multi, mean_preds_multi = inference_multistep(smc_transformer, test_sample,
                                                            #save_path=save_path_means_multi)

        #print('preds_multi', preds_multi.shape)
        sigma_obs = tf.math.sqrt(smc_transformer.cell.Sigma_obs)
        P = smc_transformer.cell.num_particles

        get_distrib_all_timesteps(preds_NP, sigma_obs=sigma_obs, P=P, save_path_distrib=save_path_distrib, len_future=59)
        #get_distrib_all_timesteps(preds_multi, sigma_obs=sigma_obs, P=P, save_path_distrib=save_path_distrib_multi)