from utils.utils_train import CustomSchedule, restoring_checkpoint, write_to_csv
import tensorflow as tf
import os
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from train.train_functions import train_SMC_transformer
from eval.inference_functions import inference_onestep, inference_multistep, get_distrib_all_timesteps
from algos.generic import Algo
import json
import datetime


class SMCTAlgo(Algo):
    def __init__(self, dataset, args):
        super(SMCTAlgo, self).__init__(dataset=dataset, args=args)
        self.lr = CustomSchedule(args.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr,
                                                  beta_1=0.9,
                                                  beta_2=0.98,
                                                  epsilon=1e-9)
        self.out_folder = self._create_out_folder(args=args)
        self.logger = self.create_logger()
        self.ckpt_path = self.create_ckpt_path()
        self.save_hparams(args)
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=4)
        self.smc_transformer = SMC_Transformer(d_model=args.d_model,
                                               output_size=self.output_size,
                                               seq_len=self.seq_len,
                                               full_model=args.full_model,
                                               dff=args.dff,
                                               attn_window=args.attn_w)
        self._init_SMC_T(args=args)
        self.sigmas_after_training = None
        self.ckpt_manager, _ = self._load_ckpt()

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            out_file = '{}_Recurrent_T_depth_{}_bs_{}_fullmodel_{}_dff_{}_attn_w_{}'.format(args.dataset, args.d_model,
                                                                                            self.bs, args.full_model,
                                                                                            args.dff, args.attn_w)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            if args.smc:
                out_file = out_file + '__p_{}'.format(args.particles)
                out_file = out_file + '_SigmaObs_{}'.format(args.sigma_obs)
                out_file = out_file + '_sigmas_{}'.format(args.sigmas)
            out_folder = os.path.join(self.output_path, out_file, datetime_folder)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            return out_folder

    def _init_SMC_T(self, args):
        if args.smc:
            self.logger.info("SMC Transformer for {} particles".format(args.particles))
            if args.sigmas is not None:
                dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [args.sigmas for _ in range(4)]))
            else:
                dict_sigmas = None
            self.smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas, sigma_obs=args.sigma_obs,
                                                         num_particles=args.particles)
            assert self.smc_transformer.cell.noise == self.smc_transformer.cell.attention_smc.noise == True
            self.logger.info("Sigma_obs init: {}".format(self.smc_transformer.cell.Sigma_obs))

    def _check_consistency_hparams(self, args):
        if args.save_path is not None:
            with open(os.path.join(self.save_path, "config.json")) as json_file:
                dict_hparams = json.load(json_file)
            assert int(dict_hparams["d_model"]) == args.d_model, "consistency error in d_model parameter"
            assert int(dict_hparams["dff"]) == args.dff, "consistency error in dff parameter"
            assert dict_hparams["attn_w"] == str(args.attn_w), "consistency error in attn_w parameter"
            assert dict_hparams["full_model"] == args.full_model, "consistency error in full_model parameter"

    def train(self):
        self.logger.info('hparams...')
        self.logger.info(
            'd_model: {} - batch size {} - full model? {} - dff: {} -attn window: {}'.format(
                self.smc_transformer.d_model, self.bs,
                self.smc_transformer.full_model, self.smc_transformer.dff,
                self.smc_transformer.cell.attention_smc.attn_window))
        if not self.cv:
            train_SMC_transformer(smc_transformer=self.smc_transformer,
                                  optimizer=self.optimizer,
                                  EPOCHS=self.EPOCHS,
                                  train_dataset=self.train_dataset,
                                  val_dataset=self.val_dataset,
                                  ckpt_manager=self.ckpt_manager,
                                  logger=self.logger,
                                  start_epoch=self.start_epoch)
            self.sigmas_after_training = dict(zip(['sigma_obs', 'k', 'q', 'v', 'z'],
                                                  [self.smc_transformer.cell.Sigma_obs,
                                                   self.smc_transformer.cell.attention_smc.sigma_k.numpy(),
                                                   self.smc_transformer.cell.attention_smc.sigma_q.numpy(),
                                                   self.smc_transformer.cell.attention_smc.sigma_v.numpy(),
                                                   self.smc_transformer.cell.attention_smc.sigma_z.numpy()]))
            dict_json = {key: str(value) for key, value in self.sigmas_after_training.items()}
            final_sigmas_path = os.path.join(self.out_folder, "sigmas_after_training.json")
            with open(final_sigmas_path, 'w') as fp:
                json.dump(dict_json, fp)  # TODO: add this at each checkpoint saving?
            self.logger.info('-' * 60)
        else:
            for num_train, (train_dataset, val_dataset) in enumerate(zip(self.train_dataset, self.val_dataset)):
                ckpt_manager, start_epoch = self._load_ckpt(num_train=num_train + 1)
                train_SMC_transformer(smc_transformer=self.smc_transformer,
                                      optimizer=self.optimizer,
                                      EPOCHS=self.EPOCHS,
                                      train_dataset=train_dataset,
                                      val_dataset=val_dataset,
                                      ckpt_manager=ckpt_manager,
                                      logger=self.logger,
                                      start_epoch=start_epoch) #TODO: add a num_train argument here...
                sigmas_after_training = dict(zip(['sigma_obs', 'k', 'q', 'v', 'z'],
                                                      [self.smc_transformer.cell.Sigma_obs,
                                                       self.smc_transformer.cell.attention_smc.sigma_k.numpy(),
                                                       self.smc_transformer.cell.attention_smc.sigma_q.numpy(),
                                                       self.smc_transformer.cell.attention_smc.sigma_v.numpy(),
                                                       self.smc_transformer.cell.attention_smc.sigma_z.numpy()]))
                dict_json = {key: str(value) for key, value in sigmas_after_training.items()}
                final_sigmas_path = os.path.join(self.out_folder, "sigmas_after_training_{}.json".format(num_train + 1))
                with open(final_sigmas_path, 'w') as fp:
                    json.dump(dict_json, fp)  # TODO: add this at each checkpoint saving?
                if num_train == 0:
                    self.sigmas_after_training = sigmas_after_training
                self.logger.info(
                    "training of a SMC Transformer for train/val split number {} done...".format(num_train + 1))
                self.logger.info('-' * 60)

    def _load_ckpt(self, num_train=1):
        # creating checkpoint manager
        ckpt = tf.train.Checkpoint(transformer=self.smc_transformer,
                                   optimizer=self.optimizer)
        smc_T_ckpt_path = os.path.join(self.ckpt_path, "SMC_transformer_{}".format(num_train))
        ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path, max_to_keep=50)
        # if a checkpoint exists, restore the latest checkpoint.
        start_epoch = restoring_checkpoint(ckpt_manager=ckpt_manager, ckpt=ckpt, args_load_ckpt=True,
                                           logger=self.logger)
        if start_epoch is not None:
            self.start_epoch = start_epoch
        else:
            start_epoch = 0
        if self.save_path is not None:
            #self._check_consistency_hparams(args)
            with open(os.path.join(self.save_path, "sigmas_after_training.json")) as json_file:
                dict_json = json.load(json_file)
            self.sigmas_after_training = {key: float(value) for key, value in dict_json.items()}
            self._reinit_sigmas()
        return ckpt_manager, start_epoch

    def _reinit_sigmas(self):
        if self.sigmas_after_training is not None:
            dict_sigmas = {key: self.sigmas_after_training[key] for key in ['k', 'q', 'v', 'z']}
            sigma_obs = self.sigmas_after_training["sigma_obs"]
            self.smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas, sigma_obs=sigma_obs,
                                                         num_particles=self.smc_transformer.cell.num_particles)

    def _EM_after_training(self, inputs, targets, index, iterations=30):
        targets_tiled = tf.tile(targets, multiples=[1, self.smc_transformer.cell.num_particles, 1, 1])
        for it in range(1, iterations + 1):
            (preds, preds_resampl), _, _ = self.smc_transformer(inputs=inputs,
                                                                targets=targets)
            # EM estimation of the noise parameters
            err_k = self.smc_transformer.noise_K_resampled * self.smc_transformer.noise_K_resampled
            err_k = tf.reduce_mean(err_k)
            err_q = self.smc_transformer.noise_q * self.smc_transformer.noise_q
            err_q = tf.reduce_mean(err_q)
            err_v = self.smc_transformer.noise_V_resampled * self.smc_transformer.noise_V_resampled
            err_v = tf.reduce_mean(err_v)
            err_z = self.smc_transformer.noise_z * self.smc_transformer.noise_z
            err_z = tf.reduce_mean(err_z)
            # EM estimation of Sigma_obs:
            err_obs = tf.cast(targets_tiled, tf.float32) - tf.cast(preds_resampl, tf.float32)
            new_sigma_obs = err_obs * err_obs
            new_sigma_obs = tf.reduce_mean(new_sigma_obs)
            # update of the sigmas:
            self.smc_transformer.cell.attention_smc.sigma_v = (1 - it ** (
                -0.6)) * self.smc_transformer.cell.attention_smc.sigma_v + it ** (
                                                                  -0.6) * err_v
            self.smc_transformer.cell.attention_smc.sigma_k = (1 - it ** (
                -0.6)) * self.smc_transformer.cell.attention_smc.sigma_k + it ** (
                                                                  -0.6) * err_k
            self.smc_transformer.cell.attention_smc.sigma_q = (1 - it ** (
                -0.6)) * self.smc_transformer.cell.attention_smc.sigma_q + it ** (
                                                                  -0.6) * err_q
            self.smc_transformer.cell.attention_smc.sigma_z = (1 - it ** (
                -0.6)) * self.smc_transformer.cell.attention_smc.sigma_z + it ** (
                                                                  -0.6) * err_z
            self.smc_transformer.cell.Sigma_obs = (1 - it ** (-0.6)) * self.smc_transformer.cell.Sigma_obs + it ** (
                -0.6) * new_sigma_obs
            print('it:', it)
            print("sigma_obs: {}, sigma_k: {}, sigma_q: {}, sigma_v: {}, sigma_z: {}".format(
                self.smc_transformer.cell.Sigma_obs,
                self.smc_transformer.cell.attention_smc.sigma_k,
                self.smc_transformer.cell.attention_smc.sigma_q,
                self.smc_transformer.cell.attention_smc.sigma_v,
                self.smc_transformer.cell.attention_smc.sigma_z
            ))

        dict_sigmas = dict(zip(['sigma_obs', 'sigma_k', 'sigma_q', 'sigma_v', 'sigma_z'],
                               [self.smc_transformer.cell.Sigma_obs,
                                self.smc_transformer.cell.attention_smc.sigma_k,
                                self.smc_transformer.cell.attention_smc.sigma_q,
                                self.smc_transformer.cell.attention_smc.sigma_v,
                                self.smc_transformer.cell.attention_smc.sigma_z]))
        write_to_csv(output_dir=os.path.join(self.inference_path, "sigmas_after_EM_{}.csv".format(index)),
                     dic=dict_sigmas)

    def launch_inference(self, **kwargs):
        # create inference folder
        self.inference_path = os.path.join(self.out_folder, "inference_results")
        if not os.path.isdir(self.inference_path):
            os.makedirs(self.inference_path)
        for index in kwargs["list_samples"]:
            inputs, targets, test_sample = self.dataset.get_data_sample_from_index(index, past_len=self.past_len)
            self.smc_transformer.seq_len = self.past_len
            self._EM_after_training(inputs=inputs, targets=targets, index=index)
            save_path_means, save_path_means_multi, save_path_preds_multi, save_path_distrib, save_path_distrib_multi = self._get_inference_paths(
                index=index)
            self.smc_transformer.seq_len = self.seq_len
            preds_NP, mean_preds = inference_onestep(smc_transformer=self.smc_transformer,
                                                     test_sample=test_sample,
                                                     save_path=save_path_means,
                                                     past_len=1)
            if kwargs["multistep"]:
                preds_multi, mean_preds_multi = inference_multistep(self.smc_transformer, test_sample,
                                                                    save_path=save_path_means_multi,
                                                                    past_len=self.past_len,
                                                                    future_len=self.seq_len - self.past_len)

            # print('preds_multi', preds_multi.shape)
            sigma_obs = tf.math.sqrt(self.smc_transformer.cell.Sigma_obs)

            get_distrib_all_timesteps(preds_NP, sigma_obs=sigma_obs, P=self.smc_transformer.cell.num_particles,
                                      save_path_distrib=save_path_distrib,
                                      len_future=self.seq_len - 1)
            if kwargs["multistep"]:
                get_distrib_all_timesteps(preds_multi, sigma_obs=sigma_obs, P=self.smc_transformer.cell.num_particles,
                                          save_path_distrib=save_path_distrib_multi,
                                          len_future=self.seq_len - self.past_len)
            self._reinit_sigmas()

    def _get_inference_paths(self, index):
        save_path_means = os.path.join(self.inference_path, 'mean_preds_sample_{}.npy'.format(index))
        save_path_means_multi = os.path.join(self.inference_path, 'mean_preds_sample_{}_multi.npy'.format(index))
        save_path_preds_multi = os.path.join(self.inference_path, 'particules_sample_{}_multi.npy'.format(index))
        save_path_distrib = os.path.join(self.inference_path,
                                         'distrib_future_timesteps_sample_{}.npy'.format(
                                             index))
        save_path_distrib_multi = os.path.join(self.inference_path,
                                               'distrib_future_timesteps_sample_{}_multi.npy'.format(index))
        return save_path_means, save_path_means_multi, save_path_preds_multi, save_path_distrib, save_path_distrib_multi

    def test(self):
        self.logger.info("computing test mse metric at the end of training...")
        # computing loss on test_dataset:
        for (inp, tar) in self.test_dataset:
            (preds_test, preds_test_resampl), _, _ = self.smc_transformer(inputs=inp,
                                                                          targets=tar)  # predictions test are the ones not resampled.
            test_metric_avg_pred = tf.keras.losses.MSE(tar,
                                                       tf.reduce_mean(preds_test, axis=1, keepdims=True))  # (B,1,S)
            test_metric_avg_pred = tf.reduce_mean(test_metric_avg_pred)

        self.logger.info("test mse metric from avg particle: {}".format(test_metric_avg_pred))
