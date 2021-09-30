from src.utils.utils_train import CustomSchedule, restoring_checkpoint, write_to_csv
import tensorflow as tf
import os
from src.models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from src.train.train_functions import train_SMC_transformer
from src.eval.inference_functions import inference_multistep
from src.algos.generic import Algo
import json
import datetime
import numpy as np
from src.eval.language_metrics import BLEU_score, SELFBLEU_score


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
                                               maximum_position_encoding=args.pe,
                                               attn_window=args.attn_w, num_layers=args.num_layers, num_heads=args.num_heads)
        self.distribution = args.smc
        self.particles = args.particles
        self._init_SMC_T(args=args)
        self.sigmas_after_training = None
        self.ckpt_manager, _ = self._load_ckpt()
        assert self.past_len < self.seq_len, "past_len should be inferior to the sequence length of the dataset"
        self.future_len = args.future_len if args.future_len is not None else (self.seq_len - self.past_len)

    def _create_out_folder(self, args):
        if args.save_path is not None:
            return args.save_path
        else:
            # out_file = '{}_Recurrent_T_depth_{}_bs_{}_fullmodel_{}_dff_{}_attn_w_{}'.format(args.dataset, args.d_model,
            # self.bs, args.full_model,
            # args.dff, args.attn_w)
            out_file = '{}_l{}_h{}_d{}_{}p_sigmas{}'.format(args.algo, args.num_layers, args.num_heads, args.d_model, args.particles, args.sigmas)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            # if args.smc:
            #     out_file = out_file + '__p_{}'.format(args.particles)
            #     out_file = out_file + '_SigmaObs_{}'.format(args.sigma_obs)
            #     out_file = out_file + '_sigmas_{}'.format(args.sigmas)
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
            self.smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                                         num_particles=args.particles)
            assert self.smc_transformer.cell.noise == self.smc_transformer.cell.attention_smc.noise == True
            self.logger.info("Sigmas init: {}".format(dict_sigmas))

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
            'num layers: {} - num_heads: {} - d_model: {} - batch size {} - full model? {} - dff: {} -attn window: {}'.format(
                self.smc_transformer.num_layers,
                self.smc_transformer.num_heads,
                self.smc_transformer.d_model, self.bs,
                self.smc_transformer.full_model, self.smc_transformer.dff,
                self.smc_transformer.cell.attention_smc.attn_window))
        train_SMC_transformer(smc_transformer=self.smc_transformer,
                              optimizer=self.optimizer,
                              EPOCHS=self.EPOCHS,
                              train_dataset=self.train_dataset,
                              val_dataset=self.val_dataset,
                              output_path=self.out_folder,
                              ckpt_manager=self.ckpt_manager,
                              logger=self.logger,
                              start_epoch=self.start_epoch,
                              num_train=1)
        if self.distribution:
            self.sigmas_after_training = dict(zip(['k', 'q', 'v', 'z'],
                                                   [self.smc_transformer.cell.attention_smc.sigma_k.numpy(),
                                                   self.smc_transformer.cell.attention_smc.sigma_q.numpy(),
                                                   self.smc_transformer.cell.attention_smc.sigma_v.numpy(),
                                                   self.smc_transformer.cell.attention_smc.sigma_z.numpy()]))
            dict_json = {key: str(value) for key, value in self.sigmas_after_training.items()}
            final_sigmas_path = os.path.join(self.out_folder, "sigmas_after_training.json")
            with open(final_sigmas_path, 'w') as fp:
                json.dump(dict_json, fp)  # TODO: add this at each checkpoint saving?
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
        if self.save_path is not None and self.distribution:
            # self._check_consistency_hparams(args)
            sigma_file = "sigmas_after_training.json"
            with open(os.path.join(self.save_path, sigma_file)) as json_file:
                dict_json = json.load(json_file)
            self.sigmas_after_training = {key: float(value) for key, value in dict_json.items()}
            self.logger.info("updating sigmas values with the latest ones...{}".format(dict_json))
            self._reinit_sigmas()
        return ckpt_manager, start_epoch

    def _reinit_sigmas(self):
        if self.sigmas_after_training is not None:
            dict_sigmas = {key: self.sigmas_after_training[key] for key in ['k', 'q', 'v', 'z']}
            self.smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
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
            print('it:', it)
            print("sigma_k: {}, sigma_q: {}, sigma_v: {}, sigma_z: {}".format(
                self.smc_transformer.cell.attention_smc.sigma_k,
                self.smc_transformer.cell.attention_smc.sigma_q,
                self.smc_transformer.cell.attention_smc.sigma_v,
                self.smc_transformer.cell.attention_smc.sigma_z
            ))

        dict_sigmas = dict(zip(['sigma_obs', 'sigma_k', 'sigma_q', 'sigma_v', 'sigma_z'],
                               [self.smc_transformer.cell.Sigma_obs.numpy(),
                                self.smc_transformer.cell.attention_smc.sigma_k.numpy(),
                                self.smc_transformer.cell.attention_smc.sigma_q.numpy(),
                                self.smc_transformer.cell.attention_smc.sigma_v.numpy(),
                                self.smc_transformer.cell.attention_smc.sigma_z.numpy()]))
        write_to_csv(output_dir=os.path.join(self.inference_path, "sigmas_after_EM_{}.csv".format(index)),
                     dic=dict_sigmas)

    def _decode_targets(self, inputs, targets):
        decoded_first_word = self.dataset.tokenizer.decode([tf.squeeze(inputs[:,:,0,:]).numpy()])
        decoded_target = self.dataset.tokenizer.decode(tf.squeeze(targets).numpy())
        decoded_target = decoded_first_word + ' ' + decoded_target
        decoded_future_targets = self.dataset.tokenizer.decode(tf.squeeze(targets[:, :, self.past_len:, :]).numpy())
        if decoded_future_targets != '':
            len_future_targets = len(decoded_future_targets.split(sep=' '))
        else:
            len_future_targets = 0
        return decoded_target, len_future_targets

    def _evaluate_BLEU_score(self, decoded_particles, decoded_target):
        decoded_particles = [particles.split(sep=' ') for particles in decoded_particles]
        decoded_target = decoded_target.split(sep=' ')
        bleu_scores = []
        for sentence in decoded_particles:
            bleu_score = BLEU_score(true_sentence=decoded_target, generated_sentence=[sentence])
            bleu_scores.append(round(bleu_score, 4))
        selfbleu_score = SELFBLEU_score(sentences=decoded_particles)
        max_bleu = np.max(bleu_scores)
        mean_bleu = np.mean(bleu_scores)
        return (max_bleu, mean_bleu, bleu_scores), round(selfbleu_score,4)


    def test(self, **kwargs):
        self.logger.info("--------------------------------------Generating TEXT on test dataset--------------------------------------------")
        # smc_transformer_no_noise = tf.keras.models.clone_model(model=self.smc_transformer)
        # smc_transformer_no_noise.cell.num_particles = 1
        # smc_transformer_no_noise.cell.noise = False
        selfbleu_scores, mean_bleus, max_bleus = [], [], []
        for (inputs, targets) in self.test_dataset.take(kwargs["test_samples"]):
            inp, tar = inputs[:, :, :self.past_len, :], targets[:, :, :self.past_len, :]
            self.logger.info("INPUT SENTENCE:{}".format(self.dataset.tokenizer.decode(tf.squeeze(inp).numpy())))
            decoded_targets, len_future_targets = self._decode_targets(inputs, targets)
            future_len = max(self.future_len, len_future_targets)
            particles = inference_multistep(smc_transformer=self.smc_transformer, inputs=inp,
                                        targets=tar, past_len=self.past_len,
                                        future_len=future_len)  # shape (1,P,len,1) #TODO: put a min between self.future_len and len_decoded target.
            if self.distribution:
                # particles_no_noise = inference_multistep(smc_transformer=smc_transformer_no_noise, inputs=inp,
                #                                      targets=tar, past_len=self.past_len,
                #                                      future_len=self.future_len)  # shape (1,P,len,1)
                decoded_particles = []
                for p in range(particles.shape[1]):
                    decoded_particle = self.dataset.tokenizer.decode(tf.squeeze(particles[:, p, :, :]).numpy())
                    decoded_particles.append(decoded_particle)
                    self.logger.info("DECODED TEXT SEQUENCE - particle{}:{}".format(p, decoded_particle))
                    self.logger.info("-------------------------------------------------------------------")
                # self.logger.info(
                # "-------------------- generating text with NO NOISE TRANSFORMER-----------------------------")
                # self.logger.info("DECODED TEXT SEQUENCE: {}".format(
                # self.dataset.tokenizer.decode(tf.squeeze(particles_no_noise).numpy())))
                (max_bleu, mean_bleu, _), selfbleu_score = self._evaluate_BLEU_score(decoded_particles=decoded_particles, decoded_target=decoded_targets)
                max_bleus.append(max_bleu)
                mean_bleus.append(mean_bleu)
                selfbleu_scores.append(selfbleu_score)
                self.logger.info("--------BLEU SCORES----------:")
                self.logger.info("Max bleu: {}".format(max_bleu))
                #self.logger.info("Mean bleu: {}".format(mean_bleu))
                self.logger.info("--------SELF-BLEU SCORE----------:")
                self.logger.info(selfbleu_score)
            else:
                decoded_particle = self.dataset.tokenizer.decode(tf.squeeze(particles).numpy())
                decoded_particle_ = [decoded_particle.split(sep=' ')]
                decoded_target_ = decoded_targets.split(sep=' ')
                bleu_score = BLEU_score(true_sentence=decoded_target_, generated_sentence=decoded_particle_)
                mean_bleus.append(bleu_score)
                self.logger.info("DECODED TEXT SEQUENCE: {}".format(
                    decoded_particle))
                self.logger.info("BLEU SCORE:{}".format(round(bleu_score, 4)))
            self.logger.info("----------------------------------------------------------------------------------------------------------")
        self.logger.info("------------------------------------------------OVERALL BLEU SCORES------------------------------------------------------------------------")
        self.logger.info("MEAN BLEU:{}".format(np.mean(mean_bleus)))
        self.logger.info("ALL MEAN BLEU:{}".format(mean_bleus))
        if self.distribution:
            self.logger.info("MAX BLEU:{}".format(np.mean(max_bleus)))
            self.logger.info("ALL MAX BLEU:{}".format(max_bleus))
            self.logger.info("SELF BLEU:{}".format(np.mean(selfbleu_scores)))
            self.logger.info("ALL SELF BLEU:{}".format(selfbleu_scores))
        self.logger.info(
            "---------------------------------------------------------------------------------------------------------------------------------------------------------")


    def compute_test_loss(self, save_particles=False):
        test_loss, MEAN_PREDS = [], []
        for (inp, tar) in self.test_dataset:
            (preds_test, preds_test_resampl), _, _ = self.smc_transformer(inputs=inp,
                                                                          targets=tar)  # predictions test are the ones not resampled.
            test_metric_avg_pred = tf.keras.losses.MSE(tar,
                                                       tf.reduce_mean(preds_test, axis=1, keepdims=True))  # (B,1,S)
            #test_metric_avg_pred = tf.keras.losses.MSE(inp,
                                                       #tf.reduce_mean(preds_test, axis=1, keepdims=True))  # (B,1,S)
            test_metric_avg_pred = tf.reduce_mean(test_metric_avg_pred).numpy()
            mean_preds = tf.reduce_mean(preds_test, axis=1)
            test_loss.append(test_metric_avg_pred)
            MEAN_PREDS.append(mean_preds)
        if save_particles:
            np.save(os.path.join(self.out_folder, "particles_preds_test.npy"), preds_test.numpy())
            np.save(os.path.join(self.out_folder, "resampled_particles_preds_test.npy"), preds_test_resampl.numpy())
            print('preds particles shape', preds_test.shape)
            print('preds particles resampled shape', preds_test_resampl.shape)
            self.logger.info("saving predicted particles on test set...")
        MEAN_PREDS = tf.stack(MEAN_PREDS, axis=0)
        return np.mean(test_loss), MEAN_PREDS
