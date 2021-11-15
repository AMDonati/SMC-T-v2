from src.utils.utils_train import CustomSchedule, restoring_checkpoint, write_to_csv
import tensorflow as tf
import os
from src.models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from src.models.Baselines.GPT2Decoder import GPT2Decoder
from src.train.train_functions import train_SMC_transformer
from src.algos.generic import Algo
import json
import datetime
import numpy as np
import math


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
        self.ckpt_path = self.create_ckpt_path(args)
        self.save_hparams(args)
        # if args.num_layers == 0:
        #     self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=2, num_dim_targets=4)
        # else:
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_datasets(num_dim=4)
        self.smc_transformer = SMC_Transformer(d_model=args.d_model,
                                               output_size=self.output_size,
                                               seq_len=self.seq_len,
                                               full_model=args.full_model,
                                               dff=args.dff,
                                               maximum_position_encoding=args.pe,
                                               attn_window=args.attn_w, num_layers=args.num_layers,
                                               num_heads=args.num_heads, reduce_gpt2output=args.reduce_gpt2output)
        self.distribution = args.smc
        self.particles = args.particles
        if args.EM_param is not None:
            self.EM = True
        else:
            self.EM = False
        self.EM_param = args.EM_param
        self._init_SMC_T(args=args)
        self.sigmas_after_training = None
        self.ckpt_manager, _ = self._load_ckpt()
        assert self.past_len < self.seq_len, "past_len should be inferior to the sequence length of the dataset"
        self.future_len = args.future_len if args.future_len is not None else (self.seq_len - self.past_len)

    def _create_out_folder(self, args):
        if args.save_path is not None:
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            out_folder = os.path.join(args.save_path, datetime_folder)
        else:
            out_file = '{}_{}_l{}_h{}_d{}'.format(args.dataset, args.algo, args.num_layers, args.num_heads,
                                                  args.d_model)
            datetime_folder = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            if args.smc:
                out_file = out_file + '__p{}'.format(args.particles)
                out_file = out_file + '_sigmas_{}'.format(args.sigmas)
            out_folder = os.path.join(self.output_path, out_file, datetime_folder)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        return out_folder

    def _init_SMC_T(self, args):
        if args.smc:
            self.logger.info("SMC Transformer for {} particles".format(args.particles))
            if args.sigmas is not None:
                if args.noise_dim == "multi":
                    if not isinstance(args.sigmas, list):
                        sigmas = [args.sigmas] * args.d_model
                    elif len(args.sigmas) == args.d_model:
                        sigmas = args.sigmas
                    else:
                        raise ValueError(
                            "Error in sigmas argument: should be either a scalar, either a list of length d_model args.")
                else:
                    sigmas = args.sigmas
                dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [sigmas for _ in range(4)]))
            else:
                dict_sigmas = None
            self.smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                                         num_particles=args.particles, EM=self.EM)
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
        self.smc_transformer.training = True
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
                              num_train=1,
                              EM_param=self.EM_param)
        if self.distribution:
            self.sigmas_after_training = dict(zip(['k', 'q', 'v', 'z'],
                                                  [self.smc_transformer.cell.attention_smc.logvar_k.numpy(),
                                                   self.smc_transformer.cell.attention_smc.logvar_q.numpy(),
                                                   self.smc_transformer.cell.attention_smc.logvar_v.numpy(),
                                                   self.smc_transformer.cell.attention_smc.logvar_z.numpy()]))
            dict_json = {key: str(value) for key, value in self.sigmas_after_training.items()}
            final_sigmas_path = os.path.join(self.out_folder, "logvar_after_training.json")
            with open(final_sigmas_path, 'w') as fp:
                json.dump(dict_json, fp)  # TODO: add this at each checkpoint saving?
        self.smc_transformer.save_weights(os.path.join(self.out_folder, "model"))
        self.logger.info('-' * 60)

    def _load_ckpt(self, num_train=1):
        # TODO: replace this par model.load_weights()?
        if self.save_path is not None:
            self.smc_transformer.load_weights(os.path.join(self.save_path, "model"))
        ckpt = tf.train.Checkpoint(model=self.smc_transformer,
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
        return ckpt_manager, start_epoch

    def _reinit_sigmas(self):
        if self.logvar_after_training is not None:
            dict_sigmas = {key: math.exp(self.logvar_after_training[key]) for key in ['k', 'q', 'v', 'z']}
            self.smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas,
                                                         num_particles=self.smc_transformer.cell.num_particles)

    def _update_attention_mask(self, attention_mask, last_pred):
        new_padding_mask = tf.where(last_pred == self.dataset.PAD_IDX, x=tf.zeros(last_pred.shape, dtype=tf.int32),
                                    y=tf.ones(last_pred.shape, dtype=tf.int32))
        new_padding_mask = tf.cast(new_padding_mask, dtype=tf.int32)
        attention_mask_ = tf.concat([attention_mask, new_padding_mask], axis=-2)
        return attention_mask_

    def inference_multistep_with_resampling(self, inputs, targets, attention_mask=None, past_len=4, future_len=5,
                                            decoding='sampling'):
        self.smc_transformer.cell.training = False
        P = self.smc_transformer.cell.num_particles
        # forward pass on test_sample_past
        list_top_k_words, list_particles_norm = [], []
        self.smc_transformer.seq_len = past_len
        self.smc_transformer.cell.init_inference_parameters(self.dataset.tokenizer)
        for i in range(future_len + 1):
            if i == 0:
                (preds, _), (K, V, _), _ = self.smc_transformer(inputs, targets,
                                                                attention_mask)  # K,V shape (1, P, 40, D)
                last_pred = preds[:, :, -1, :]
            else:
                encoded_inputs = self.smc_transformer.get_encoded_input(inputs, attention_mask)
                last_pred, (K, V) = self.smc_transformer.cell.call_inference(inputs=inputs,
                                                                             encoded_inputs=encoded_inputs,
                                                                             states=(K, V), timestep=past_len + i - 1)
            if decoding == "sampling":
                dict_top_k_words = self._extract_top_k_words(last_pred)
                list_top_k_words.append(dict_top_k_words)
                particles_norm = self._get_particle_norm(last_pred)
                list_particles_norm.append(particles_norm)
                last_pred = tf.random.categorical(logits=tf.squeeze(last_pred, axis=0), num_samples=1, dtype=tf.int32)
            elif decoding == "greedy":
                last_pred = tf.expand_dims(tf.math.argmax(tf.squeeze(last_pred, axis=0), axis=-1, output_type=tf.int32),
                                           axis=-1)
            if i == 0:
                inputs = tf.tile(inputs, multiples=[1, P, 1, 1])
                targets = tf.tile(targets, multiples=[1, P, 1, 1])
            if i < future_len:  # dummy target (not used when resampling is stopped.)
                self.smc_transformer.seq_len += 1
            last_pred = tf.expand_dims(last_pred, axis=-2)
            last_pred = tf.expand_dims(last_pred, axis=0)
            inputs = tf.concat([inputs, last_pred], axis=-2)
            targets = tf.concat(
                [targets, tf.zeros(shape=(targets.shape[0], targets.shape[1], 1, targets.shape[-1]), dtype=tf.int32)],
                axis=-2)
        if decoding == "sampling":
            return inputs, list_top_k_words, list_particles_norm
        elif decoding == "greedy":
            return inputs, None, None

    def inference_multistep_best_particle(self, inputs, targets, attention_mask=None, past_len=4, future_len=5,
                                          decoding='sampling', num_samples=10):
        self.smc_transformer.training = False
        if self.smc_transformer.cell.noise:
            P = self.smc_transformer.cell.num_particles
        else:
            self.smc_transformer.cell.num_particles = num_samples
            P = num_samples
        # forward pass on test_sample_past
        list_top_k_words, list_particles_norm = [], []
        self.smc_transformer.seq_len = past_len
        if self.smc_transformer.cell.noise:
            self.smc_transformer.cell.add_stop_resampling(past_len)
        for i in range(future_len + 1):
            (preds, _), _, filtering_weights = self.smc_transformer(inputs, targets,
                                                                    attention_mask)  # K,V shape (1, P, 40, D)
            if i == 0:
                indice = tf.random.categorical(filtering_weights[:, :, -1], 1)  # (B,P,1)
                indice = tf.squeeze(indice)
                last_pred = tf.expand_dims(preds[:, indice, -1, :], axis=1)
                last_pred = tf.tile(last_pred, multiples=[1, P, 1])
                inputs = tf.tile(inputs, multiples=[1, P, 1, 1])
                targets = tf.tile(targets, multiples=[1, P, 1, 1])
            else:
                last_pred = preds[:, :, -1, :]
            if decoding == "sampling":
                dict_top_k_words = self._extract_top_k_words(last_pred)
                list_top_k_words.append(dict_top_k_words)
                particles_norm = self._get_particle_norm(last_pred)
                list_particles_norm.append(particles_norm)
                last_pred = tf.random.categorical(logits=tf.squeeze(last_pred, axis=0), num_samples=1, dtype=tf.int32)
            elif decoding == "greedy":
                last_pred = tf.expand_dims(tf.math.argmax(tf.squeeze(last_pred, axis=0), axis=-1, output_type=tf.int32),
                                           axis=-1)
            if i < future_len:  # dummy target (not used when resampling is stopped.)
                self.smc_transformer.seq_len += 1
            last_pred = tf.expand_dims(last_pred, axis=-2)
            last_pred = tf.expand_dims(last_pred, axis=0)
            inputs = tf.concat([inputs, last_pred], axis=-2)
            targets = tf.concat(
                [targets, tf.zeros(shape=(targets.shape[0], targets.shape[1], 1, targets.shape[-1]), dtype=tf.int32)],
                axis=-2)
        if decoding == "sampling":
            return inputs, list_top_k_words, list_particles_norm
        elif decoding == "greedy":
            return inputs, None, None

    def inference_multistep(self, inputs, targets, attention_mask=None, past_len=4, future_len=5, decoding='sampling'):
        self.smc_transformer.training = False
        if not self.smc_transformer.cell.noise:
            self.smc_transformer.cell.num_particles = 10
        P = self.smc_transformer.cell.num_particles
        # forward pass on test_sample_past
        list_top_k_words, list_particles_norm = [], []
        self.smc_transformer.seq_len = inputs.shape[-2]
        # stopping resampling when ground-truth is not available.
        if self.smc_transformer.cell.noise:
            self.smc_transformer.cell.add_stop_resampling(past_len)
        for i in range(future_len + 1):
            (preds, _), _, filtering_weights = self.smc_transformer(inputs, targets,
                                                                    attention_mask)  # K,V shape (1, P, 40, D)
            if i == 0:
                # resampling with last filtering weights:
                indices = tf.random.categorical(filtering_weights[:, :, -1],
                                                self.smc_transformer.cell.num_particles)  # (B,P,1)
            else:
                # uniform sampling:
                categorical_distrib = tf.ones(shape=(1,self.smc_transformer.cell.num_particles),
                                                        dtype=tf.float32) / self.smc_transformer.cell.num_particles
                indices = tf.random.categorical(categorical_distrib,
                                                self.smc_transformer.cell.num_particles)
            last_pred = preds[:, :, -1, :]
            last_pred = tf.gather(last_pred, indices, axis=1, batch_dims=1)  # shape (B,P,V)
            dict_top_k_words = self._extract_top_k_words(last_pred)
            list_top_k_words.append(dict_top_k_words)
            particles_norm = self._get_particle_norm(last_pred)
            list_particles_norm.append(particles_norm)
            if decoding == "sampling":
                last_pred = tf.random.categorical(logits=tf.squeeze(last_pred, axis=0), num_samples=1, dtype=tf.int32)
            elif decoding == "greedy":
                last_pred = tf.expand_dims(tf.math.argmax(tf.squeeze(last_pred, axis=0), axis=-1, output_type=tf.int32),
                                           axis=-1)
            if i == 0:
                inputs = tf.tile(inputs, multiples=[1, P, 1, 1])
                targets = tf.tile(targets, multiples=[1, P, 1, 1])
                if attention_mask is not None:
                    attention_mask = tf.tile(attention_mask, multiples=[1, P, 1, 1])
            if i < future_len:  # dummy target (not used when resampling is stopped.)
                self.smc_transformer.seq_len += 1
            last_pred = tf.expand_dims(last_pred, axis=-2)
            last_pred = tf.expand_dims(last_pred, axis=0)
            inputs = tf.concat([inputs, last_pred], axis=-2)
            targets = tf.concat(
                [targets, tf.zeros(shape=(targets.shape[0], targets.shape[1], 1, targets.shape[-1]), dtype=tf.int32)],
                axis=-2)
            if attention_mask is not None:
                attention_mask = self._update_attention_mask(attention_mask, last_pred)
                #attention_mask = tf.concat([attention_mask, tf.ones(shape=(attention_mask.shape[0], attention_mask.shape[1], 1, 1), dtype=tf.int32)],
                #axis=-2)
        return inputs, list_top_k_words, list_particles_norm


    def _extract_top_k_words(self, last_pred, top_k=10):
        # last_pred -> shape: (B;P,V)
        last_pred = tf.squeeze(last_pred, axis=0)  # shape (P,V)
        probas = tf.nn.softmax(last_pred, axis=-1)
        top_probas, top_tokens = tf.math.top_k(probas, k=top_k)  # shape (B=1,P,top_k)
        top_words = [self.dataset.tokenizer.decode(top_tokens[p].numpy()).split(' ') for p in
                     range(top_tokens.shape[0])]
        top_k_words_particles = [dict(zip(top_words[p], list(np.round(top_probas[p].numpy(), 5)))) for p in
                                 range(top_probas.shape[0])]
        return top_k_words_particles

    def _get_particle_norm(self, last_pred):
        last_pred = tf.squeeze(last_pred, axis=0)
        logits_norm = tf.norm(last_pred, axis=-1)  # shape (B)
        return list(np.round(logits_norm.numpy(), 4))
