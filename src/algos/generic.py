from src.utils.utils_train import create_logger
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from src.utils.utils_train import write_to_csv, create_config_file
import json
import math
from src.eval.language_metrics import gpt2_perplexity_batch, BLEU_score, SELFBLEU_score


class Algo:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.bs = args.bs
        self.EPOCHS = args.ep
        self.output_path = args.output_path
        self.save_path = args.save_path
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        self.out_folder = args.output_path
        self.start_epoch = 0
        self.mc_samples = args.mc_samples
        self.past_len = args.past_len
        self.test_predictive_distribution = None
        self.test_predictive_distribution_multistep = None
        self.distribution = False

    def create_logger(self):
        out_file_log = os.path.join(self.out_folder, 'training_log.log')
        logger = create_logger(out_file_log=out_file_log)
        return logger

    def create_ckpt_path(self, args):
        if args.save_path is not None:
            checkpoint_path = os.path.join(args.save_path, "checkpoints")
        else:
            checkpoint_path = os.path.join(self.out_folder, "checkpoints")
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        return checkpoint_path

    def save_hparams(self, args):
        dict_hparams = vars(args)
        dict_hparams = {key: str(value) for key, value in dict_hparams.items()}
        config_path = os.path.join(self.out_folder, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(dict_hparams, fp, sort_keys=True, indent=4)
        # create_config_file(os.path.join(self.out_folder, "config.ini"), args)

    def load_datasets(self, num_dim=4, num_dim_targets=None):
        train_data, val_data, test_data = self.dataset.get_datasets()
        #self.logger.info('num samples in training dataset: {}'.format(len(train_data)))
        train_dataset, val_dataset, test_dataset = self.dataset.data_to_dataset(train_data=train_data,
                                                                                val_data=val_data,
                                                                                test_data=test_data,
                                                                                num_dim=num_dim,
                                                                                num_dim_targets=num_dim_targets)
        self.dataset.check_dataset(train_dataset)
        self.dataset.check_dataset(val_dataset)
        self.dataset.check_dataset(test_dataset)
        self.output_size = self.dataset.output_size
        self.logger.info("output size: {}".format(self.output_size))
        self.num_features = 1
        self.seq_len = self.dataset.seq_len
        self.logger.info('number of timeteps: {}'.format(self.seq_len))
        return train_dataset, val_dataset, test_dataset

    def _get_inference_paths(self):
        # create inference folder
        self.inference_path = os.path.join(self.out_folder, "inference_results")
        if not os.path.isdir(self.inference_path):
            os.makedirs(self.inference_path)
        distrib_unistep_path = os.path.join(self.inference_path, 'distrib_unistep.npy')
        distrib_multistep_path = os.path.join(self.inference_path, 'distrib_multistep.npy')
        return distrib_unistep_path, distrib_multistep_path

    def _decode_targets(self, inputs, targets):
        decoded_first_word = self.dataset.tokenizer.decode([tf.squeeze(inputs)[0].numpy()])
        decoded_target = self.dataset.tokenizer.decode(tf.squeeze(targets).numpy())
        decoded_target = decoded_first_word + ' ' + decoded_target
        decoded_future_targets = self.dataset.tokenizer.decode(tf.squeeze(targets)[self.past_len:].numpy())
        if decoded_future_targets != '':
            len_future_targets = len(decoded_future_targets.split(sep=' '))
        else:
            len_future_targets = 0
        return decoded_target, len_future_targets

    def _evaluate_BLEU_score(self, decoded_particles, decoded_target):
        decoded_particles = [particles.split(sep=' ') for particles in decoded_particles]
        decoded_target = decoded_target.split(sep=' ')
        if len(decoded_particles) > 1:
            bleu_scores = []
            for sentence in decoded_particles:
                bleu_score = BLEU_score(true_sentence=decoded_target, generated_sentence=[sentence])
                bleu_scores.append(round(bleu_score, 4))
            selfbleu = round(SELFBLEU_score(sentences=decoded_particles), 4)
            var_bleu = round(np.var(bleu_scores), 4)
            mean_bleu = round(np.mean(bleu_scores), 4)
        else:
            mean_bleu = BLEU_score(true_sentence=decoded_target, generated_sentence=decoded_particles)
            var_bleu = 0.
            selfbleu = None
        return (mean_bleu, var_bleu), selfbleu

    def test(self, **kwargs):
        self.logger.info("--------------------------------------Generating TEXT on test dataset--------------------------------------------")
        metrics_sampling = dict(zip(["mean_bleu", "var_bleu", "gpt2_ppl", "selfbleu"], [[], [], [], []]))
        metrics_greedy = dict(zip(["mean_bleu", "var_bleu", "gpt2_ppl", "selfbleu"], [[], [], [], []]))
        out_file_text_sampling = os.path.join(self.out_folder, "text_sampling.txt")
        out_file_text_greedy = os.path.join(self.out_folder, "text_greedy.txt")
        for (inputs, targets, attention_mask) in self.test_dataset.take(kwargs["test_samples"]):
            if len(inputs.shape) == 4:
                inp, tar = inputs[:, :, :self.past_len, :], targets[:, :, :self.past_len, :]
            elif len(inputs.shape) == 2:
                inp, tar = inputs[:, :self.past_len], targets[:,:self.past_len]
            decoded_targets, len_future_targets = self._decode_targets(inputs, targets)
            future_len = max(self.future_len, len_future_targets)
            self.logger.info("-"*30 + "SAMPLING GENERATION" + '-'*30)
            metrics_sampling = self._generate_text(inputs=inp, targets=tar, attention_mask=attention_mask, decoded_targets=decoded_targets, future_len=future_len, metrics=metrics_sampling, out_file_text=out_file_text_sampling, decoding="sampling")
            self.logger.info("-" * 30 + "GREEDY GENERATION" + '-' * 30)
            metrics_greedy = self._generate_text(inputs=inp, targets=tar, attention_mask=attention_mask, decoded_targets=decoded_targets, future_len=future_len, metrics=metrics_greedy, out_file_text=out_file_text_greedy, decoding="greedy")
        self._save_and_log_metrics(metrics_sampling, decoding='sampling')
        self._save_and_log_metrics(metrics_greedy, decoding="greedy")

    def _generate_text(self, inputs, targets, attention_mask, decoded_targets, future_len, metrics, out_file_text, decoding="sampling"):
        # #particles, dict_top_words, particles_norm = self.inference_multistep(inputs=inputs,
        #                                                                      targets=targets, attention_mask=attention_mask, past_len=self.past_len,
        #                                                                      future_len=future_len, decoding=decoding) # shape (1,P,len,1) #TODO: put a min between self.future_len and len_decoded target.
        particles, dict_top_words, particles_norm = self.inference_multistep_with_resampling(inputs=inputs,
                                                                             targets=targets,
                                                                             attention_mask=attention_mask,
                                                                             past_len=self.past_len,
                                                                             future_len=future_len, decoding=decoding)
        decoded_particles = [self.dataset.tokenizer.decode(tf.squeeze(particles)[p].numpy()) for p in
                             range(particles.shape[1])] if self.distribution else [
            self.dataset.tokenizer.decode(tf.squeeze(particles).numpy())]
        gpt2_ppl = gpt2_perplexity_batch(decoded_particles)
        (mean_bleu, var_bleu), selfbleu = self._evaluate_BLEU_score(decoded_particles=decoded_particles,
                                                                    decoded_target=decoded_targets)
        for key, val in zip(list(metrics.keys()), [mean_bleu, var_bleu, gpt2_ppl, selfbleu]):
            if val is not None:
                metrics[key].append(val)
        self.logger.info("INPUT SENTENCE:{}".format(self.dataset.tokenizer.decode(tf.squeeze(inputs).numpy())))
        self.logger.info("DECODED TEXT SEQUENCES: {}".format('\n'.join(decoded_particles)))
        with open(out_file_text, 'a') as f:
            f.write('\n'.join(decoded_particles))
            f.write('\n'+'-'*60+'\n')
        if dict_top_words is not None:
            self._log(dict_top_words, string="TOP K WORDS")
        if particles_norm is not None:
            self._log(particles_norm, string="PARTICLES NORM")
        self.logger.info("-------------------------------------------------------------------")
        return metrics

    def _save_and_log_metrics(self, metrics, decoding="sampling"):
        self.logger.info("--------------------------------------------{} GENERATION---------------------------------------------------".format(decoding))
        mean_metrics = dict(zip(list(metrics.keys()), [np.mean(val) for val in list(metrics.values())]))
        metrics_file = os.path.join(self.out_folder, "test_metrics_all_{}.csv".format(decoding))
        write_to_csv(metrics_file, metrics)
        mean_metrics_file = os.path.join(self.out_folder, "test_metrics_mean_{}.csv".format(decoding))
        write_to_csv(mean_metrics_file, mean_metrics)
        self.logger.info(
            "------------------------------------------------MEAN SCORES------------------------------------------------------------------------")
        self.logger.info(mean_metrics)
        self.logger.info(
            "------------------------------------------------ALL SAMPLES SCORES------------------------------------------------------------------------")
        self.logger.info(metrics)
        self.logger.info(
            "---------------------------------------------------------------------------------------------------------------------------------------------------------")


    def _log(self, list_elements, string=''):
        for i, elem in enumerate(list_elements):
            self.logger.info(string + '- timestep {}'.format(i))
            for key, val in enumerate(elem):
                self.logger.info("P{}:{}".format(key, val))
            self.logger.info('-' * 30)