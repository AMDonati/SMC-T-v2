from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np

def get_weights_bleu_score(n_gram=4):
    if n_gram == 2:
        weights = [0.5, 0.5]
    elif n_gram == 3:
        weights = [1 / 3, 1 / 3, 1 / 3]
    elif n_gram == 4:
        weights = [0.25, 0.25, 0.25, 0.25]
    return weights


# class LanguageScore(Metric):
#     '''Compute the perplexity of a pretrained language model (GPT) on the generated dialog.'''
#
#     def __init__(self, agent, train_test, env, trunc, sampling):
#         Metric.__init__(self, agent, train_test, "language_score", "scalar", env, trunc, sampling)
#         self.lm_model = gpt2_model
#         self.tokenizer = gpt2_tokenizer
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.questions = []
#         self.batch_size = 1000
#
#     def fill_(self, **kwargs):
#         pass
#
#     def process_batch(self, questions):
#         loss = torch.nn.CrossEntropyLoss(reduction="none")
#         inputs = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
#         labels = inputs["input_ids"].clone()
#         labels[inputs["attention_mask"] == 0] = -100
#         outputs = self.lm_model(**inputs, labels=labels)
#         shift_logits = outputs["logits"][..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         loss_ = loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(labels.size(0),
#                                                                                                labels.size(1) - 1)
#         masked_loss = loss_.sum(dim=-1)
#         masked_loss /= inputs["attention_mask"].sum(dim=-1)
#         ppls_per_sentence = torch.exp(masked_loss)
#         return ppls_per_sentence.view(-1).tolist()
#
#     def reset(self):
#         self.questions = []
#
#     def compute_(self, **kwargs):
#         if kwargs["state"].text.shape[-1] > 1:
#             state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text[:, 1:].cpu().numpy()[0])
#             self.questions.append(state_decoded)
#         if len(self.questions) == self.batch_size:
#             ppl = self.process_batch(self.questions)
#             self.metric.extend(ppl)
#             self.reset()
#
#     def get_min_ppl_idxs(self, num_diversity):
#         if len(self.questions) > 0:
#             ppl = self.process_batch(self.questions)
#             self.metric_history.extend(ppl)
#             self.reset()
#         ppls = torch.tensor(self.metric_history).view(-1, num_diversity)
#         idx_to_keep = torch.argmin(ppls, dim=1).tolist()
#         return idx_to_keep
#
#     def post_treatment(self, num_episodes, idx_to_keep=None):
#         if len(self.questions) > 0:
#             ppl = self.process_batch(self.questions)
#             self.metric_history.extend(ppl)
#             self.reset()
#         # self.filter_reranking(num_episodes, idx_to_keep)
#         self.post_treatment_()
#         self.save_series_and_stats(idx_to_keep, num_episodes)

def BLEU_score(true_sentence, generated_sentence, split_str=False):
    weights = get_weights_bleu_score(4)
    if split_str:
        true_sentence = true_sentence.split(sep=' ')
        generated_sentence = [generated_sentence.split(sep=' ')]
    score = sentence_bleu(references=generated_sentence, hypothesis=true_sentence, smoothing_function=SmoothingFunction().method2, weights=weights)
    return score

def SELFBLEU_score(sentences):
    scores = []
    for i, sentence in enumerate(sentences):
        ref_sentences = np.delete(sentences, i)
        score = BLEU_score(true_sentence=sentence, generated_sentence=ref_sentences)
        scores.append(score)
    return np.mean(score)


if __name__ == '__main__':
    true_sentence = "My name is Alice"
    generated_sentence = true_sentence
    score1 = BLEU_score(true_sentence, generated_sentence, split_str=True)
    print(score1)
    generated_sentence2 = "My name is Bob"
    score2 = BLEU_score(true_sentence, generated_sentence2, split_str=True)
    print(score2)
    generated_sentence3= "I love my name"
    score3 = BLEU_score(true_sentence, generated_sentence3, split_str=True)
    print(score3)
    generated_sentence4 = "I like it!"
    score4 = BLEU_score(true_sentence, generated_sentence4, split_str=True)
    print(score4)