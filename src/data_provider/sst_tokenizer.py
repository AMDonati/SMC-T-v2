import torch
import json
import os

class SSTTokenizer:
    def __init__(self, dataset, vocab_path='data/sst/vocab.json', min_token_count=1):
        self.SPECIAL_TOKENS = { #TODO: add EOS and SOS tokens. Add it on input_ids and target_ids.
            '<PAD>': 0,
            '<UNK>': 1,
        }
        self.vocab_path = vocab_path
        if os.path.isfile(vocab_path):
            print("Loading vocab")
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
        else:
            print("Building vocab")
            self.vocab, start_tokens, tokens_to_count = self.build_vocab(dataset, min_token_count=min_token_count)
        self.allow_unk = False if min_token_count == 1 else True
        self.idx_to_token = dict(zip(list(self.vocab.values()), list(self.vocab.keys())))

    def _get_tokens(self, dataset):
        def tokenize_sentence(example):
            example["tokens"] = example["tokens"].split("|")
            return example
        processed_dataset = dataset.map(tokenize_sentence)
        return processed_dataset

    def build_vocab(self, dataset, min_token_count=1):
        dataset = self._get_tokens(dataset)
        token_to_count = {}
        start_tokens = []
        for seq_tokens in dataset["tokens"]:
            start_tokens.append(seq_tokens[0])
            for token in seq_tokens:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1
        # remove "" token
        if "" in list(token_to_count.keys()):
            del token_to_count[""]
        token_to_idx = {}
        for token, idx in self.SPECIAL_TOKENS.items():
            token_to_idx[token] = idx
        for token, count in sorted(token_to_count.items()):
            if count >= min_token_count:
                token_to_idx[token] = len(token_to_idx)
        # getting the unique starting words.
        start_tokens = list(set(start_tokens))
        # saving vocab:
        with open(self.vocab_path, 'w') as f:
            json.dump(token_to_idx, f)
        return token_to_idx, start_tokens, token_to_count

    def encode(self, text, **kwargs):
        code = self.encode_(text, token_to_idx=self.vocab, allow_unk=self.allow_unk)
        if type(code) != torch.tensor and "return_tensors" in kwargs and kwargs["return_tensors"] == "pt":
            code = torch.tensor(code)
        return code

    def decode(self, text, **kwargs):
        decode = self.decode_(text, idx_to_token=self.idx_to_token, delim=' ',
                              ignored=["<SOS>", "<PAD>"])
        if type(decode) != torch.tensor and "return_tensors" in kwargs and kwargs["return_tensors"] == "pt":
            decode = torch.tensor(decode)
        return decode

    def encode_(self, seq_tokens, token_to_idx, allow_unk):
        seq_idx = []
        for token in seq_tokens.split(' '):
            if token not in token_to_idx:
                if allow_unk:
                    token = '<UNK>'
                else:
                    raise KeyError('Token "%s" not in vocab' % token)
            seq_idx.append(token_to_idx[token])
        return seq_idx

    def decode_(self, seq_idx, idx_to_token, delim=' ', ignored=["<SOS>", "<PAD>"]):
        tokens = []
        for idx in seq_idx:
            token = idx_to_token[idx]
            if not token in ignored:
                tokens.append(token)
        if delim is None:
            return tokens
        else:
            return delim.join(tokens)


if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("sst", split='train+validation+test')
    print("full vocab tokenizer")
    sst_tokenizer = SSTTokenizer(dataset=dataset)
    print(len(sst_tokenizer.vocab))
    token_ids = sst_tokenizer.encode(dataset["sentence"][0])
    print("-------------------------------------------------------------------------------------------------------")
    print("positive sentiment tokenizer")
    sst_tokenizer = SSTTokenizer(dataset=dataset, label=1, vocab_path="../../data/sst/vocab_label1.json")
    print("negative sentiment tokenizer")
    sst_tokenizer = SSTTokenizer(dataset=dataset, label=0, vocab_path="../../data/sst/vocab_label0.json")
    print("done")