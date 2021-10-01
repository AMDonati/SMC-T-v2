from datasets import load_dataset
import os
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

if not os.path.isdir("cache/gpt2"):
    os.makedirs("cache/gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")
model.save_pretrained("cache/gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.save_pretrained("cache/gpt2")

print("done loading hugging face models")


for path in ["data/sst/train", "data/sst/val", "data/sst/test", "data/sst/all_data"]:
    if not os.path.isdir(path):
        os.makedirs(path)

dataset = load_dataset('sst')
train_set, val_set, test_set = dataset["train"], dataset["validation"], dataset["test"]
train_set.save_to_disk("cache/sst/train")
val_set.save_to_disk("cache/sst/val")
test_set.save_to_disk("cache/sst/test")
all_dataset = load_dataset("sst", split='train+validation+test')
all_dataset.save_to_disk("cache/sst/all_data")

print("done loading sst dataset")