from paddle.datasets.hu.webcorpus import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import random


def preprocess_dataset(path="datasets/webcorpus/", rgx_="wiki*", debug=False):
    dataset = load_dataset(path, regex=rgx_)

    ds = {"sentence1": [], "sentence2": [], "next_sentence_label": []}

    for doc in dataset.train:
        first = True
        for row in doc:
            if first:
                ds["sentence1"].append(row)
            else:
                rng = random.randint(0, 1)
                if rng == 1:
                    ds["sentence2"].append(row)
                else:
                    sent1 = ds["sentence1"][-1]
                    del ds["sentence1"][-1]
                    ds["sentence1"].append(row)
                    ds["sentence2"].append(sent1)
                ds["next_sentence_label"].append(rng)

            first = not first
        if len(ds["sentence1"]) != len(ds["sentence2"]):
            del ds["sentence1"][-1]
        if debug:
            break

    if debug:
        return DatasetDict({"train": Dataset.from_dict(Dataset.from_dict(ds)[:1024])})

    return DatasetDict({"train": Dataset.from_dict(ds)})


tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
# next_sentence_label


def tokenizer_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], padding='max_length', max_length=512,
                     truncation=True, return_special_tokens_mask=True)


data = preprocess_dataset(debug=True)

tokenized_dataset = data.map(tokenizer_function, batched=True, batch_size=16, drop_last_batch=True,
                             keep_in_memory=False).save_to_disk("datasets/hubert_huweb-wiki_seq512/")
