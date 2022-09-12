from paddle.datasets.hu.webcorpus import load_dataset
from transformers import AutoTokenizer
import multiprocessing
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
