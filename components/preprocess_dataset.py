from paddle.datasets.hu.webcorpus import load_dataset
from transformers import AutoTokenizer
import multiprocessing
from datasets import Dataset, DatasetDict


def load_dataset():
    dataset = load_dataset("../datasets/webcorpus/", regex="wiki*")

    ds = {"text": []}

    for doc in dataset.train:
        for row in doc:
            ds["text"].append(row)

    return DatasetDict({"train": Dataset.from_dict(ds)})
