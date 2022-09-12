from components import *
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


model = make_model("bert-large-cased")
tokenizer, tokenizer_function = load_tokenizer("SZTAKI-HLT/hubert-base-cc", 512, True)
dataset = load_dataset()

tokenized_dataset = dataset.map(tokenizer_function, batched=True, batch_size=16, remove_columns="text",
                                drop_last_batch=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

args = TrainingArguments(output_dir="hubert/large/")

