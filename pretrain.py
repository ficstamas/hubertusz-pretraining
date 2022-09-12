from components import *
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


model = make_model("bert-large-cased")
tokenizer, tokenizer_function = load_tokenizer("SZTAKI-HLT/hubert-base-cc", 512, True)
dataset = preprocess_dataset()

tokenized_dataset = dataset.map(tokenizer_function, batched=True, batch_size=16,
                                remove_columns=["sentence1", "sentence2"], drop_last_batch=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

args = TrainingArguments(output_dir="hubert/large/", use_cuda=False)

trainer = Trainer(
    model=args,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    tokenizer=tokenizer
)

trainer.train()
