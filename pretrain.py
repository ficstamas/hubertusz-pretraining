from components import *
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, IntervalStrategy
from argparse import ArgumentParser
import wandb


def main(args):
    tokenizer, tokenizer_function = load_tokenizer(args.tokenizer, args.seq_length, args.truncation)
    model = make_model(args.config, len(tokenizer))  # prajjwal1/bert-small
    dataset = preprocess_dataset(debug=args.debug)

    tokenized_dataset = dataset.map(tokenizer_function, batched=True, batch_size=16,
                                    remove_columns=["sentence1", "sentence2"], drop_last_batch=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=args.output_dir,
        per_gpu_train_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradiant_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=500,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=25_000,
        evaluation_strategy=IntervalStrategy.NO,
        seed=args.seed,
        data_seed=args.data_seed,
        report_to=["wandb"],
        bf16=args.bf16
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == '__main__':
    argument_parser = ArgumentParser(prog="HuBert Pre-Training")
    argument_parser.add_argument("--output_dir", required=True, type=str)

    argument_parser.add_argument("--tokenizer", default="SZTAKI-HLT/hubert-base-cc", type=str)
    argument_parser.add_argument("--seq_length", default=512, type=int)
    argument_parser.add_argument("--truncation", action="store_true")

    argument_parser.add_argument("--config", default="bert-large-cased", type=str)
    argument_parser.add_argument("--batch_size", default=16, type=int)
    argument_parser.add_argument("--gradiant_accumulation", default=64, type=int)
    argument_parser.add_argument("--learning_rate", default=1e-4, type=float)
    argument_parser.add_argument("--weight_decay", default=0, type=float)
    argument_parser.add_argument("--adam_beta1", default=0.9, type=float)
    argument_parser.add_argument("--adam_beta2", default=0.999, type=float)
    argument_parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    argument_parser.add_argument("--max_grad_norm", default=1.0, type=float)
    argument_parser.add_argument("--max_steps", default=250_000, type=int)
    argument_parser.add_argument("--warmup_steps", default=15_000, type=int)
    argument_parser.add_argument("--seed", default=42, type=int)
    argument_parser.add_argument("--data_seed", default=42, type=int)
    argument_parser.add_argument("--bf16", action="store_true")

    argument_parser.add_argument("--debug", action="store_true")

    arguments = argument_parser.parse_args()

    wandb.init(project="hubert-models", entity="szegedai-semantics")

    main(arguments)
