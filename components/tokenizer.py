from transformers import AutoTokenizer


def load_tokenizer(name: str, max_length: int, truncation: bool):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # next_sentence_label

    def tokenizer_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], padding='max_length', max_length=max_length,
                         truncation=truncation, return_special_tokens_mask=True)

    return tokenizer, tokenizer_function
