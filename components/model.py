from transformers import AutoConfig, BertForPreTraining


def make_model(config_name: str, vocab_size: int):
    config = AutoConfig.from_pretrained(config_name, vocab_size=vocab_size)
    model = BertForPreTraining(config=config)
    return model
