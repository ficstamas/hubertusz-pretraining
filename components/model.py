from transformers import AutoConfig, BertForPreTraining


def make_model(config_name: str, vocab_size: int, resume: bool):
    if resume:
        return BertForPreTraining.from_pretrained(config_name)
    config = AutoConfig.from_pretrained(config_name, vocab_size=vocab_size)
    model = BertForPreTraining(config=config)
    return model
