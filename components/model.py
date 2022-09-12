from transformers import AutoConfig, BertForPreTraining


def make_model(config_name: str):
    config = AutoConfig.from_pretrained(config_name)
    model = BertForPreTraining(config=config)
    return model
