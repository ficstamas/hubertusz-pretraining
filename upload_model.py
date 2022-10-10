from transformers import AutoModelForPreTraining, AutoTokenizer, TFAutoModelForPreTraining, FlaxAutoModelForPreTraining


path_to_cp = "datasets/hubert-small-plus-100k-checkpoint-last/"
repo_name = "hubert-small-wiki"
repo_url = f"https://huggingface.co/SzegedAI/{repo_name}"

model = AutoModelForPreTraining.from_pretrained(path_to_cp)
tf_model = TFAutoModelForPreTraining.from_pretrained(path_to_cp, from_pt=True)
# jax_model = FlaxAutoModelForPreTraining.from_pretrained(path_to_cp, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(path_to_cp)

model.push_to_hub(repo_path_or_name=repo_name, repo_url=repo_url)
tf_model.push_to_hub(repo_path_or_name=repo_name, repo_url=repo_url)
# jax_model.push_to_hub(repo_path_or_name=repo_name, repo_url=repo_url)
tokenizer.push_to_hub(repo_path_or_name=repo_name, repo_url=repo_url)
# hubert-tiny-500k-checkpoint-500000