from huggingface_hub import HfApi
api = HfApi()
answer_file = f"sc_apps_train.json"

api.upload_file(
    path_or_fileobj=answer_file,
    path_in_repo=answer_file,
    repo_id="BoyuanJackchen/starcoder_answers_with_tokens",
    repo_type="dataset",
)
