import os
import random
import json

import re

import torch
import sys


class TextPromptDataset:
    def __init__(self, path, num=None, is_txt=False):
        super().__init__()

        self.prompts = []
        if not is_txt:
            jsonl_file_path = path
            prompt_path = re.sub(r'(train|valid)_anno', r'\1_prompt', path)
            if not os.path.exists(prompt_path):
                with open(jsonl_file_path, 'r') as jsonl_file:
                    for line in jsonl_file:
                        json_object = json.loads(line)
                        key = list(json_object['Task2'].keys())[0]
                        assert key in ['Caption', 'Caption:', 'caption']
                        self.prompts += [json_object['Task2'][key]]
                self.prompts = list(set(filter(None, self.prompts)))
                self.prompts.sort()
                random.seed(2023)
                random.shuffle(self.prompts)
                with open(prompt_path, 'w') as jsonl_file:
                    json.dump(self.prompts, jsonl_file)
            else:
                with open(prompt_path, 'r') as jsonl_file:
                    self.prompts = json.load(jsonl_file)
        else:
            if os.path.exists(path):
                with open(path, 'rt') as txt_file:
                    for row in txt_file:
                        
                        row = row.strip('\n')
                        
                        self.prompts.append(row)
            else:
                raise FileNotFoundError(f"TXT file {path} does not exist！")


    def shuffle(self, *args, **kwargs):
        random.shuffle(self.prompts)
        return self

    def select(self, selected_range):
        self.prompts = [self.prompts[idx] for idx in selected_range]
        return self

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        with torch.no_grad():
            text_inputs = self.tokenizer(
                self.prompts[index],
                padding="max_length",
                max_length=120,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_attention_mask = text_inputs.attention_mask
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.text_encoder.device),
                attention_mask=prompt_attention_mask.to(self.text_encoder.device)
            )[0]
        data = {"prompt_embeds": prompt_embeds.squeeze(0),
                "prompt_attention_mask": prompt_attention_mask.squeeze(0)}
        return data