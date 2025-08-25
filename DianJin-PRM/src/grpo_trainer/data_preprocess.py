"""
Preprocess the cflue dataset
"""
import argparse
import os
import json
import datasets
import random
from datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./cflue/knowledge/train.json")
    parser.add_argument("--prompt_template_dir", default='DianJin-PRM/src/templates/rollout_prompt.txt')
    parser.add_argument("--output_dir", default="./output/")
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    with open(args.prompt_template_dir, 'r') as f:
        PROMPT_TEMPLATE = f.read()

    new_data = []
    for item in data:
        # print(item)
        temp_item = {
            'data_source': 'cflue',
            'prompt': [
                {
                    "content": PROMPT_TEMPLATE.format(question=item['question'] + '\n' + item['choices']),
                    "role": "user"
                }
            ],
            'ability': 'MATH',
            'reward_model': {'ground_truth': item['answer'], 'style': 'rule-lighteval/MATH_v2'},
            'extra_info': {}
            }
        new_data.append(temp_item)

    train_dataset = Dataset.from_list(new_data)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            orig_extra_info = example.pop("extra_info")
            extra_info = orig_extra_info.copy()
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": example["reward_model"]["ground_truth"],
                    },
                },
            }
            example["extra_info"] = extra_info
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    output_dir = args.output_dir

    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
