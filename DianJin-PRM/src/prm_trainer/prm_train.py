import argparse
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
from model.fin_prm import Qwen3ForProcessRewardModel
from model.fin_config import Qwen3PRMConfig

class DataCollatorForProcessReward:
    def __init__(self, tokenizer, max_length=8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        texts, labels_list = [], []
        for f in features:
            trace = f["trace"].replace('\n\n', '<extra_0>') + '<extra_0>'
            labels = f["step_labels"]
            labels_list.append(labels)
            text = f"##Question\n{f['question']}\n{f['choices']}\n\n##Thinking Trajectory\n{trace}\n\n##Final Answer\n{f['final_answer']}<extra_1>"
            texts.append(text)

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        step_sep_id_0 = self.tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
        step_sep_id_1 = self.tokenizer.encode("<extra_1>", add_special_tokens=False)[0]
        input_ids = batch["input_ids"]
        
        label_tensors = []
        for labels_list, input_id in zip(labels_list, input_ids):
            label_tensor = torch.full_like(input_id, -100)
            mask = (input_id == step_sep_id_0) | (input_id == step_sep_id_1)
            label_positions = mask.nonzero(as_tuple=True)[0]
            for pos, lab in zip(label_positions, labels_list):
                label_tensor[pos] = lab
            if input_id[-1] == step_sep_id_1: 
                label_tensor[-1] = f["trajectory_label"]
            label_tensors.append(label_tensor)
        batch["labels"] = torch.stack(label_tensors)
        return batch
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="DianJin-PRM/src/model/config.json")
    parser.add_argument('--pretrained_model_path', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--data_path', type=str, default='DianJin-PRM/src/data/train.json')
    parser.add_argument('--output_path', type=str, default="DianJin-PRM/src/checkpoint")
    parser.add_argument('--max_length', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--deepspeed_config', type=str, default='DianJin-PRM/src/train/deepspeed_config.json')
    parser.add_argument('--local_rank', type=int, default=0, help='deepspeed distributed launch will pass this argument')
    args = parser.parse_args()

    config = Qwen3PRMConfig.from_pretrained(args.config_path)
    model = Qwen3ForProcessRewardModel(config=config)
    pretrained_model = AutoModel.from_pretrained(args.pretrained_model_path)
    model.model.load_state_dict(pretrained_model.state_dict(), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    print("Model and tokenizer loaded successfully.")
    
    special_tokens_dict = {'additional_special_tokens': ['<extra_0>', '<extra_1>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("json", data_files=args.data_path, split="train")

    data_collator = DataCollatorForProcessReward(tokenizer, max_length=8192)

    training_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",  
        optim="paged_adamw_8bit",
        report_to="none",
        output_dir=args.output_path,
        do_train=True,
        max_seq_length=args.max_length,
        dataset_text_field='text',
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},  # 跳过预处理数据集
        remove_unused_columns=False,  # 保留未使用的列
        deepspeed=args.deepspeed_config,
        dataloader_pin_memory=False,
        disable_tqdm=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    trainer.train()

if __name__ == "__main__":
    main()
