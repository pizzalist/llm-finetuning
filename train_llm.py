# read
# huggingface-cli login --token [...read_token...] # your code
# wirte
# huggingface-cli login --token [...read_token...]
import pandas as pd
import os

from huggingface_hub import login

cache_dir = '/home/noah/workspace/dl-study/nlp_study/llama2/cache'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    
os.environ['HF_HOME'] = cache_dir

from datasets import load_dataset, Dataset

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


import wandb

from typing import List, Union

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
login(token='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')

# data_name = 'heegyu/open-korean-instructions'
data_path = "/home/noah/workspace/dl-study/nlp_study/llm-finetuning/data/aihub_news_sum_20per_only_text.csv"
data = pd.read_csv(data_path)

fine_tuning_model_name = f'{model_name}-news-summary'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right' 

# LoRA의 하이퍼파라미터를 설정 
# 알파값을 16으로 설정하여 스케일링
# r은 64로 설정
# 입력 임베딩 사이즈 64랭크까지 압축
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias='none',
    task_type='CAUSAL_LM',
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
)

wandb.login()
wandb.init(project=fine_tuning_model_name.split('/')[-1])


# dataset = load_dataset(data_name, split='train[:10%]')
# print(len(dataset))

dataset = Dataset.from_pandas(data)
print(len(dataset))

# eval_dataset = load_dataset(data_name, split='train[10%:12%]')
# print(len(eval_dataset))


base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  quantization_config=bnb_config, # 양자화 설정
                                                  use_cache=False) # 모델이 출력을 캐시할지 여부)
base_model.config.pretraining_tp = 1
base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)
peft_model = get_peft_model(base_model, peft_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model.to(device)


output_dir = "./test"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim='paged_adamw_32bit',
    logging_steps=1,
    save_strategy='steps',
    learning_rate=2e-4,
    weight_decay=0.01,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type='cosine',
    disable_tqdm=True,
    report_to='wandb',
    seed=42,
    save_steps=30, 
    save_total_limit=5,
)

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=min(tokenizer.model_max_length, 4096),
    tokenizer=tokenizer,
    packing=True,
    args=training_args
)

# ###########################################
# # 7-1. 만약 이전에 돌렸던 모델을 가져온다면, 아래의 코드 실행

# # resume_from_checkpoint = './test/checkpoint-270'
# resume_from_checkpoint = False

# if resume_from_checkpoint:
#     checkpoint_name = os.path.join(
#         resume_from_checkpoint, "pytorch_model.bin"
#     )  # All checkpoint

#     if not os.path.exists(checkpoint_name):
#         checkpoint_name = os.path.join(
#             resume_from_checkpoint, "adapter_model.bin"
#         )  # only LoRA model
#         resume_from_checkpoint = (
#             True
#         ) # kyujin: I will use this checkpoint

#     if os.path.exists(checkpoint_name):
#         print(f"Restarting from {checkpoint_name}")
#         adapters_weights = torch.load(checkpoint_name)
#         set_peft_model_state_dict(model, adapters_weights)

#     else:
#         print(f"Checkpoint {checkpoint_name} not found")
        
torch.cuda.empty_cache()
trainer.train()

peft_model.save_pretrained(output_dir)
model_path = os.path.join(output_dir, "pytorch_model.bin")
torch.save({}, model_path)
tokenizer.save_pretrained(output_dir)
wandb.finish()
