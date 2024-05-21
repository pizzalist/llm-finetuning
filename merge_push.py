from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from huggingface_hub import notebook_login

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# data_name = 'heegyu/open-korean-instructions'
fine_tuning_model_name = f'{model_name}-finetuned-news-summary'
output_dir = "./test/checkpoint-1320"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right' 

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
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
    save_steps=30, # you can change!
    save_total_limit=5,
    # eval_steps=1,  # 평가를 수행할 스텝 수를 지정
    # evaluation_strategy="steps",  # 검증을 수행할 전략 설정 ("steps", "epoch", "no")
)
trained_model = AutoPeftModelForCausalLM.from_pretrained(
    training_args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map='auto'
)

lora_merged_model = trained_model.merge_and_unload()
lora_merged_model.save_pretrained('merged', safe_serialization=True)
tokenizer.save_pretrained('merged')

lora_merged_model.push_to_hub('letgoofthepizza/Llama-3-8B-Instruct-news-summary')
tokenizer.push_to_hub('letgoofthepizza/Llama-3-8B-Instruct-news-summary')
