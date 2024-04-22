# !huggingface-cli login --token [...read_token...] # your code

import os
cache_dir = '/home/noah/workspace/dl-study/nlp_study/llama2/cache'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    
os.environ['HF_HOME'] = cache_dir

from datasets import load_dataset

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments 
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from huggingface_hub import notebook_login

import wandb

from typing import List, Union

model_name = 'google/gemma-7b'
data_name = 'heegyu/open-korean-instructions'
fine_tuning_model_name = f'{model_name}-finetuned-open-korean-instructions'
base_model = AutoModelForCausalLM.from_pretrained(model_name) # 모델이 출력을 캐시할지 여부)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Check special token
bos = tokenizer.bos_token_id # 문장 시작 토큰
eos = tokenizer.eos_token_id # 문장 끝 토큰
pad = tokenizer.pad_token_id # 문장 패딩 토큰
tokenizer.padding_side = "right" # 패딩 오른쪽

# LoRA의 하이퍼파라미터를 설정 
# 알파값을 16으로 설정하여 스케일링
# r은 64로 설정
# 입력 임베딩 사이즈 64랭크까지 압축
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias='none',
    task_type='CAUSAL_LM'
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
dataset = load_dataset(data_name)
print(dataset['train'])

if (pad == None) or (pad == eos):
    tokenizer.pad_token_id = 0  # 만약 패딩값이 없거나 eos값과 같다면,
print("length of tokenizer:",len(tokenizer)) # 32000

# 5-2. Instruction tuning을 위한 template 작성.

instruct_template = {
    "prompt_input": "아래는 작업을 설명하는 지침과 추가 입력을 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 답변을 작성해주세요.\n\n### 지침:\n{instruction}\n\n### 입력:\n{input}\n\n### 답변:\n",
    "prompt_no_input" : "아래는 작업을 설명하는 지침입니다. 요청을 적절히 완료하는 답변을 작성해주세요.\n\n### 지침:\n{instruction}\n\n### 답변:\n",
    "response_split": "### 답변:"
}

###########################################
# 5-3. 데이터셋 불러오는 클래스

class Prompter(object):

    def __init__(self, verbose: bool = False):
        self.template = instruct_template

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:

        if input: # input text가 있다면
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )

        if label:
            res = f"{res}{label}"

        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

prompter = Prompter()

###########################################
cutoff_len = 4096
# Tokenizer에서 나오는 input값 설정 옵션
train_on_inputs = False
add_eos_token = False

# 5-4. Token generation 함수
def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):

        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"])

    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:

        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"])

        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token)

        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
    return tokenized_full_prompt

val_data = None
train_data = dataset["train"].shuffle() # random
train_data = train_data.map(generate_and_tokenize_prompt)

print(train_data)

base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  quantization_config=bnb_config, # 양자화 설정
                                                  use_cache=False) # 모델이 출력을 캐시할지 여부)
base_model.config.pretraining_tp = 1
base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)
peft_model = get_peft_model(base_model, peft_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model.to(device)

output_dir= "./test"

training_args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_ratio=0.06,
        num_train_epochs=1,
        learning_rate=4e-3,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="no",
        save_strategy="steps",
        max_grad_norm = 1.0,
        save_steps = 50, # you can change!
        lr_scheduler_type='cosine',
        output_dir=output_dir,
        save_total_limit=10,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        group_by_length = False
)

trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

###########################################
# 7-1. 만약 이전에 돌렸던 모델을 가져온다면, 아래의 코드 실행

# resume_from_checkpoint = './test/checkpoint-270'
resume_from_checkpoint = False

if resume_from_checkpoint:
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # All checkpoint

    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model
        resume_from_checkpoint = (
            True
        ) # kyujin: I will use this checkpoint

    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)

    else:
        print(f"Checkpoint {checkpoint_name} not found")
        
torch.cuda.empty_cache()
trainer.train(resume_from_checkpoint=resume_from_checkpoint)


model.save_pretrained(output_dir)
model_path = os.path.join(output_dir, "pytorch_model.bin")
torch.save({}, model_path)
tokenizer.save_pretrained(output_dir)
wandb.finish()

## 하이퍼 파라미터 뭔지 알고 
## 조정해보기