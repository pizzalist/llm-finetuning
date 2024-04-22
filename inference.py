from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from huggingface_hub import notebook_login

model_name = 'google/gemma-7b-it'
ft_model_name = 'letgoofthepizza/Mistral-7B-v0.1-finetuned-open-korean-instructions'

tokenizer_base = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer_base.pad_token = tokenizer_base.eos_token
tokenizer_base.padding_side = 'right' 

tokenizer_ft = AutoTokenizer.from_pretrained(ft_model_name, trust_remote_code=True)
tokenizer_ft.pad_token = tokenizer_ft.eos_token
tokenizer_ft.padding_side = 'right' 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  quantization_config=bnb_config, # 양자화 설정
                                                  use_cache=False) # 모델이 출력을 캐시할지 여부)

ft_model = AutoModelForCausalLM.from_pretrained(ft_model_name,
                                                  quantization_config=bnb_config, # 양자화 설정
                                                  use_cache=False) # 모델이 출력을 캐시할지 여부)


prompt1 = """ 
이란의 첫 직접 공격을 받은 이스라엘이 전면전으로 확대되지 않는 선에서 ‘고통스러운 보복’을 준비하고 있다는 현지 언론 보도가 나왔다.
 
이스라엘 채널12 방송은 15일(현지시간) “전시 내각에서 다수의 보복 방식이 논의되고 있다”며 “이 선택지는 모두 역내 전쟁을 촉발하지 않으면서 이란에는 고통스러운 방식”이라고 전했다. 전시내각은 이 가운데 “미국 등 동맹이 반대하지 않는 방식을 선택하려 한다”고 덧붙였다.
앞서 이란은 지난 13일 밤 170기의 드론과 순항미사일 30기, 탄도미사일 120기를 동원해 이스라엘을 공습했다. 이스라엘군은 “이 가운데 99%를 요격했으며 일부 탄도 미사일이 남부 네바팀 공군기지에 떨어졌으나 큰 피해는 없다”고 주장했다.
 
미 ABC 방송은 그러나 “이란이 쏜 탄도미사일 9발이 이스라엘과 미국 등의 방어망을 뚫었다”며 “이 중 5발이 네바팀 기지에 떨어지면서 C-130 수송기와 사용하지 않는 활주로, 빈 창고 등이 파손됐다”고 보도했다.
 
이란 국영 프레스TV도 “이란 혁명수비대가 이스라엘을 보복 공습하면서 극초음속 미사일 여러 발을 발사했고 이는 모두 표적에 명중했다”고 전했다.
 
이 매체는 소식통을 인용해 “이스라엘과 협력국(미국 등 서방, 중동 내 친미 국가)는 이란의 극초음속 미사일을 요격하지 못했다”며 “이란이 극초음속 미사일을 실전에서 사용한 것은 이번이 처음”이라고 밝혔다.
 
혁명수비대는 이번 공습에 드론과 미사일 300여발을 발사했으나 이 가운데 극초음속 미사일이 포함됐는지 등을 구체적으로 공개하지 않았다.
 
앞서 혁명수비대는 지난해 11월 자체 개발한 극초음속 미사일 ‘파타흐-1’의 시험 발사에 성공했다고 발표했다. 당시 발표한 제원에 따르면 파타흐-1은 마하 13∼15의 속도로 날아가 최장 1천400㎞ 거리의 표적을 타격할 수 있다. 고체연료를 사용하며 대기권 밖에서도 궤도를 변경할 수 있으며 스텔스 기능도 탑재한 것으로 전해졌다.

## 다음 뉴스 기사를 3줄로 요약해줘.
"""

prompt2 = """ 
이란의 첫 직접 공격을 받은 이스라엘이 전면전으로 확대되지 않는 선에서 ‘고통스러운 보복’을 준비하고 있다는 현지 언론 보도가 나왔다.
 
이스라엘 채널12 방송은 15일(현지시간) “전시 내각에서 다수의 보복 방식이 논의되고 있다”며 “이 선택지는 모두 역내 전쟁을 촉발하지 않으면서 이란에는 고통스러운 방식”이라고 전했다. 전시내각은 이 가운데 “미국 등 동맹이 반대하지 않는 방식을 선택하려 한다”고 덧붙였다.
앞서 이란은 지난 13일 밤 170기의 드론과 순항미사일 30기, 탄도미사일 120기를 동원해 이스라엘을 공습했다. 이스라엘군은 “이 가운데 99%를 요격했으며 일부 탄도 미사일이 남부 네바팀 공군기지에 떨어졌으나 큰 피해는 없다”고 주장했다.
 
미 ABC 방송은 그러나 “이란이 쏜 탄도미사일 9발이 이스라엘과 미국 등의 방어망을 뚫었다”며 “이 중 5발이 네바팀 기지에 떨어지면서 C-130 수송기와 사용하지 않는 활주로, 빈 창고 등이 파손됐다”고 보도했다.
 
이란 국영 프레스TV도 “이란 혁명수비대가 이스라엘을 보복 공습하면서 극초음속 미사일 여러 발을 발사했고 이는 모두 표적에 명중했다”고 전했다.
 
이 매체는 소식통을 인용해 “이스라엘과 협력국(미국 등 서방, 중동 내 친미 국가)는 이란의 극초음속 미사일을 요격하지 못했다”며 “이란이 극초음속 미사일을 실전에서 사용한 것은 이번이 처음”이라고 밝혔다.
 
혁명수비대는 이번 공습에 드론과 미사일 300여발을 발사했으나 이 가운데 극초음속 미사일이 포함됐는지 등을 구체적으로 공개하지 않았다.
 
앞서 혁명수비대는 지난해 11월 자체 개발한 극초음속 미사일 ‘파타흐-1’의 시험 발사에 성공했다고 발표했다. 당시 발표한 제원에 따르면 파타흐-1은 마하 13∼15의 속도로 날아가 최장 1천400㎞ 거리의 표적을 타격할 수 있다. 고체연료를 사용하며 대기권 밖에서도 궤도를 변경할 수 있으며 스텔스 기능도 탑재한 것으로 전해졌다.

## 다음 뉴스 기사를 3줄로 요약해줘.
"""
input_ids = tokenizer_base(prompt1, return_tensors='pt', truncation=True).input_ids.cuda()
input_ids_ft = tokenizer_ft(prompt2, return_tensors='pt', truncation=True).input_ids.cuda()

print(f"-------------------------\n")
print(f"Prompt:\n{prompt1}\n")
print(f"-------------------------\n")

print(f"Base Model Response :\n")
output_base = base_model.generate(input_ids=input_ids, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
print(f"{tokenizer_base.batch_decode(output_base.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt1):]}")
print(f"-------------------------\n")

print(f"Trained Model Response :\n")
trained_model = ft_model.generate(input_ids=input_ids_ft, max_new_tokens=500, do_sample=True, top_p=0.9,temperature=0.5)
print(f"{tokenizer_ft.batch_decode(trained_model.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt2):]}")
print(f"-------------------------\n")

