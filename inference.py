from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from huggingface_hub import notebook_login

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
ft_model_name = "letgoofthepizza/Llama-3-8B-Instruct-news-summary"

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


# prompt1 = """ 
# 최근 미국 소비자물가지수(CPI)가 둔화 조짐을 보이며 뉴욕 증시의 투자 심리를 살렸다. 물가가 잡히면 금리 인하 명분이 살아나기 때문이다. 그러나 국내 증시는 글로벌 대비 언더퍼폼(underperform·수익률 저조)하며 다소 답답한 흐름을 이어가고 있다. 전체적으로 오르지 않고 개별 업종·종목의 순환매 장세가 이어지는 분위기다. 환기가 필요한 시점이다. 조준기 SK증권 연구원은 “현재 국내 증시는 호재가 나타나면 그 부분을 어느 정도 반영하기는 하나 강도가 그리 강하진 않다”며 “코스피 지수가 한 단계 더 도약하려면 추가적인 호재가 등장해야 한다”고 말했다.

# 향후 상당 기간 증시 향방을 결정할 이벤트로는 어떤 게 있을까. 전문가들은 한국 시각으로 이틀 후인 23일(목요일) 새벽에 발표될 미국 반도체 기업 엔비디아의 실적을 꼽는다. 임정은 KB증권 연구원은 “인공지능(AI) 반도체 붐의 핵심인 엔비디아는 최근 증시 상승세의 강력한 동력 역할을 해왔고, 실적 기대치 또한 높아진 상황”이라고 했다.

# 증권가는 엔비디아의 지난 회계연도(2023년 5월~2024년 4월) 매출을 246억달러(약 34조원)으로 보고 있다. 1년 전보다 242% 늘어난 수치다. 순익 컨센서스(전망치 평균)도 128억3000만달러(약 17조원)로 전년 대비 6배가량 높다. 엔비디아는 최근 5개 분기 실적 발표에서 매출과 주당순이익(EPS) 모두 서프라이즈를 기록하며 기대에 부응했
# 물론 최근 2개 분기 실적 발표에서는 실제 실적과 가이던스 모두 예상치를 웃돌았는데도 차익 실현 매물이 출회한 바 있다. 이번에도 호실적을 발표했는데, 주가는 하락하는 것이 아닐까. 실제 미국에서는 엔비디아가 예상치를 웃도는 실적을 발표하겠지만, 주가는 하락할 수도 있다면서 차라리 엔비디아로 인해 호황을 맞고 있는 에너지주에 투자하는 게 낫다는 투자 의견이 나오기도 했다.

# 그럼에도 전문가들은 일단 실적이 눈높이를 충족하는 것이 중요하다고 강조한다. 조준기 연구원은 “일시 조정 이후 다시 달렸던 경험이 있기에 일단 좋은 실적과 가이던스는 증시 추세 강화의 필수 요건이라고 생각한다”고 했다.
# ## 다음 뉴스 기사를 한글로 요약해줘.
# """

# prompt2 = """ 
# 최근 미국 소비자물가지수(CPI)가 둔화 조짐을 보이며 뉴욕 증시의 투자 심리를 살렸다. 물가가 잡히면 금리 인하 명분이 살아나기 때문이다. 그러나 국내 증시는 글로벌 대비 언더퍼폼(underperform·수익률 저조)하며 다소 답답한 흐름을 이어가고 있다. 전체적으로 오르지 않고 개별 업종·종목의 순환매 장세가 이어지는 분위기다. 환기가 필요한 시점이다. 조준기 SK증권 연구원은 “현재 국내 증시는 호재가 나타나면 그 부분을 어느 정도 반영하기는 하나 강도가 그리 강하진 않다”며 “코스피 지수가 한 단계 더 도약하려면 추가적인 호재가 등장해야 한다”고 말했다.

# 향후 상당 기간 증시 향방을 결정할 이벤트로는 어떤 게 있을까. 전문가들은 한국 시각으로 이틀 후인 23일(목요일) 새벽에 발표될 미국 반도체 기업 엔비디아의 실적을 꼽는다. 임정은 KB증권 연구원은 “인공지능(AI) 반도체 붐의 핵심인 엔비디아는 최근 증시 상승세의 강력한 동력 역할을 해왔고, 실적 기대치 또한 높아진 상황”이라고 했다.

# 증권가는 엔비디아의 지난 회계연도(2023년 5월~2024년 4월) 매출을 246억달러(약 34조원)으로 보고 있다. 1년 전보다 242% 늘어난 수치다. 순익 컨센서스(전망치 평균)도 128억3000만달러(약 17조원)로 전년 대비 6배가량 높다. 엔비디아는 최근 5개 분기 실적 발표에서 매출과 주당순이익(EPS) 모두 서프라이즈를 기록하며 기대에 부응했
# 물론 최근 2개 분기 실적 발표에서는 실제 실적과 가이던스 모두 예상치를 웃돌았는데도 차익 실현 매물이 출회한 바 있다. 이번에도 호실적을 발표했는데, 주가는 하락하는 것이 아닐까. 실제 미국에서는 엔비디아가 예상치를 웃도는 실적을 발표하겠지만, 주가는 하락할 수도 있다면서 차라리 엔비디아로 인해 호황을 맞고 있는 에너지주에 투자하는 게 낫다는 투자 의견이 나오기도 했다.

# 그럼에도 전문가들은 일단 실적이 눈높이를 충족하는 것이 중요하다고 강조한다. 조준기 연구원은 “일시 조정 이후 다시 달렸던 경험이 있기에 일단 좋은 실적과 가이던스는 증시 추세 강화의 필수 요건이라고 생각한다”고 했다.
# ## 다음 뉴스 기사를 한글로 요약해줘.
# """
prompt1 = """ 
경제 인플레이션에 대해 설명해줘.
"""

prompt2 = """ 
경제 인플레이션에 대해 설명해줘.
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

