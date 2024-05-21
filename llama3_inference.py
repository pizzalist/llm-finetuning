import transformers
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "letgoofthepizza/Llama-3-8B-Instruct-news-summary"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
)

messages = [
    {"role": "system", "content": "최근 전국에 불고 있는 ‘깡 열풍’과 함께 농심 새우깡이 화제다.\n 누리꾼 사이에서 ‘깡’ 하면 먼저 떠오르는 새우깡이 ‘1일 1깡’의 패러디 소재로 떠오르고 있다.\n 새우깡이 비의 노래 ‘깡’과 함께 ‘밈(meme)’의 대상이 된 것이다.\n 이런 인기에 힘입어 새우깡은 최근 한 달간 매출이 전년 대비 30% 증가했다.\n 전 세대 사로잡은 ‘생새우’의 풍미새우깡이 ‘깡’ 열풍에 함께할 수 있었던 것은 오래 사랑받아온 ‘국민스낵’이기 때문이다.\n 새우깡의 인기는 출시 49년이 지난 지금도 이어지고 있다.\n 새우깡은 국민스낵·국민안주·국민먹거리로 불리며 모든 세대가 즐겨 먹는 스낵이 됐다.\n 지금도 연간 약 700억원의 매출을 기록하며 스낵시장을 이끌고 있다.\n 새우깡이 국민의 사랑을 받을 수 있었던 이유는 무엇일까.\n 새우깡에는 매력의 비밀 세 가지가 있다.\n 농심은 1971년 국내 첫 스낵 개발에 나서며 맛도 좋고 칼슘도 풍부한 새우를 주재료로 결정했다.\n 한국인이 좋아하는 고소한 새우소금구이 맛을 살리자는 게 제품 개발 콘셉트였다.\n 농심은 실제 생새우를 갈아 넣는 방법을 택했다.\n 새우깡 한 봉지(90g)에는 5~7cm 크기의 생새우 4~5마리가 들어간다.\n 새우깡 특유의 고소한 새우 풍미의 비밀이 바로 이것이다.\n 농심은 이 맛을 지키기 위해 최고 품질의 생새우만 사용한다.\n 또 하나의 비밀은 만드는 방법에 있다.\n 농심은 최적의 맛과 조직감을 살리기 위해 가열된 소금에 굽는 방법으로 새우깡을 만들었다.\n 기름지지 않으면서 적당히 부풀어 올라 특유의 바삭한 조직감을 구현할 수 있었다.\n 1년간 밤새워 연구해 개발제품 개발에 오랜 시간이 걸린 이유도 농심이 이 공법 개발을 위해 수없이 실험을 반복했기 때문이다.\n 수많은 실패를 딛고 완성된 새우깡은 모방제품과 차별점을 둘 수 있었다.\n 새우깡이 출시됐던 1971년 당시에는 지금의 ‘스낵’과 같은 먹거리가 없었다.\n 농심은 어린이부터 노인까지 누구나 먹을 수 있는 스낵을 만들면 성공 가능성이 높다고 판단하고 개발에 나섰다.\n 백지 상태에서 개발을 하다 보니 수많은 시행착오를 겪어야 했다.\n 농심 연구원들은 1년간 밤을 새워가며 연구에 몰두했다.\n 개발에 사용된 밀가루 양만 4.\n 5t 트럭 80여 대분에 이를 정도였다.\n ‘새우깡’이라는 이름도 친근한 이미지를 심어주는 데 큰 역할을 했다.\n 새우깡이라는 브랜드명은 개발 당시 농심 신춘호 사장의 어린 딸이 ‘아리랑’을 ‘아리깡~ 아리깡’이라고 부르는 것에서 힌트를 얻었다.\n "},
    {"role": "user", "content": """
다음 뉴스기사를 한글로 요약 해줘. 
"""
},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
