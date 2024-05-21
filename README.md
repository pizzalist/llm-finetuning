# LLM Fine-tuning Project

이 프로젝트는 Hugging Face의 대형 언어 모델(LLM)을 불러와서 파인튜닝하고, 결과를 HuggingFace에 merge & push한 후, 모델을 사용하여 inference를 수행하는 방법을 다룹니다.

## 프로젝트 구조

본 프로젝트는 다음 파일들을 포함하고 있습니다:

- `train_llm.py`: 불러온 LLM을 특정 데이터셋에 맞추어 파인튜닝하는 스크립트입니다. [파일 보기](./train_llm.py)
- `merge_push.py`: 학습된 모델 merge & Hugging Face 업로드 [파일 보기](./merge_push.py)
- `inference.py`: Hugging Face에서 불러온 모델을 inference를 실행하는 스크립트입니다. [파일 보기](./inference.py)


Result Models
- [letgoofthepizza Llama-3-8B-Instruct-news-summary](https://huggingface.co/letgoofthepizza/Llama-3-8B-Instruct-news-summary)
- [letgoofthepizza gemma-7b-it-finetuned-open-korean-instructions](https://huggingface.co/letgoofthepizza/gemma-7b-it-finetuned-open-korean-instructions)
- [letgoofthepizza Llama-2-7b-chat-hf-finetuned-open-korean-instructions](https://huggingface.co/letgoofthepizza/Llama-2-7b-chat-hf-finetuned-open-korean-instructions)
- [letgoofthepizza Mistral-7B-v0.1-finetuned-open-korean-instructions ](https://huggingface.co/letgoofthepizza/Mistral-7B-v0.1-finetuned-open-korean-instructions)

## 업데이트
### 24.05.21
- llama3 news 요약 task fine tuning 업데이트 (진행중)
- `langchain_hf.py`: Hugging Face x LangChain package: HuggingFacePipeline   [파일 보기](./langchain_hf.py)