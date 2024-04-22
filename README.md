# LLM Fine-tuning Project

이 프로젝트는 Hugging Face의 대형 언어 모델(LLM)을 불러와서 파인튜닝하고, 결과를 HuggingFace에 merge & push한 후, 모델을 사용하여 inference를 수행하는 방법을 다룹니다.

## 프로젝트 구조

본 프로젝트는 다음 파일들을 포함하고 있습니다:

- `train_llama2.py`: 불러온 LLM을 특정 데이터셋에 맞추어 파인튜닝하는 스크립트입니다. [파일 보기](./train_llama2.py)
- `merge_push.py`: 학습된 모델 merge & Hugging Face 업로드 [파일 보기](./merge_push.py)
- `inference.py`: Hugging Face에서 불러온 모델을 inference를 실행하는 스크립트입니다. [파일 보기](./inference.py)
