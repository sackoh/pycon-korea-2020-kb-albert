# Inroduction of ALBERT to understand Korean language in Financial domain with Transformers

This tutorial presented at PyCon Korea 2020 was created to share presentation slides and example codes.
Since this tutorial mostly focused on concept of language model and its usages with Hugging Face's Transformers, 
please refer the [link](https://github.com/KB-Bank-AI/KB-ALBERT-KO) if you are interested in pre-trained KB-ALBERT model and its details.
I appreciate Hugging Face Teams and their contributions to open source communities. 

파이콘 코리아 2020에서 발표한 "금융 언어 이해를 위해 개발된 ALBERT 톺아보기 with Transformers"의 
발표자료 및 예제 공유를 위한 깃헙 페이지입니다. 본 발표는 *언어모델의 컨셉*과 *Hugging Face의 Transformers 라이브러리 사용 방법*을 
중점으로 다루고 있습니다. KB-ALBERT 모델 자체 정보에 대해서는 [link](https://github.com/KB-Bank-AI/KB-ALBERT-KO)에서 
찾아 볼 수 있습니다.

<br>

## Slides

Please find Korean slides [here](https://github.com/sackoh/pycon-korea-2020-kb-albert/blob/master/assets/PyConKR-2020_%EC%98%A4%EC%84%B1%EC%9A%B0_%EA%B8%88%EC%9C%B5%20%EC%96%B8%EC%96%B4%20%EC%9D%B4%ED%95%B4%EB%A5%BC%20%EC%9C%84%ED%95%B4%20%EA%B0%9C%EB%B0%9C%EB%90%9C%20ALBERT%20%ED%86%BA%EC%95%84%EB%B3%B4%EA%B8%B0_v1.0.pdf), but english version is being prepared.

발표에 사용된 PPT 자료 [link](https://github.com/sackoh/pycon-korea-2020-kb-albert/blob/master/assets/PyConKR-2020_%EC%98%A4%EC%84%B1%EC%9A%B0_%EA%B8%88%EC%9C%B5%20%EC%96%B8%EC%96%B4%20%EC%9D%B4%ED%95%B4%EB%A5%BC%20%EC%9C%84%ED%95%B4%20%EA%B0%9C%EB%B0%9C%EB%90%9C%20ALBERT%20%ED%86%BA%EC%95%84%EB%B3%B4%EA%B8%B0_v1.0.pdf) 입니다.

## Video Recording

A recording of the tutorial would be published on YouTube.

발표 녹화 촬영 분은 PyCon YouTube에 게시될 예정입니다.

<br>

## Abstract
In Natural Language Processing (NLP), pretraining and fine-tuning have become usual in many tasks. 
Pretraining language models (PLMs) (e.g., BERT, XLNet, RoBERTa, ALBERT, ELECTRA) contain many layers and parameters 
and require large corpus and hardware resources.
So, it is hard for individual users to pretrain LMs with limited resources. 
Google, Facebook and other contributors have shared their works and pretrained models. 
Following these open source trends, In Korea, ETRI, SKT have also opened PLMs (e.g. KorBERT, KoBERT) for both individual 
and organizational users.
However, there are still some restrictions for practitioners to fine-tune at other domains.
This tutorial might be helpful to ones who want to fine-tune and deploy PLMs easily for their domains.

최근 자연어처리 분야는 사전학습한 언어모델들(PLMs) (e.g. BERT, XLNet, ALBERT, ELECTRA)을 활용한 미세조정 등을 하는 것이 
일반적이게 됐습니다. 구글, 페이스북을 비롯하여 국내에서는 ETRI, SKT에서 KorBERT, KoBERT를 공개했습니다.
하지만 실제 이를 자신의 도메인에서 활용하기란 쉽지 않습니다. 최근 가장 인기있는 라이브러리 중 하나인 🤗 Transformers를 
통해 자신의 목적에 맞게 초보자부터 전문적인 딥러닝 연구자까지 누구나 쉽게 블록 형태로 transformer 아키텍처를 불러와 
사용할 수 있게 되었습니다. 이번 발표가 자신의 데이터를 가지고 최신 SOTA PLM을 활용해 NLP 및 Text Mining을 해보기를 
원하는 분들에게 도움이 될 수 있기를 바랍니다. 그리고 약간의 Domain Adaptation과 관련된 내용을 담았습니다.

<br>

## Outline

- Introduction to Language Model for Financial domain
    - Language Model and Pretraining LM
    - ALBERT
    - Domain Adaptation to financial domain
    - transfer learning and some limitations
- 🤗 Transformers
    - Features of Hugging Face's Transformers
    - Reasons why I use Transformers
- Fine-tuning ALBERT with Transformers 

### Korean notebooks

아쉽게도 KB-ALBERT를 완전한 오픈소스로 공개하지 못하여, 별도 신청 과정을 통해야 모델을 다운받아 사용할 수 있습니다.
[깃헙 링크](https://github.com/KB-Bank-AI/KB-ALBERT-KO/tree/master/kb-albert-char)에서 신청방법을 확인해주세요.

빠른 테스트를 위해 BERT multilingual 모델로 아래 notebook 예제들을 테스트해볼 수 있습니다.
**작성된 예제들은 `bert-base-multilingual-cased`로 실행 가능합니다.**

| Notebooks | Description |  |
|:--- | :--- | ---: |
| [Features of 🤗 Transformers](https://github.com/sackoh/pycon-korea-2020-kb-albert/blob/master/01-Features-of-Transformers.ipynb) | `Transformer`의 주요 특징들과 예제 코드 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sackoh/pycon-korea-2020-kb-albert/blob/master/01-Features-of-Transformers.ipynb) |
| [Fine-tuning ALBERT in PyTorch](https://github.com/sackoh/pycon-korea-2020-kb-albert/blob/master/02-Fine-tuning-ALBERT-in-PyTorch.ipynb) | PyTorch로 네이버 영화리뷰 감성분석을 위한 모델을 fine-tuning 하는 예제. GPU 환경 가능 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sackoh/pycon-korea-2020-kb-albert/blob/master/02-Fine-tuning-ALBERT-in-PyTorch.ipynb) |
| [Fine-tuning ALBERT in TensorFlow](https://github.com/sackoh/pycon-korea-2020-kb-albert/blob/master/03-Fine-tuning-ALBERT-in-TensorFlow.ipynb) | TensorFlow로 네이버 영화리뷰 감성분석을 위한 모델을 fine-tuning 하는 예제. GPU 환경 가능 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sackoh/pycon-korea-2020-kb-albert/blob/master/03-Fine-tuning-ALBERT-in-TensorFlow.ipynb) |
| [Fine-tuning ALBERT with TPU](https://github.com/sackoh/pycon-korea-2020-kb-albert/blob/master/04-Fine-tuning-ALBERT-with-TPU.ipynb) | TPU를 통해 훨씬 빠른 속도로 모델을 fine-tuning 해보는 예제 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sackoh/pycon-korea-2020-kb-albert/blob/master/04-Fine-tuning-ALBERT-with-TPU.ipynb) |

<br>

### References
- https://github.com/huggingface/transformers
- https://github.com/KB-Bank-AI/KB-ALBERT-KO
