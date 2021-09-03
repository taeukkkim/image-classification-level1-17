# K-AI image-classification

<center>
<img src=https://img.shields.io/badge/pytorch-1.6.0-%23EE4C2C?logo=pytorch>  <img src=https://img.shields.io/badge/pandas-1.1.5-%23150458?logo=pandas>
</center>

---

# 프로젝트 개요

![https://i.imgur.com/T9RxL0d.png](https://i.imgur.com/T9RxL0d.png)

---

![](https://s3-ap-northeast-2.amazonaws.com/aistages-public-junyeop/app/Users/00000025/files/56bd7d05-4eb8-4e3e-884d-18bd74dc4864..png)

- 코로나 시대에 마스크를 제대로 착용하여 확산을 방지하고자 함
- 사람 얼굴 이미지 만으로 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 모델 구현

---

# How to Use

<center><img src="https://i.imgur.com/UumsqOE.png" alt="My Image"></center>

---

## 필수 설치

- `pip install -r requirements.txt`
- .env.example 파일을 참고하여 .env파일을 만든다.

## dataset

- 데이터 전처리 과정을 담당하는 역할이다.
- Cross-Validation이 구현된 모델이 포함되어 있다.
- 개인이 스스로 transforms을 구성할 수 있다.
- 앙상블을 위해 클래스를 나눈 데이터셋이 구현되어 있다.

## loss

- 다음과 같은 Loss가 구현되어 있다.
  - cross_entropy
  - focal
  - label_smoothing
  - F1Loss
  - FocalLabelSmoothingLoss
  - FocalLabelSmoothingLossWithF1

## Model

- 다음과 같은 모델이 포함되어 있다.
  - ResNet
  - EfficientNet
  - VGG
  - Xception
  - ViT
- 해당 모델을 기반으로 자신의 모델을 구성할 수 있다.
- 필요에 따라 Multi Sample Dropout을 구성할 수 있다.

## Train

- 훈련과정에 필요한 옵션(loss, epochs, optimizer 등)이 포함 되어 있으며, 인자를 넘겨주는 식으로 옵션을 설정할 수 있다.
- Optuna를 사용하여 하이퍼파라미터 최적화를 구성할 수 있다.
- 필요에 따라 cutmix를 사용할 수 있다.
- sh 샘플들을 활용하여 자신만의 sh파일을 만들 수 있다.

## Inference

- 자신의 모델을 테스트 데이터 셋에 적용한다.
- 결과 파일을 출력하며 해당파일을 제출하여 점수를 확인한다.

## Evaluation

- 자신의 모델의 성능을 측정하는 모델이다.
- Train 과정에서 Validation을 실행하지만, 최종모델성능을 확인하는 데 사용하기 좋다.
