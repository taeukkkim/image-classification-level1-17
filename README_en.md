# pstage_01_image_classification

Code for solution in mask, gender, age classification of boostcamp Aistages

# Getting Started    
## Dependencies
- torch==1.6.0
- torchvision==0.7.0
- tensorboard==2.4.1
- pandas==1.1.5
- opencv-python==4.5.1.48
- scikit-learn~=0.24.1
- matplotlib==3.2.1
- timm==0.4.12
- albumentations==1.0.3
- pandas-streaming==0.2.175
- facenet-pytorch==2.5.2

## Installation
```
git clone https://github.com/pudae/kaggle-understanding-clouds.git
pip install -r requirements.txt
```

# Dataset
This dataset consist of face images and labels. label is class that combinates mask, gender and age. Unfortunately Aistages said that the dataset in this competition can not be made public. 

## Class Description
Class description is like below.
|Class|Mask|Gender|Age|
|------|---|---|---|
|0|Wear|Male|<30|
|1|Wear|Male|>=30 and <60|
|2|Wear|Male|>=60|
|3|Wear|Female|<30|
|4|Wear|Female|>=30 and <60|
|5|Wear|Female|>=60|
|6|Incorrect|Male|<30|
|7|Incorrect|Male|>=30 and <60|
|8|Incorrect|Male|>=60|
|9|Incorrect|Female|<30|
|10|Incorrect|Female|>=30 and <60|
|11|Incorrect|Female|>=60|
|12|Not Wear|Male|<30|
|13|Not Wear|Male|>=30 and <60|
|14|Not Wear|Male|>=60|
|15|Not Wear|Female|<30|
|16|Not Wear|Female|>=30 and <60|
|17|Not Wear|Female|>=60|

## Dataset folder path
```
train/
├─train.csv
└─images/
  ├─(id)_(gender)_(race)_(age)/
  | ├─mask1.jpg
  | ├─mask2.jpg
  | ├─mask3.jpg
  | ├─mask4.jpg
  | ├─mask5.jpg
  | ├─incorrect_mask.jpg
  | └─normal.jpg
  ├─000001_Female_Asian_20/
  | ├─mask1.jpg
  | ├─mask2.jpg
  | ├─mask3.jpg
  | ├─mask4.jpg
  | ├─mask5.jpg
  | ├─incorrect_mask.jpg
  | └─normal.jpg
  .
  .
  .
  └─...
eval/
├─train.csv
└─images/
  ├─(id).jpg/
  ├─....jpg/
  .
  .
  .
  └─....jpg/
```

# Code Components
```
├── FaceCrop.ipynb
├── README.md
├── dataset.py
├── evaluation.py
├── inference.py
├── loss.py
├── model
├── model.py
├── requirements.txt
├── sh
│   ├── inference_ViT.sh
│   ├── inference_effnet.sh
│   ├── inference_resnet.sh
│   ├── train_ViT.sh
│   ├── train_ViT_optuna.sh
│   ├── train_effnet.sh
│   ├── train_resnet.sh
│   └── train_resnet_multi.sh
└── train.py
```

# Train
We train models Effici


- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

# Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`
