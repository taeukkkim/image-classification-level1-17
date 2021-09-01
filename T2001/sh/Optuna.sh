#!/bin/bash

python /opt/ml/image-classification-level1-17/T2001/train_with_optuna.py \
--load_params True \
--exist_ok False \
--batch_size 64 \
--valid_batch_size 64 \
--log_interval 200 \
--tensorboard False \
--model multilabel_dropout_IR \
--name OptunaSample \
--augmentation MyAugmentation \
--epochs 100