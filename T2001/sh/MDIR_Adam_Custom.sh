#!/bin/bash

python /opt/ml/image-classification-level1-17/T2001/train.py \
--load_params True \
--exist_ok False \
--epochs 5000 \
--lr 0.005 \
--batch_size 64 \
--valid_batch_size 64 \
--optimizer Adam \
--log_interval 100 \
--model multilabel_dropout_IR \
--name MDIR_Adam_StepLR_Custom \
--augmentation CustomAugmentation