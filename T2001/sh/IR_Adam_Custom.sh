#!/bin/bash

python /opt/ml/image-classification-level1-17/T2001/train.py \
--load_params True \
--exist_ok False \
--epochs 1000 \
--lr 1e-2 \
--batch_size 32 \
--valid_batch_size 32 \
--optimizer Adam \
--log_interval 200 \
--model InceptionResnet \
--name InceptionResnet \
--cpu True