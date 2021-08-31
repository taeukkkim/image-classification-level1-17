#!/bin/bash

python /opt/ml/image-classification-level1-17/T2001/train.py \
--load_params True \
--exist_ok False \
--epochs 3 \
--lr 1e-3 \
--batch_size 64 \
--valid_batch_size 64 \
--log_interval 20 \
--model multilabel_dropout_IR \
--name MDIR_SGD_Cosine