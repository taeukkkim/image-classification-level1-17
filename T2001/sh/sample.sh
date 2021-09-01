#!/bin/bash

python /opt/ml/image-classification-level1-17/T2001/train.py \
--load_params True \
--exist_ok False \
--epochs 3 \
--lr 1e-4 \
--batch_size 32 \
--valid_batch_size 32 \
--optimizer Adam \
--log_interval 10 \
--model MyModelBaseIRV2 \
--name MDIR_Adam_aasad