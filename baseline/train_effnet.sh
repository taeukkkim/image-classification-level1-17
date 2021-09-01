python train.py \
         --epochs 15 \
         --dataset MaskSplitByProfileDataset \
         --augmentation get_transforms \
         --resize 224 224 \
         --lr 0.00003 \
         --lr_decay_step 50 \
         --lr_gamma 0.7 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model EfficientNet \
         --model_version b3\
         --optimizer Adam \
         --criterion focal\
         --log_interval 100\
         --name T2249/EfficientNet_b3_0901_2
         