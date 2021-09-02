python ../train.py \
         --epochs 10 \
         --dataset MaskSplitByProfileDataset \
         --augmentation get_transforms \
         --resize 512 384 \
         --lr 0.0001 \
         --lr_decay_step 2 \
         --lr_gamma 0.5 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model EfficientNet \
         --model_version b3_pruned\
         --optimizer Adam \
         --criterion focal\
         --log_interval 100\
         --name EfficientNet_b3_pruned_0831
         
