python ../train.py \
         --epochs 10 \
         --dataset MaskSplitByProfileDataset \
         --augmentation get_transforms \
         --resize 224 224 \
         --lr 0.0001 \
         --lr_decay_step 2 \
         --lr_gamma 0.5 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model ViT \
         --model_version vit_base_patch16_224\
         --optimizer Adam \
         --criterion focal\
         --log_interval 100\
         --name T2065/ViT_vit_base_patch16_224_0831
         
