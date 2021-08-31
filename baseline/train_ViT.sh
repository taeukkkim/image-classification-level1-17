python train.py \
         --epochs 30 \
         --dataset MaskSplitByProfileDataset \
         --augmentation get_transforms \
         --resize 224 224 \
         --lr 0.00005 \
         --lr_decay_step 2 \
         --lr_gamma 0.7 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model ViT \
         --model_version vit_base_patch16_224_in21k\
         --optimizer Adam \
         --criterion focal\
         --log_interval 100\
         --name ViT_vit_base_patch16_224_in21k_0831
         
