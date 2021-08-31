python train.py \
         --epochs 30 \
         --dataset MaskSplitByProfileDataset \
         --augmentation get_transforms \
         --resize 224 224 \
         --lr 0.00003 \
         --lr_decay_step 50 \
         --lr_gamma 0.7 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model ViT \
         --model_version vit_small_patch16_224\
         --optimizer Adam \
         --criterion focal\
         --log_interval 100\
         --name ViT_vit_small_patch16_224_0831
         
