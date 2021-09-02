python ../train.py \
         --dataset MaskSplitByProfileDataset \
         --augmentation get_transforms \
         --resize 224 224 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model ViT \
         --model_version vit_base_patch16_224\
         --criterion focal\
         --log_interval 100 \
         --name ViT_vit_base_patch16_224_0831 \
         --optuna True \
         --optuna_epochs 3
         
