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
         --name ViT \
         --optuna True \
         --optuna_epoch_min 1 \
         --optuna_epoch_max 1 \
         --optuna_lr_min 1e-3 \
         --optuna_lr_max 1e-2 \
         --optuna_ntrials 1
         
