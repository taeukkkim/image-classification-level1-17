python ../train.py \
         --epochs 15 \
         --dataset MaskSplitStratifyDataset \
         --label mask\
         --augmentation get_transforms \
         --resize 512 384 \
         --lr 0.00003 \
         --lr_decay_step 50 \
         --lr_gamma 0.7 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model Resnet18 \
         --optimizer Adam \
         --criterion focal\
         --log_interval 100\
         --name Resnet18_0901_Stratify_mask
         

python ../train.py \
         --epochs 15 \
         --dataset MaskSplitStratifyDataset \
         --label gender\
         --augmentation get_transforms \
         --resize 512 384 \
         --lr 0.00003 \
         --lr_decay_step 50 \
         --lr_gamma 0.7 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model Resnet18 \
         --optimizer Adam \
         --criterion focal\
         --log_interval 100\
         --name Resnet18_0901_Stratify_gender


python ../train.py \
         --epochs 15 \
         --dataset MaskSplitStratifyDataset \
         --label age\
         --augmentation get_transforms \
         --resize 512 384 \
         --lr 0.00003 \
         --lr_decay_step 50 \
         --lr_gamma 0.7 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model Resnet18 \
         --optimizer Adam \
         --criterion focal\
         --log_interval 100\
         --name Resnet18_0901_Stratify_age
