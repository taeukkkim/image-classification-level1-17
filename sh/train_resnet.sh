python ../train.py \
         --epochs 10 \
         --dataset MaskSplitByProfileDataset \
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
         --name Resnet18_0901_gender

python ../train.py \
         --epochs 10 \
         --dataset MaskSplitByProfileDataset \
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
         --name Resnet18_0901_age

# python train.py \
#          --epochs 20 \
#          --dataset MaskSplitByProfileDataset \
#          --label mask\
#          --augmentation get_transforms \
#          --resize 512 384 \
#          --lr 0.00003 \
#          --lr_decay_step 50 \
#          --lr_gamma 0.7 \
#          --batch_size 32 \
#          --valid_batch_size 32 \
#          --model Resnet18 \
#          --optimizer Adam \
#          --criterion focal\
#          --log_interval 100\
#          --name T2065/Resnet18_0901_mask
         
