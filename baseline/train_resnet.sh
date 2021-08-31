python train.py \
         --epochs 30 \
         --dataset MaskBaseDataset \
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
         --name Resnet18_ktw
         