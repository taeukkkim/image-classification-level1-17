python train_T2214.py \
         --epochs 30 \
         --dataset MaskSplitByProfileDatasetMulti \
         --augmentation get_transforms \
         --resize 512 384 \
         --lr 0.00003 \
         --lr_decay_step 50 \
         --lr_gamma 0.7 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model Resnet18_multi \
         --optimizer Adam \
         --criterion focal_smoothing\
         --log_interval 20\
         --multi True \
         --name resnet18_multi
         