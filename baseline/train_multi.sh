# python train_T2214.py \
#          --epochs 100 \
#          --dataset MaskSplitByProfileDatasetMulti \
#          --augmentation get_transforms \
#          --model Resnet18_multi \
#          --optimizer Adam \
#          --criterion focal_smoothing\
#          --resize 512 384 \
#          --lr 0.001 \
#          --lr_decay_step 30 \
#          --lr_gamma 0.5 \
#          --batch_size 64 \
#          --valid_batch_size 64 \
#          --log_interval 20\
#          --multi True \
#          --name T2214/resnet18_multi
         
python train_T2214.py \
         --epochs 50 \
         --dataset MaskSplitStratifyDatasetMulti \
         --augmentation get_transforms \
         --model EfficientNet_multi \
         --optimizer Adam \
         --criterion focal_smoothing_f1\
         --resize 256 192 \
         --lr 0.001 \
         --lr_decay_step 25 \
         --lr_gamma 0.1 \
         --batch_size 128 \
         --valid_batch_size 128 \
         --log_interval 20\
         --multi True \
         --name T2214/EfficientNet_multi_1e-4_focal_label_f1_weight_decay25_gamma01
         