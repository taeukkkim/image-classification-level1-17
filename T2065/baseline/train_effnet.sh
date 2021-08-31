python train.py \
         --epochs 30 \
         --dataset MaskSplitByProfileDataset \
         --augmentation get_transforms \
         --resize 512 384 \
         --lr 0.00003 \
         --lr_decay_step 2 \
         --lr_gamma 0.7 \
         --batch_size 32 \
         --valid_batch_size 32 \
         --model EfficientNet \
         --model_version b3\
         --optimizer Adam \
         --criterion cross_entropy\
         --log_interval 100\
         --name EfficientNet_b3_0831
         

# python inference.py \
#          --batch_size 32 \  
#          --resize 512 384 \ 
#          --model EfficientNet \  
#          --model_version b3 
