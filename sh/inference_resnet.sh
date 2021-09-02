
python ../inference.py \
         --batch_size 32 \
         --resize 512 384 \
         --model Resnet18 \
         --name Resnet18_0901_gender \
         --output_name resnet18_0901_gender

python ../inference.py \
         --batch_size 32 \
         --resize 512 384 \
         --model Resnet18 \
         --name Resnet18_0901_age \
         --output_name resnet18_0901_age

# python inference.py \
#          --batch_size 32 \
#          --resize 512 384 \
#          --model Resnet18 \
#          --name T2065/Resnet18_0901_mask \
#          --output_name resnet18_0901_mask

