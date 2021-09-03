python inference.py \
         --batch_size 32 \
         --resize 512 384 \
         --model EfficientNet \
         --model_version tf_efficientnet_b7 \
         --name EfficientNet_b7_0902_mask \
         --output_name EfficientNet_b7_0902_mask

python inference.py \
         --batch_size 32 \
         --resize 512 384 \
         --model EfficientNet \
         --model_version tf_efficientnet_b7 \
         --name EfficientNet_b7_0902_gender \
         --output_name EfficientNet_b7_0902_gender

python inference.py \
         --batch_size 32 \
         --resize 512 384 \
         --model EfficientNet \
         --model_version tf_efficientnet_b7 \
         --name EfficientNet_b7_0902_age \
         --output_name EfficientNet_b7_0902_age