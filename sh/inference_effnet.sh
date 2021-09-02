python ../inference.py \
         --batch_size 32 \
         --resize 512 384 \
         --model EfficientNet \
<<<<<<< HEAD:baseline/inference_effnet.sh
         --model_version tf_efficientnet_b7 \
         --name T2065/EfficientNet_b7_0902_mask \
         --output_name EfficientNet_b7_0902_mask

python inference.py \
         --batch_size 32 \
         --resize 512 384 \
         --model EfficientNet \
         --model_version tf_efficientnet_b7 \
         --name T2065/EfficientNet_b7_0902_gender \
         --output_name EfficientNet_b7_0902_gender

python inference.py \
         --batch_size 32 \
         --resize 512 384 \
         --model EfficientNet \
         --model_version tf_efficientnet_b7 \
         --name T2065/EfficientNet_b7_0902_age \
         --output_name EfficientNet_b7_0902_age
=======
         --model_version b3_pruned \
         --name EfficientNet_b3_pruned_0831 
>>>>>>> refactor:sh/inference_effnet.sh
