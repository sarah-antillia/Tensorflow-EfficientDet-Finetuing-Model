# 3_inference.sh
# 2024/01/21 (C) antillia.com
python ../../../efficientdet/SavedModelInferencer.py \
  --runmode=saved_model_infer \
  --model_name=efficientdet-d0 \
  --saved_model_dir=./saved_model \
  --min_score_thresh=0.4 \
  --hparams="num_classes=1,label_map=./label_map.yaml" \
  --input_image=./test/*.jpg \
  --classes_file=./classes.txt \
  --ground_truth_json=./test/annotation.json \
  --output_image_dir=./test_outputs
