# 1_train.sh
# 2024/01/20 (C) antillia.com
python ../../../efficientdet/ModelTrainer.py \
  --mode=train_and_eval \
  --train_file_pattern=./train/*.tfrecord  \
  --val_file_pattern=./valid/*.tfrecord \
  --model_name=efficientdet-d0 \
  --hparams="learning_rate=0.05,image_size=512x512,num_classes=1,label_map=./label_map.yaml" \
  --model_dir=./models \
  --label_map_pbtxt=./label_map.pbtxt \
  --eval_dir=./eval \
  --ckpt=../../../efficientdet/efficientdet-d0  \
  --train_batch_size=4 \
  --early_stopping=map \
  --patience=10 \
  --eval_batch_size=4 \
  --eval_samples=100  \
  --num_examples_per_epoch=400 \
  --num_epochs=100

