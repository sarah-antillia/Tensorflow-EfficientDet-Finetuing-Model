# 1_train.sh
# 2023/12/30 (C) antillia.com
python ../../../efficientdet/ModelTrainer.py \
  --mode=train_and_eval \
  --train_file_pattern=./train/*.tfrecord  \
  --val_file_pattern=./valid/*.tfrecord \
  --model_name=efficientdet-d0 \
  --hparams="autoaugment_policy=v1,input_rand_hflip=False,image_size=512x512,num_classes=160,label_map=./label_map.yaml" \
  --model_dir=./models \
  --label_map_pbtxt=./label_map.pbtxt \
  --eval_dir=./eval \
  --ckpt=../../../efficientdet/efficientdet-d0  \
  --train_batch_size=3 \
  --early_stopping=map \
  --patience=10 \
  --eval_batch_size=3 \
  --eval_samples=800  \
  --num_examples_per_epoch=2000 \
  --num_epochs=250

