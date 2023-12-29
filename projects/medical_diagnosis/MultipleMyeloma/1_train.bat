rem 1_train.bat
rem 2023/12/30 (C) antillia.com
rem Modified to use
rem efficientdet-d1,
rem autoaugment_policy=v1 
rem and image_size=640x640

python ../../../efficientdet/ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./train/*.tfrecord  ^
  --val_file_pattern=./valid/*.tfrecord ^
  --model_name=efficientdet-d1 ^
  --hparams="autoaugment_policy=v1,image_size=640x640,num_classes=1,label_map=./label_map.yaml" ^
  --model_dir=./models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./eval ^
  --ckpt=../../../efficientdet/efficientdet-d1  ^
  --train_batch_size=2 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=2 ^
  --eval_samples=100  ^
  --num_examples_per_epoch=400 ^
  --num_epochs=200
