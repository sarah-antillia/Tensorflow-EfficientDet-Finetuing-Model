# 2_create_saved_model.sh
# 2023/12/30 (C) antillia.com
python ../../../efficientdet/SavedModelCreator.py \
  --runmode=saved_model \
  --model_name=efficientdet-d0 \
  --ckpt_path=./models  \
  --hparams="image_size=512x512,num_classes=2" \
  --saved_model_dir=./saved_model
