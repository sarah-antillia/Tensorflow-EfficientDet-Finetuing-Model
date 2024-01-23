<h2>
EfficientDet-Liver-Tumor (Updated: 2024/01/24)
</h2>
This an experiment to dectet Liver-Tumor by EfficientDet-Liver-Tumor Model based on 
<a href="https://github.com/google/automl/tree/master/efficientdet">
goole/automl/efficientdet</a>.
<br>
<li>2024/01/24: Modified the download link to the original small dataset  
<a href="https://drive.google.com/file/d/12JUafeMhAwChEX4ofk33LFxZMvmcSOA8/view?usp=sharing">
Liver-Tumor-EfficientDet-Dataset_small.zip</a>
</li>
<br> 

<h3>1. Dataset Citation</h3>

The image dataset used here has been taken from the following web site.

<pre>
Kvasir-SEG Data (Polyp segmentation & detection)
https://www.kaggle.com/datasets/debeshjha1/kvasirseg
</pre>


<br>
<h2>2. Download Dataset</h2>
If you would like to train and evaluate Liver-Tumor EfficientDet Model by yourself,
please download <b>Liver-Tumor </b> dataset 
from the googole drive 
<a href="https://drive.google.com/file/d/12JUafeMhAwChEX4ofk33LFxZMvmcSOA8/view?usp=sharing">
Liver-Tumor-EfficientDet-Dataset_small.zip</a>
, and expand it under this Liver-Tumor.<br>
It contains the following datasets <br>
<pre>

Liver-Tumor-EfficientDet-Dataset
├─classes.txt
├─label_map.pbtxt
├─label_map.yaml
├─test
├─train
└─valid
</pre>

You can also download the latest large dataset from the googole drive 
<a href="https://drive.google.com/file/d/1P0-fafL1Xs-59D3Flh-pYPmRPOqln32W/view?usp=sharing">
Liver-Tumor-EfficientDet-Dataset.zip</a>
<br>

<h3>3. Training Liver-Tumor Model</h3>
Please move to <b>./projects/medial_diagnosis/Liver-Tumor</b>,
and run the following bat file to train Liver-Tumor EfficientDet Model by using the train and valid tfrecords.
<pre>
1_train.bat
</pre>

<pre>
rem 1_train.bat
rem 2024/01/21 (C) antillia.com
python ../../../efficientdet/ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./train/*.tfrecord  ^
  --val_file_pattern=./valid/*.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="autoaugment_policy=v1,image_size=512x512,num_classes=1,label_map=./label_map.yaml" ^
  --model_dir=./models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./eval ^
  --ckpt=../../../efficientdet/efficientdet-d0  ^
  --train_batch_size=4 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=4 ^
  --eval_samples=200  ^
  --num_examples_per_epoch=1000 ^
  --num_epochs=50
</pre>

If Linux or Windows11/WSL2, please run the following shell script.<br>
<pre>
1_train.sh
</pre>


<b>label_map.yaml:</b>
<pre>
1: 'Liver-Tumor'
</pre>
The console output from the training process is the following, from which you can see that 
Average Precision [IoU=0.50:0.95] is not so high against expectations.<br>
<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics at epoch 50</a></b><br>
<img src="./asset/train_console_at_epoch_50_0121_autoaugment_policy=v1.png" width="820" height="auto">
<br>

<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics f and map</a></b><br>
<img src="./asset/coco_metrics_at_epoch_50_0121.png" width="820" height="auto">
<br>
<br>
<b><a href="./eval/train_losses.csv">Train losses</a></b><br>
<img src="./asset/train_losses_at_epoch_50_0121.png" width="820" height="auto">
<br>
<br>

<h3>4. Create a saved_model from the checkpoint</h3>
  Please run the following bat file to create a saved_model from the checkpoint files in <b>./models</b> folder.<br> 
<pre>
2_create_saved_model.bat
</pre>
<pre>
rem 2_create_saved_model.bat
rem 2024/01/21 (C) antillia.com
python ../../../efficientdet/SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./models  ^
  --hparams="image_size=512x512,num_classes=1" ^
  --saved_model_dir=./saved_model
</pre>


If Linux or Windows11/WSL2, please run the following shell script.<br>
<pre>
2_create_saved_model.sh
</pre>

<br>

<h3>
5. Inference Liver-Tumor by using the saved_model
</h3>
 Please run the following bat file to infer Liver-Tumor of <b>test</b> dataset:
<pre>
3_inference.bat
</pre>
<pre>
rem 3_inference.bat
rem 2024/01/21 (C) antillia.com
python ../../../efficientdet/SavedModelInferencer.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./saved_model ^
  --min_score_thresh=0.4 ^
  --hparams="num_classes=1,label_map=./label_map.yaml" ^
  --input_image=./test/*.jpg ^
  --classes_file=./classes.txt ^
  --ground_truth_json=./test/annotation.json ^
  --output_image_dir=./test_outputs
</pre>
If Linux or Windows11/WSL2, please run the following shell script.<br>
<pre>
3_inference.sh
</pre>
Inference console output<br>
<img src="./asset/inference_console_at_epoch_50_0121_autoaugment_policy=v1.png" width="820" height="auto">
<br>
<img src="./asset/test_outputs.png" width="1024" height="auto"><br>

<br>
<b><a href="./test_outputs/all_prediction.csv">all_prediction.csv</a></b><br>
<br>

<br>
<h3>
7. Some Inference results of Liver-Tumor
</h3>
<img src="./test_outputs/flipped_10001_81.jpg" width="512" height="auto"><br>
<a href="./test_outputs/flipped_10001_81.jpg_objects.csv">flipped_10001_81.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/flipped_10004_130.jpg" width="512" height="auto"><br>
<a href="./test_outputs/flipped_10004_130.jpg_objects.csv">flipped_10004_130.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/flipped_10010_114.jpg" width="512" height="auto"><br>
<a href="./test_outputs/flipped_10010_114.jpg_objects.csv">flipped_10010_114.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/mirrored_10118_110.jpg" width="512" height="auto"><br>
<a href="./test_outputs/mirrored_10118_110.jpg_objects.csv">mirrored_10118_110.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/mirrored_10105_112.jpg" width="512" height="auto"><br>
<a href="./test_outputs/mirrored_10105_112.jpg_objects.csv">mirrored_10105_112.jpg_objects.csv</a><br>
<br>


<h3>
References
</h3>
