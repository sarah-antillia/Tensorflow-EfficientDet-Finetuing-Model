<h2>
EfficientDet-Kidney-Tumor (2024/01/22)
</h2>
This an experiment to dectet Kidney-Tumor by EfficientDet-Kidney-Tumor Model based on <a href="https://github.com/google/automl/tree/master/efficientdet">
goole/automl/efficientdet</a>.
<br>

<h3>1. Dataset Citation</h3>

The image dataset used here has been taken from the following web site.

<pre>
Kvasir-SEG Data (Polyp segmentation & detection)
https://www.kaggle.com/datasets/debeshjha1/kvasirseg
</pre>


<br>
<h2>2. Download Dataset</h2>
If you would like to train and evaluate Kidney-Tumor EfficientDet Model by yourself,
please download <b>Kidney-Tumor </b> dataset 
from the googole drive <a href="https://drive.google.com/file/d/1Erhl7YYr4MxhSO3981DsljSR9s-e9aku/view?usp=sharing">
Kidney-Tumor-EfficientDet-Dataset.zip</a>
,and expand it under this Kidney-Tumor.<br>
It contains the following datasets <br>
<pre>

Kidney-Tumor-EfficientDet-Dataset
├─classes.txt
├─label_map.pbtxt
├─label_map.yaml
├─test
├─train
└─valid
</pre>


<h3>3. Training Kidney-Tumor Model</h3>
Please move to <b>./projects/medial_diagnosis/Kidney-Tumor</b>,
and run the following bat file to train Kidney-Tumor EfficientDet Model by using the train and valid tfrecords.
<pre>
1_train.bat
</pre>

<pre>
rem 1_train.bat
rem 2024/01/20 (C) antillia.com
python ../../../efficientdet/ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./train/*.tfrecord  ^
  --val_file_pattern=./valid/*.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="learning_rate=0.05,image_size=512x512,num_classes=1,label_map=./label_map.yaml" ^
  --model_dir=./models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./eval ^
  --ckpt=../../../efficientdet/efficientdet-d0  ^
  --train_batch_size=4 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=4 ^
  --eval_samples=300  ^
  --num_examples_per_epoch=1000 ^
  --num_epochs=100
</pre>

If Linux or Windows11/WSL2, please run the following shell script.<br>
<pre>
1_train.sh
</pre>


<b>label_map.yaml:</b>
<pre>
1: 'Kidney-Tumor'
</pre>
The console output from the training process is the following, from which you can see that 
Average Precision [IoU=0.50:0.95] is not so high against expectations.<br>
<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics at epoch 50</a></b><br>
<img src="./asset/tran_console_output_at_epoch_56_0120.png" width="820" height="auto">
<br>

<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics f and map</a></b><br>
<img src="./asset/coco_metrics_at_epoch_56_0120.png" width="820" height="auto">
<br>
<br>
<b><a href="./eval/train_losses.csv">Train losses</a></b><br>
<img src="./asset/train_losses_at_epoch_56_0120.png" width="820" height="auto">
<br>
<br>

<h3>4. Create a saved_model from the checkpoint</h3>
  Please run the following bat file to create a saved_model from the checkpoint files in <b>./models</b> folder.<br> 
<pre>
2_create_saved_model.bat
</pre>
<pre>
rem 2_create_saved_model.bat
rem 2024/01/20 (C) antillia.com
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
5. Inference Kidney-Tumor by using the saved_model
</h3>
 Please run the following bat file to infer Kidney-Tumor of <b>test</b> dataset:
<pre>
3_inference.bat
</pre>
<pre>
rem 3_inference.bat
rem 2024/01/20 (C) antillia.com
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
<img src="./asset/infer_console_output_at_epoch_56_0120.png" width="820" height="auto">
<br>
<img src="./asset/test_outputs.png" width="1024" height="auto"><br>

<br>
<b><a href="./test_outputs/all_prediction.csv">all_prediction.csv</a></b><br>
<br>

<br>
<h3>
7. Some Inference results of MRI-Kidney-Tumor
</h3>
<img src="./test_outputs/10009_119.jpg" width="512" height="auto"><br>
<a href="./test_outputs/10009_119.jpg_objects.csv">10009_119.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/10025_38.jpg" width="512" height="auto"><br>
<a href="./test_outputs/10025_38.jpg_objects.csv">10025_38.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/10057_40.jpg" width="512" height="auto"><br>
<a href="./test_outputs/10057_40.jpg_objects.csv">10057_40.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/resized_0_8_10205_46.jpg" width="512" height="auto"><br>
<a href="./test_outputs/resized_0_8_10205_46.jpg_objects.csv">.csv</a><br>
<br>

<img src="./test_outputs/resized_0_8_10173_31.jpg" width="512" height="auto"><br>
<a href="./test_outputsresized_0_8_10173_31.jpg_objects.csv"/resized_0_8_10173_31.jpg_objects.csv</a><br>
<br>


<h3>
References
</h3>
