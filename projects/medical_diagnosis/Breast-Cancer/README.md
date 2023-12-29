<h2>
EfficientDet-Breast-Cancer
</h2>
Please see also our first experiment:<br>
<a href=" https://github.com/sarah-antillia/EfficientDet-Breast-Cancer">EfficientDet-Breast-Cancer</a>
<br>

<h2>
1. Dataset Citation
</h2>
The original dataset <b>Breast Ultrasound Images Dataset (BUSI)</b> has been taken
from the following website.<br>
<pre>
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
</pre>
Citation:<br>
<pre>
Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. 
Dataset of breast ultrasound images. Data in Brief. 
2020 Feb;28:104863. 
DOI: 10.1016/j.dib.2019.104863.
</pre>

<h2>2. Download Dataset</h2>
If you would like to train and evaluate Breast-Cancer EfficientDet Model by yourself,
please download <b>TFRecord</b> dataset <b>Break-Cancer(BUSI)</b> train and valid dataset 
from the following Google drive.<br>

<a href="https://drive.google.com/file/d/1XaqPnH90ZQ9_FuwaUZSvwcUPWS7BdRvE/view?usp=sharing">TFRecord-BUSI-20230414.zip</a>.
<br>


<h3>3. Train Breast-Cancer Model by using the pretrained-model</h3>
Please move to <b>./projects/medical_diagnosis/Breast-Cancer</b>,
and run the following bat file to train Breast-Cancer EfficientDet Model by using the train and valid tfrecords.
<pre>
1_train.bat
</pre>

<pre>
rem 1_train.bat
python ../../ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./train/*.tfrecord  ^
  --val_file_pattern=./valid/*.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="autoaugment_policy=v2,learning_rate=0.01,image_size=512x512,num_classes=2,label_map=./label_map.yaml" ^
  --model_dir=./models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./eval ^
  --ckpt=../../efficientdet-d0  ^
  --train_batch_size=4 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=4 ^
  --eval_samples=200  ^
  --num_examples_per_epoch=400 ^
  --num_epochs=100
</pre>
If Linux or Windows11/WSL2, please run the following shell script.<br>
<pre>
1_train.sh
</pre>

<br>
<b>label_map.yaml:</b>
<pre>
1: 'benign'
2: 'malignant'
</pre>
The console output from the training process is the following, from which you can see that 
mAP [IoU=0.50:0.95] is very low.<br>
<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics at epoch 87</a></b><br>
<img src="./asset/tran_console_output_at_epoch_87.png" width="820" height="auto">
<br>

<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics f and map</a></b><br>
<img src="./asset/coco_metrics_at_epoch_87.png" width="820" height="auto">
<br>
<br>
<b><a href="./eval/train_losses.csv">Train losses</a></b><br>
<img src="./asset/train_losses_at_epoch_87.png" width="820" height="auto">
<br>
<br>

<b><a href="./eval/coco_ap_per_class.csv">COCO ap per class</a></b><br>
<img src="./asset/coco_ap_per_class_at_epoch_87.png" width="820" height="auto">
<br>

<h3>
4. Create a saved_model from the checkpoint
</h3>
  Please run the following bat file to create a saved_model from the checkpoint files in <b>./models</b> folder.<br> 
<pre>
2_create_saved_model.bat
</pre>
<pre>
rem 2_create_saved_model.bat  
python ../../SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./models  ^
  --hparams="image_size=512x512,num_classes=2" ^
  --saved_model_dir=./saved_model
</pre>
If Linux or Windows11/WSL2, please run the following shell script.<br>
<pre>
2_create_saved_model.sh
</pre>

<br>

<h3>
5. Inference Breast-Cancer by using the saved_model
</h3>
 Please run the following bat file to infer Breast Cancer mages of test dataset:
<pre>
3_inference.bat
</pre>
<pre>
rem 3_inference.bat
python ../../SavedModelInferencer.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./saved_model ^
  --min_score_thresh=0.4 ^
  --hparams="num_classes=2,label_map=./label_map.yaml" ^
  --input_image=./test/*.jpg ^
  --classes_file=./classes.txt ^
  --ground_truth_json=./test/annotation.json ^
  --output_image_dir=./test_outputs
</pre>

If Linux or Windows11/WSL2, please run the following shell script.<br>
<pre>
3_inference.sh
</pre>



<br>
<h3>6. Some Inference results of Breast-Cancer</h3>
<img src="./test_outputs/benign (1).jpg" width="512" height="auto"><br>
<a href="./test_outputs//benign (1).jpg_objects.csv">benign (1).jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/benign (7).jpg" width="512" height="auto"><br>
<a href="./test_outputs//benign (7).jpg_objects.csv">benign (7).jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/benign (19).jpg" width="512" height="auto"><br>
<a href="./test_outputs//benign (19).jpg_objects.csv">benign (19).jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/benign (33).jpg" width="512" height="auto"><br>
<a href="./test_outputs//benign (33).jpg_objects.csv">benign (33).jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/benign (54).jpg" width="512" height="auto"><br>
<a href="./test_outputs//benign (54).jpg_objects.csv">benign (54).jpg_objects.csv</a><br>
<br>



<img src="./test_outputs/malignant (34).jpg" width="512" height="auto"><br>
<a href="./test_outputs//malignant (34).jpg_objects.csv">malignant (34).jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/malignant (71).jpg" width="512" height="auto"><br>
<a href="./test_outputs//malignant (71).jpg_objects.csv">malignant (71).jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/malignant (128).jpg" width="512" height="auto"><br>
<a href="./test_outputs//malignant (128).jpg_objects.csv">malignant (1).jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/malignant (196).jpg" width="512" height="auto"><br>
<a href="./test_outputs//malignant (196).jpg_objects.csv">malignant (196).jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/malignant (202).jpg" width="512" height="auto"><br>
<a href="./test_outputs//malignant (202).jpg_objects.csv">malignant (202).jpg_objects.csv</a><br>
<br>


<h3>7. COCO metrics of inference result</h3>
<p>
The 3_inference.bat computes also the COCO metrics(f, map, mar) to the <b>test</b> dataset as shown below.
</p>

<a href="./test_outputs/prediction_f_map_mar.csv">prediction_f_map_mar.csv</a>

<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics at epoch 87</a></b><br>
<img src="./asset/inference_console_output_at_epoch_87.png" width="820" height="auto">
<br>
<p>

