<h2>
EfficientDet-Ovarian-Tumor-1Class
</h2>
Please see also our first experiment:<a href="https://github.com/sarah-antillia/EfficientDet-Augmented-Ovarian-Ultrasound-Images">
EfficientDet-Augmented-Ovarian-Ultrasound-Images</a>
<br>

<h3>1. Dataset Citation</h3>
The original dataset used here has been take from the following web site:<br>
<b>Multi-Modality Ovarian Tumor Ultrasound (MMOTU) image dataset</b><br>
<pre>
https://github.com/cv516buaa/mmotu_ds2net
</pre>
Citation:<br>
<pre>
@article{DBLP:journals/corr/abs-2207-06799,
  author    = {Qi Zhao and
               Shuchang Lyu and
               Wenpei Bai and
               Linghan Cai and
               Binghao Liu and
               Meijing Wu and
               Xiubo Sang and
               Min Yang and
               Lijiang Chen},
  title     = {A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised
               Cross-Domain Semantic Segmentation},
  journal   = {CoRR},
  volume    = {abs/2207.06799},
  year      = {2022},
}
</pre>

<b>See also:</b>
<p>
A Multi-Modality Ovarian Tumor Ultrasound
Image Dataset for Unsupervised Cross-Domain
Semantic Segmentation
</p>
<pre>
https://arxiv.org/pdf/2207.06799v3.pdf
</pre>


<h3>2. Download Dataset</h3>

If you would like to train and evaluate Ovarian-Tumor-1Class EfficientDet Model by yourself, 
please download the <b>TFRecord-OTUSI-20230420.zip</b>
from <a href="https://drive.google.com/file/d/1D3q1iuOfdBWqG-Zw_ugQfX0mNwd_sD4e/view?usp=sharing"><b>Ovarian-Tumor-1Class-EfficientDet-Dataset.zip</b></a>
</p>

<h3>3. Train Ovarian-Tumor Model by using the pretrained-model</h3>

Please move to <b>./projects/medical_diagnosis/Ovarian-Tumor-1Class</b> directory,
and run the following bat file to train Ovarian-Tumor EfficientDet Model by using the train and valid tfrecords.
<pre>
1_train.bat
</pre>

<pre>
rem 1_train.bat
rem 2024/01/30 (C) antillia.com
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
  --eval_samples=400  ^
  --num_examples_per_epoch=1200 ^
  --num_epochs=100
</pre>
In case of Linux or Windows/WSL2, please run the following shell script instead of the above bat file.<br>
<pre>
1_train.sh
</pre>

<br>
<b>label_map.yaml:</b>
<pre>
1: 'Ovarian-Tumor'
</pre>
The console output from the training process is the following, from which you can see that 
<b>Average Precision [IoU=0.50:0.95]</b> is very low, 
however it is better than that of the previous value of <a href="https://github.com/sarah-antillia/EfficientDet-Augmented-Ovarian-Ultrasound-Images">Ovarian-Tumor-8Classes.</a>
<br>
<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics at epoch 67</a></b><br>
<img src="./asset/train_console_output_at_epoch_67.png" width="820" height="auto">
<br>

<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics f and map</a></b><br>
<img src="./asset/train_metrics.png" width="820" height="auto">
<br>
<br>
<b><a href=".projects/medical_diagnosis/Ovarian-Tumor/eval/train_losses.csv">Train losses</a></b><br>
<img src="./asset/train_losses.png" width="820" height="auto">
<br>
<br>

<b><a href="./eval/coco_ap_per_class.csv">COCO ap per class</a></b><br>
<img src="./asset/coco_ap_per_class_at_epoch_67.png" width="820" height="auto">
<br>

<h3>
4. Create a saved_model from the checkpoint
</h3>
  Please run the following bat file to create a saved_model from the checkpoint files in <b>./models</b> folder.<br> 
<pre>
2_create_saved_model.bat
</pre>
<pre>

</pre>
In case of Linux or Windows/WSL2, please run the following shell script instead of the above bat file.<br>
<pre>
2_create_saved_model.sh
</pre>
<pre>
rem 2_create_saved_model.bat
rem 2024/01/30 (C) antillia.com
python ../../../efficientdet/SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./models  ^
  --hparams="image_size=512x512,num_classes=1" ^
  --saved_model_dir=./saved_model
</pre>
<br>
<h3>
5. Inference Ovarian Tumor by using the saved_model
</h3>
 Please run the following bat file to infer Ovarian Tumor images in <b>test</b> dataset:
<pre>
3_inference.bat
</pre>
<pre>
rem 3_inference.bat
rem 2024/01/30 (C) antillia.com
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

In case of Linux or Windows/WSL2, please run the following shell script instead of the above bat file.<br>
<pre>
3_inference.sh
</pre>

<br>
Inference test_outputs:<br>
<img src="./asset/inference_test_output_samples.png" width="1024" height="auto">
<br>
<h3>
7.2. Some Enlarged Inference Results
</h3>
<img src="./test_outputs/rotated_300_1469.jpg" width="512" height="auto"><br>
<a href="./test_outputs/rotated_300_1469.jpg_objects.csv">rotated_300_1469.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/rotated_180_124.jpg" width="512" height="auto"><br>
<a href="./test_outputs/rotated_180_124.jpg_objects.csv">rotated_180_124.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/rotated_120_960.jpg" width="512" height="auto"><br>
<a href="./test_outputs/rotated_120_960.jpg_objects.csv">rotated_120_960.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/flipped_339.jpg" width="512" height="auto"><br>
<a href="./test_outputs/flipped_339.jpg_objects.csv">flipped_339.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/flipped_6.jpg" width="512" height="auto"><br>
<a href="./test_outputs/flipped_6.jpg_objects.csv">flipped_6.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/mirrored_235.jpg" width="512" height="auto"><br>
<a href="./test_outputs/mirrored_235.jpg_objects.csv">mirrored_235.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/1461.jpg" width="512" height="auto"><br>
<a href="./test_outputs/1461.jpg_objects.csv">1461.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/1389.jpg" width="512" height="auto"><br>
<a href="./test_outputs/1389.jpg_objects.csv">1389.jpg_objects.csv</a><br>
<br>

<img src="./test_outputs/1290.jpg" width="512" height="auto"><br>
<a href="./test_outputs/1290.jpg_objects.csv">1290.jpg_objects.csv</a><br>
<br>

<h3>7.3. COCO metrics of inference result</h3>
The 3_inference.bat computes also the COCO metrics(f, map, mar) to the <b>test_dataset</b> as shown below:<br>
<a href="./test_outputs/prediction_f_map_mar.csv">prediction_f_map_mar.csv</a>

<br>
<b><a href="./eval/coco_metrics.csv">COCO metrics at epoch 54</a></b><br>
<img src="./asset/inference_console_output_at_epoch_67.png" width="820" height="auto">
<br>
<p>
<p>
From the picture above, you can see that <b>Average Precision @[IoU=0.50:0.05]</b> is very low,
however it is better than that of the previous value of <a href="https://github.com/sarah-antillia/EfficientDet-Augmented-Ovarian-Ultrasound-Images">Ovarina-Tumor-8Classes.</a

<br>
<h3>
References
</h3>
<b>1. A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation</b><br>
Qi Zhao*, Member, IEEE, Shuchang Lyu*, Graduate Student Member, IEEE, Wenpei Bai*, Linghan Cai*,<br>
Binghao Liu, Meijing Wu, Xiubo Sang, Min Yang, Lijiang Chen, Member, IEEE<br>

<pre>
https://arxiv.org/pdf/2207.06799v3.pdf
</pre>

<b>2. Ovarian tumor diagnosis using deep convolutional neural networks and a denoising convolutional autoencoder</b><br>
Yuyeon Jung, Taewan Kim, Mi-Ryung Han, Sejin Kim, Geunyoung Kim, Seungchul Lee & Youn Jin Choi <br>
<pre>
https://www.nature.com/articles/s41598-022-20653-2

</pre>
