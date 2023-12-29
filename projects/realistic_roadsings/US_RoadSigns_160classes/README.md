<h2>
EfficientDet-US-RoadSigns-160classes
</h2>
Please see also our first experiment:
<a href="https://github.com/atlan-antillia/EfficientDet-Slightly-Realistic-USA-RoadSigns-160classes">EfficientDet-Slightly-Realistic-USA-RoadSigns-160classes</a>

<br>
<h3>1 Download TFRecord dataset</h3>
If you would like to train and evalute US-RoadSigns EfficientDet Model by yourself,
please download TRecord_USA_RoadSigns 160classes_V7.1 dataset from 
<a href="https://drive.google.com/drive/folders/1Uq0kw1-Y72Lrwu07ddNuQ3LgOa0J6Whh?usp=sharing">USA_RoadSigns_160classes_V7.1</a>
<br>
Please put the downloaded train and valid dataset in 
<b>./projects/slightly_realistic_roadsigns/US_RoadSigns_160classes</b> folder.

<h3>2. Train RoadSign Model</h3>
Please move to <b>US_RoadSigns_160classes</b> directory,
and run the following bat file to train RoadSigns Model by using the train and valid tfrecords.<br>
Please note that we have used the following <b>autoaugmentation_policy</b> to improve object detection accuracy(f-value, map, mar) of this model 
<pre>
autoaugment_policy=v1
</pre> 
in that bat file.
<pre>
1_train.bat
</pre>

<pre>
rem 1_train.bat
python ../../../efficientdet/ModelTrainer.py ^
  --mode=train_and_eval ^
  --train_file_pattern=./train/*.tfrecord  ^
  --val_file_pattern=./valid/*.tfrecord ^
  --model_name=efficientdet-d0 ^
  --hparams="autoaugment_policy=v1,input_rand_hflip=False,image_size=512x512,num_classes=160,label_map=./label_map.yaml" ^
  --model_dir=./models ^
  --label_map_pbtxt=./label_map.pbtxt ^
  --eval_dir=./eval ^
  --ckpt=../../../efficientdet/efficientdet-d0  ^
  --train_batch_size=3 ^
  --early_stopping=map ^
  --patience=10 ^
  --eval_batch_size=3 ^
  --eval_samples=800  ^
  --num_examples_per_epoch=2000 ^
  --num_epochs=250
</pre>

In case of Linux or Windows/WSL2, please run the following shell script instead of the above bat file.<br>
<pre>
1_train.sh
</pre>
<br>
<b>label_map.yaml:</b>
<pre>
1: '270_degree_loop'
2: 'Added_lane'
3: 'Added_lane_from_entering_roadway'
4: 'All_way'
5: 'Be_prepared_to_stop'
6: 'Bicycle_wrong_way'
7: 'Bicycles'
8: 'Bicycles_and_pedestrians'
9: 'Bicycles_left_pedestrians_right'
10: 'Bike_lane'
11: 'Bike_lane_slippery_when_wet'
12: 'Bump'
13: 'Bus_lane'
14: 'Center_lane'
15: 'Chevron_alignment'
16: 'Circular_intersection_warning'
17: 'Cross_roads'
18: 'Curve'
19: 'Dead_end'
20: 'Deer_crossing'
21: 'Detour'
22: 'Detour_right'
23: 'Dip'
24: 'Do_not_drive_on_tracks'
25: 'Do_not_enter'
26: 'Do_not_pass_stopped_trains'
27: 'Double_side_roads'
28: 'Emergency_signal'
29: 'End_detour'
30: 'Except_right_turn'
31: 'Fallen_rocks'
32: 'Flagger_present'
33: 'Fog_area'
34: 'Go_on_slow'
35: 'Golf_cart_crossing'
36: 'Gusty_winds_area'
37: 'Hairpin_curve'
38: 'Hazardous_material_prohibited'
39: 'Hazardous_material_route'
40: 'Hidden_driveway'
41: 'Hill_bicycle'
42: 'Horizontal_alignment_intersection'
43: 'Horse_drawn_vehicle_ahead'
44: 'Keep_left'
45: 'Keep_left_2'
46: 'Keep_right'
47: 'Keep_right_2'
48: 'Lane_ends'
49: 'Left_lane'
50: 'Left_turn_only'
51: 'Left_turn_or_straight'
52: 'Left_turn_yield_on_green'
53: 'Loading_zone'
54: 'Low_clearance'
55: 'Low_ground_clearance_railroad_crossing'
56: 'Merge'
57: 'Merging_traffic'
58: 'Metric_low_clearance'
59: 'Minimum_speed_limit_40'
60: 'Minimum_speed_limit_60km'
61: 'Narrow_bridge'
62: 'National_network_prohibited'
63: 'National_network_route'
64: 'Night_speed_limit_45'
65: 'Night_speed_limit_70km'
66: 'No_bicycles'
67: 'No_entre'
68: 'No_hitch_hiking'
69: 'No_horseback_riding'
70: 'No_large_trucks'
71: 'No_left_or_u_turn'
72: 'No_left_turn'
73: 'No_left_turn_across_tracks'
74: 'No_outlet'
75: 'No_parking'
76: 'No_parking_any_time'
77: 'No_parking_bus_stop'
78: 'No_parking_from_830am_to_530pm'
79: 'No_parking_from_830am_to_530pm_2'
80: 'No_parking_in_fire_lane'
81: 'No_parking_loading_zone'
82: 'No_parking_on_pavement'
83: 'No_pedestrian_crossing'
84: 'No_pedestrians'
85: 'No_right_turn'
86: 'No_rollerblading'
87: 'No_standing_any_time'
88: 'No_stopping_on_pavement'
89: 'No_straight_through'
90: 'No_train_horn_warning'
91: 'No_turns'
92: 'No_u_turn'
93: 'No_unauthorized_vehicles'
94: 'Offset_roads'
95: 'One_direction'
96: 'One_way'
97: 'Parking_with_time_restrictions'
98: 'Pass_on_either_side'
99: 'Pass_road'
100: 'Path_narrows'
101: 'Pedestrian_crossing'
102: 'Railroad_crossing'
103: 'Railroad_crossing_ahead'
104: 'Railroad_intersection_warning'
105: 'Ramp_narrows'
106: 'Reserved_parking_wheelchair'
107: 'Reverse_curve'
108: 'Reverse_turn'
109: 'Right_lane'
110: 'Right_turn_only'
111: 'Right_turn_or_straight'
112: 'Road_closed'
113: 'Road_closed_ahead'
114: 'Road_narrows'
115: 'Road_slippery_when_wet'
116: 'Rough_road'
117: 'Runaway_vehicles_only'
118: 'School'
119: 'School_advance'
120: 'School_bus_stop_ahead'
121: 'School_bus_turn_ahead'
122: 'School_speed_limit_ahead'
123: 'Sharp_turn'
124: 'Side_road_at_a_perpendicular_angle'
125: 'Side_road_at_an_acute_angle'
126: 'Single_lane_shift_left'
127: 'Skewed_railroad_crossing'
128: 'Snowmobile'
129: 'Speed_limit_50'
130: 'Speed_limit_80km'
131: 'Stay_in_lane'
132: 'Steep_grade'
133: 'Steep_grade_percentage'
134: 'Stop'
135: 'Stop_here_for_pedestrians'
136: 'Stop_here_for_peds'
137: 'Stop_here_when_flashing'
138: 'Straight_ahead_only'
139: 'T_roads'
140: 'Tractor_farm_vehicle_crossing'
141: 'Tractor_farm_vehicle_crossing_2'
142: 'Truck crossing_2'
143: 'Truck_crossing'
144: 'Truck_rollover_warning'
145: 'Truck_route_sign'
146: 'Truck_speed_limit_40'
147: 'Turn_only_lanes'
148: 'Turning_vehicles_yield_to_pedestrians'
149: 'Two_direction'
150: 'Two_way_traffic'
151: 'Wait_on_stop'
152: 'Weight_limit_10t'
153: 'Winding_road'
154: 'Work_zone_for_speed_limit'
155: 'Workers_on_road'
156: 'Wrong_way'
157: 'Y_roads'
158: 'Yield'
159: 'Yield_here_to_pedestrians'
160: 'Yield_here_to_peds'
</pre>
<br>
<br>
<b>traing_console_output_at_epoch232</b><br>
<b><a href="./eval/coco_metrics.csv">COCO metrics at epoch 232</a></b><br>
<img src="./asset/train_console_output_at_epoch232.png" width="1024" height="auto">
<br>

<br>
<b><a href="./eval/coco_metrics.csv">COCO meticss f and map</a></b><br>
<img src="./asset/coco_metrics_at_epoch232.png" width="1024" height="auto">
<br>
<br>
<b><a href="./eval/train_losses.csv">Train losses</a></b><br>
<img src="./asset/train_losses_at_epoch232.png" width="1024" height="auto">
<br>
<br>

<b><a href="./eval/coco_ap_per_class.csv">COCO ap per class</a></b><br>
<img src="./asset/coco_ap_per_class_at_epoch232.png" width="1024" height="auto">
<br>

<h3>
3. Create a saved_model from the checkpoint
</h3>
  Please run the following bat file to create a saved_model from the checkpoint files in <b>./models</b> folder.<br> 
<pre>
2_create_saved_model.bat
</pre>
<pre>
rem 2_create_saved_model.bat
python ../../../efficientdet/SavedModelCreator.py ^
  --runmode=saved_model ^
  --model_name=efficientdet-d0 ^
  --ckpt_path=./models  ^
  --hparams="image_size=512x512,num_classes=160" ^
  --saved_model_dir=./saved_model
</pre>
In case of Linux or Windows/WSL2, please run the following shell script instead of the above bat file.<br>
<pre>
2_create_saved_model.sh
</pre>

<br>
<h3>
4. Inference RoadSigns by using the saved_model
</h3>
 Please run the following bat file to infer the roadsigns in images of test_dataset:
<pre>
3_inference.bat
</pre>

<pre>
rem 3_inference.bat
python ../../../efficientdet/SavedModelInferencer.py ^
  --runmode=saved_model_infer ^
  --model_name=efficientdet-d0 ^
  --saved_model_dir=./saved_model ^
  --min_score_thresh=0.4 ^
  --hparams="num_classes=160,label_map=./label_map.yaml" ^
  --input_image=./realistic_test_dataset/*.jpg ^
  --classes_file=./classes.txt ^
  --ground_truth_json=./realistic_test_dataset/annotation.json ^
  --output_image_dir=./realistic_test_dataset_outputs
</pre>

<br>
<h3>
5. Some Inference results of US RoadSigns
</h3>

<img src="./realistic_test_dataset_outputs/usa_roadsigns_1001.jpg" width="1280" height="auto"><br>
<a href="./realistic_test_dataset_outputs/usa_roadsigns_1001.jpg_objects.csv">roadsigns1001.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1012.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1012.jpg_objects.csv">roadsigns1002.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1023.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1023.jpg_objects.csv">roadsigns1003.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1034.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1034.jpg_objects.csv">roadsigns1004.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1045.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1045.jpg_objects.csv">roadsigns1005.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1056.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1056.jpg_objects.csv">roadsigns1006.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1067.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1067.jpg_objects.csv">roadsigns1007.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1078.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1078.jpg_objects.csv">roadsigns1008.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1089.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1089.jpg_objects.csv">roadsigns1009.jpg_objects.csv</a><br>
<br>
<img src="./realistic_test_dataset_outputs/usa_roadsigns_1099.jpg" width="1280" height="auto"><br>
<a  href="./realistic_test_dataset_outputs/usa_roadsigns_1099.jpg_objects.csv">roadsigns1010.jpg_objects.csv</a><br>
<br>

<h3>5. COCO metrics of inference result</h3>
The 3_inference.bat computes also the COCO metrics(f, map, mar) to the <b>realistic_test_dataset</b> as shown below:<br>
<a href="./realistic_test_dataset_outputs/prediction_f_map_mar.csv">prediction_f_map_mar.csv</a>

<br>
<img src="./asset/inference_console_output.png" width="740" height="auto"><br>
