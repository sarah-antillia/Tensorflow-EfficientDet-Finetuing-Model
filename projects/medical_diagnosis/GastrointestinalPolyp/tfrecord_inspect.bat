rem tfrecord_inspect.bat
rem 2023/12/30
python ../../../efficientdet/TFRecordInspector.py ^
  ./train/*.tfrecord ^
  ./label_map.pbtxt ^
  ./Inspector/train
