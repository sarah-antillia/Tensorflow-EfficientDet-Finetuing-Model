rem tfrecord_inspect.bat
python ../../../efficientdet/TFRecordInspector.py ^
  ./train/*.tfrecord ^
  ./label_map.pbtxt ^
  ./Inspector/train
