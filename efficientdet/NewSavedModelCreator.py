# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Tool to inspect a model."""

# 2023/12/15 NewSavedModelCreator.py
#

import os
import sys
sys.path.append("../../")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION']='false'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from typing import Text, Tuple, List
import platform

from absl import app

from absl import logging
import shutil

import numpy as np
from PIL import Image
#import tensorflow.compat.v1 as tf
import tensorflow as tf

import hparams_config

#import inference
import inference2 as inference

import utils
#from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import

#from DetectResultsWriter  import DetectResultsWriter
#from ModelInspector       import ModelInspector

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from ConfigParser import ConfigParser

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


PROJECT  = "project"
HARDWARE = "hardware"
MODEL    = "model"
TRAIN    = "train"
VALID    = "valid"
HYPER_PARAMETERS= "hyper_parameters"

class SavedModelCreator(ModelInspector):
  """A simple helper class for inspecting a model."""

  def __init__(self,
               model_name: Text,
               logdir: Text,
               tensorrt: Text = False,
               use_xla: bool = False,
               ckpt_path: Text = None,
               export_ckpt: Text = None,
               saved_model_dir: Text = None,
               tflite_path: Text = None,
               batch_size: int = 1,
               hparams: Text = '',
               **kwargs):  # pytype: disable=annotation-type-mismatch
    super().__init__(
               model_name,
               logdir,
               tensorrt,
               use_xla,
               ckpt_path,
               export_ckpt,
               saved_model_dir,
               tflite_path,
               batch_size,
               hparams,
               **kwargs)

  def export_saved_model(self, **kwargs):
    """Export a saved model for inference."""
    #tf.enable_resource_variables()
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.build()
    # 2022/05/13 Try to rmtree(saved_model_dir)
    if os.path.exists(self.saved_model_dir):
      shutil.rmtree(self.saved_model_dir)
      print("=== rmtree saved_model_dir {}".format(self.saved_model_dir))
    """
    temp_dir = FLAGS.saved_model_dir + "/variables/variables_temp"
    print("=== create temp_dir {}".format(temp_dir))
    if not os.path.exists(temp_dir):
      os.makedirs(temp_dir)
    """
    driver.export(self.saved_model_dir, self.tflite_path, self.tensorrt)


def main(_):

  train_eval_infer_config = ""
  if len(sys.argv)==2:
    train_eval_infer_config = sys.argv[1]
  else:
    raise Exception("Usage: python SavedModelCreator.py train_eval_infer.config")

  if os.path.exists(train_eval_infer_config) == False:
    raise Exception("Not found train_config {}".format(train_eval_infer_config)) 
  
  parser = ConfigParser(train_eval_infer_config)
  parser.dump_all()

  #ckpt_path 
  model_dir = parser.get(MODEL, "model_dir")
    
  org_model_dir = model_dir
  osname = platform.platform()
  print("=== Platform {}".format(osname))
  if "Windows" in osname:
    model_dir = model_dir.replace("/", "\\")
    print("--- Converted model_dir from:{} to:{}".format(org_model_dir, model_dir))

  if not os.path.exists(model_dir):
    raise Exception("FATAL ERROR: Not found model_dir {}".format(model_dir))

  saved_model_dir = parser.get(MODEL, "saved_model_dir")
  org_saved_model_dir = saved_model_dir
  osname = platform.platform()
  print("=== Platform {}".format(osname))
  if "Windows" in osname:
    saved_model_dir = saved_model_dir.replace("/", "\\")
    print("--- Converted model_dir from:{} to:{}".format(org_saved_model_dir, saved_model_dir))

  if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
    
  model_name = parser.get(MODEL, "model_name")
  
  num_classes = parser.get(HYPER_PARAMETERS, "num_classes") 
  image_size  = parser.get(HYPER_PARAMETERS, "image_size")
  hparams     = "image_size=" + image_size + ",num_classes=" + str(num_classes)

  runmode = "saved_model"
  use_xla = parser.get(HARDWARE, "use_xla")
  tflite_path = None
  export_ckpt = parser.get(MODEL, "export_ckpt", dvalue=None)
  batch_size  = parser.get(TRAIN, "batch_size")
  input_image = None
  output_image_dir=None
  input_video=None
  output_video=None
  line_thickness=2

  
max_boxes_to_draw = parser.get(MODEL, "max_boxes_to_draw")
  nms_method = parser.get(MODEL, "nms_method")
  
  #hparams="image_size=512x512,num_classes=78"
  creator = SavedModelCreator(
      model_name=model_name,
      logdir=None,
      tensorrt=None,
      use_xla=use_xla,
      ckpt_path=model_dir,
      export_ckpt=export_ckpt,
      saved_model_dir=saved_model_dir,
      tflite_path=tflite_path,
      batch_size= batch_size,
      hparams = hparams,
      score_thresh=min_score_thresh,
      max_output_size=max_boxes_to_draw,
      nms_method=nms_method)

  creator.run_model(
      runmode,
      input_image=FLAGS.input_image,
      output_image_dir=FLAGS.output_image_dir,
      input_video=FLAGS.input_video,
      output_video=FLAGS.output_video,
      line_thickness=FLAGS.line_thickness,
      max_boxes_to_draw=FLAGS.max_boxes_to_draw,
      min_score_thresh=FLAGS.min_score_thresh,
      nms_method=FLAGS.nms_method,
      bm_runs=FLAGS.bm_runs,
      threads=FLAGS.threads,
      trace_filename=FLAGS.trace_filename)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  #tf.enable_v2_tensorshape()
  #tf.disable_eager_execution()
  if tf.executing_eagerly():
   tf.compat.v1.disable_eager_execution()
  app.run(main)
