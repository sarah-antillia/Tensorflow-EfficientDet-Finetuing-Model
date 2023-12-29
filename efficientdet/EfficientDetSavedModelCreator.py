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

# Copyright 2020-2024 antillia.com Toshiyuki Arai

# 2023/12/30 Updated

# EfficientDetSavedModelCreator.py

r"""Tool to inspect a model."""
import os
# <added date="2021/0810"> arai
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# </added>
import sys
import shutil

import time
from typing import Text, Tuple, List

from absl import app
#from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

import traceback

from ConfigParser import *

from EfficientDetModelInspector import EfficientDetModelInspector

class EfficientDetSavedModelCreator(EfficientDetModelInspector):
  """A simple helper class for inspecting a model."""

  def __init__(self, config_file):
    super().__init__(config_file)
    print("=== EfficientDetSavedModelCreator")
    self.runmode         = self.parser.get(SAVED_MODEL, "runmode")
    self.saved_model_dir = self.parser.get(SAVED_MODEL, "saved_model_dir")

    if os.path.exists(self.model_dir) == False:
         message = "---- ckpt_model '" + self.model_dir + "' was not found -----"
         raise Exception (message)
    
    if os.path.exists(self.saved_model_dir):
        shutil.rmtree(self.saved_model_dir)
        print("=== removed saved_model_dir {}".format(self.saved_model_dir))
    if not os.path.exists(self.saved_model_dir):
        os.makedirs(self.saved_model_dir)


def main(_):
  config_file  = ""
  if len(sys.argv)==2:
    config_file      = sys.argv[1]
  else:
    raise Exception("Not found detect_config {}".format(config_file))
  
  if os.path.exists(config_file) == False:
    raise Exception("Not found train_config {}".format(config_file)) 
  
  creator = EfficientDetSavedModelCreator(config_file)
  #export_saved_model
  creator.run_model()


##
#
if __name__ == '__main__':
  
  logging.set_verbosity(logging.WARNING)
  if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
  app.run(main)
