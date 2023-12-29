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

# 2022/06/01 Copyright (C) antillia.com
# This code has been taken from main.py in google/automl/efficientdet.

# 2023/12/15 Modified to use train_eval_infer.config

"""The main training script."""
import multiprocessing
import os
import platform
import sys
from tkinter import E
#2022/05/12
sys.path.append("../../")

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from absl import app
from absl import flags
from absl import logging
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf

#Tensorflow2.0
from tensorflow.python.framework import ops

import dataloader

# 2021/11/20
#import det_model_fn
import det_model_fn2 as det_model_fn

import hparams_config
import utils

# <added date="2021/11/20"> atlan-antillia
from LabelMapReader          import LabelMapReader
from mAPEarlyStopping        import mAPEarlyStopping
from FvalueEarlyStopping     import FvalueEarlyStopping
from COCOMetricsWriter       import COCOMetricsWriter
from TrainingLossesWriter    import TrainingLossesWriter
from CategorizedAPWriter     import CategorizedAPWriter
# </added>

from ConfigParser import ConfigParser

PROJECT  = "project"
HARDWARE = "hardware"
MODEL    = "model"
TRAIN    = "train"
VALID    = "valid"
HYPER_PARAMETERS= "hyper_parameters"
EARLY_STOPPING = "early_stopping"


def main(_):
  train_eval_infer_config = ""
  if len(sys.argv)==2:
    train_eval_infer_config = sys.argv[1]
  else:
    raise Exception("Usage: python ModelTrainer.py train_eval_infer.config")

  if os.path.exists(train_eval_infer_config) == False:
    raise Exception("Not found train_config {}".format(train_eval_infer_config)) 
  
  parser = ConfigParser(train_eval_infer_config)
  parser.dump_all()
  dataset_dir = parser.get(PROJECT, "dataset_dir")
  print("--- dataset_dir {}".format(dataset_dir))
  strategy = parser.get(HARDWARE, "strategy")
  if strategy == 'tpu':
    tpu = parser.get(HARDWARE, "tpu")
    tpu_zone    = parser.get(HARDWARE, "tpu_zone")
    gcp_project = parser.get(HARDWARE, "gcp_project")
    tf.disable_eager_execution()
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu, zone=tpu_zone, project=gcp_project)
    tpu_grpc_url = tpu_cluster_resolver.get_master()
    tf.Session.reset(tpu_grpc_url)
  else:
    tpu_cluster_resolver = None

  model_dir   = parser.get(MODEL, "model_dir")
  org_model_dir = model_dir
  osname = platform.platform()
  print("=== Platform {}".format(osname))
  if "Windows" in osname:
    model_dir = model_dir.replace("/", "\\")
    print("--- Converted model_dir from:{} to:{}".format(org_model_dir, model_dir))

  # Check data path
  mode = parser.get(TRAIN, "mode")
  print("dataset_dir {}".format(dataset_dir))
  file_pattern = parser.get(TRAIN, "file_pattern")
  print("file_pattern {}".format(file_pattern))

  train_file_pattern= os.path.join(dataset_dir, parser.get(TRAIN, "file_pattern"))
  valid_file_pattern= os.path.join(dataset_dir, parser.get(VALID, "file_pattern"))

  if mode in ('train', 'train_and_eval'):
    if train_file_pattern is None:
      raise RuntimeError('Must specify --train_file_pattern for train.')
  if mode in ('eval', 'train_and_eval'):
    if valid_file_pattern is None:
      raise RuntimeError('Must specify --val_file_pattern for eval.')

  # Parse and override hparams
  config = hparams_config.get_detection_config(self.model_name)


  # augmentation = "input_rand_hflip=False"
  # augmentation = "aug_policy=v1"

  hparams      = "learning_rate=" + str(self.learning_rate) + "," \
                 + self.augmentation + \
                 ",image_size=" + self.image_size + \
                 ",num_classes=" + str(self.num_classes) + \
                 ",label_map=" + self.label_map_yaml 
  
  #hparams="input_rand_hflip=False,image_size=512x512,num_classes=78,label_map=./label_map.yaml
  print("---- hparams {}".format(hparams))

  num_epochs = parser.get(TRAIN, "epochs")
  config.override(hparams)
  if num_epochs:  # NOTE: remove this flag after updating all docs.
    config.num_epochs = num_epochs

  eval_dir = parser.get(VALID, "eval_dir")
  if os.path.exists(eval_dir) ==False:
    os.makedirs(eval_dir)

  label_map_pbtxt = parser.get(TRAIN, "label_map_pbtxt") 

  if os.path.exists(label_map_pbtxt) ==False:
    raise Exception("Not found " + label_map_pbtxt)

  labelMapReader          = LabelMapReader()
  label_map, classes      = labelMapReader.read(label_map_pbtxt)
  print("=== label_map {}".format(label_map))

  coco_metrics_csv_path   = os.path.join(eval_dir,  "coco_metrics.csv")
  coco_metrics_writer     = COCOMetricsWriter(coco_metrics_csv_path)
  
  train_losses_csv_path   = os.path.join(eval_dir, "train_losses.csv")
  train_losses_writer     = TrainingLossesWriter(train_losses_csv_path)
  
  categorized_ap_csv_path = os.path.join(eval_dir, "coco_ap_per_class.csv")
  categorized_ap_writer   = CategorizedAPWriter(label_map_pbtxt, categorized_ap_csv_path)
  
  metrics = parser.get(EARLY_STOPPING, "metrics")
  patience   = parser.get(EARLY_STOPPING, "patience")

  if patience > 0:
    early_stopping = None
    if metrics == "map":
      early_stopping = mAPEarlyStopping(patience)
    elif metrics == "fvalue":
      early_stopping = FvalueEarlyStopping(patience)
    else:
      print("Invalid FLAGS.early_stopping {}".format(early_stopping))
    
  # Parse image size in case it is in string format.
  print("---- config.image_size {}".format(config.image_size))
  
  config.image_size = utils.parse_image_size(config.image_size)
  print("=== config \n{}".format(config))
  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # In the TPUEstimator context, the meaning of `shard` and `replica` is the
  # same; follwing the API, here has mixed use of both.
  use_spatial_partition = parser.get(TRAIN, "use_spatial_partition")
  num_cores             = parser.get(TRAIN, "cores")
  if use_spatial_partition:
    # Checks input_partition_dims agrees with num_cores_per_replica.
    num_cores_per_replica = parser.get(TRAIN, "cores_per_replica")
    input_partition_dims  = parser.get(TRAIN, "input_partition_dims")

    if num_cores_per_replica != np.prod(input_partition_dims):
      raise RuntimeError('--num_cores_per_replica must be a product of array'
                         'elements in --input_partition_dims.')

    labels_partition_dims = {
        'mean_num_positives': None,
        'source_ids'        : None,
        'groundtruth_data'  : None,
        'image_scales'      : None,
        'image_masks'       : None,
    }
    # The Input Partition Logic: We partition only the partition-able tensors.
    feat_sizes = utils.get_feat_sizes(
        config.get('image_size'), config.get('max_level'))
    for level in range(config.get('min_level'), config.get('max_level') + 1):

      def _can_partition(spatial_dim):
        partitionable_index = np.where(
            spatial_dim % np.array(input_partition_dims) == 0)
        return len(partitionable_index[0]) == len(input_partition_dims)

      spatial_dim = feat_sizes[level]
      if _can_partition(spatial_dim['height']) and _can_partition(
          spatial_dim['width']):
        labels_partition_dims['box_targets_%d' %
                              level] = input_partition_dims
        labels_partition_dims['cls_targets_%d' %
                              level] = input_partition_dims
      else:
        labels_partition_dims['box_targets_%d' % level] = None
        labels_partition_dims['cls_targets_%d' % level] = None
    num_cores_per_replica = num_cores_per_replica
    input_partition_dims = [input_partition_dims, labels_partition_dims]
    num_shards = num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims  = None
    num_shards            = num_cores

  iterations_per_loop = parser.get(VALID, "iterations_per_loop")  
  ckpt                = parser.get(MODEL, "ckpt")
  backbone_ckpt       = parser.get(MODEL, "backbone_ckpt")
  profile             = parser.get(MODEL, "profile")
  num_examples_per_epoch = parser.get(TRAIN, "examples_per_epoch")
  val_json_file          = parser.get(VALID, "val_json_file")
  testdev_dir            = parser.get(VALID, "testdev_dir", dvalue=None)
  params = dict(
      config.as_dict(),
      model_name             = model_name,
      iterations_per_loop    = iterations_per_loop,
      model_dir              = model_dir,
      num_shards             = num_shards,
      num_examples_per_epoch = num_examples_per_epoch,
      strategy               = strategy,
      backbone_ckpt          = backbone_ckpt,
      ckpt                   = ckpt,
      val_json_file          = val_json_file,
      testdev_dir            = testdev_dir,
      profile                = profile,
      mode                   = mode)
  #
  config_proto = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
  if strategy != 'tpu':
    use_xla = parser.get(HARDWARE, "use_xla")
    if use_xla:
      config_proto.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_1)
      config_proto.gpu_options.allow_growth = True

  model_dir               = model_dir
  model_fn_instance       = det_model_fn.get_model_fn(model_name)
  max_instances_per_image = config.max_instances_per_image
  eval_batch_size = parser.get(VALID, "batch_size")
  eval_samples = parser.get(VALID, "eval_samples")

  if eval_samples:
    eval_steps = int((eval_samples + eval_batch_size - 1) //
                     eval_batch_size)
  else:
    eval_steps = None
    
  train_batch_size = parser.get(TRAIN, "batch_size")
  total_examples = int(num_epochs * num_examples_per_epoch)
  train_steps    = total_examples // train_batch_size
  logging.info(params)

  if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)

  config_file = os.path.join(model_dir, 'config.yaml')
  if not tf.io.gfile.exists(config_file):
    tf.io.gfile.GFile(config_file, 'w').write(str(config))

  use_fake_data = parser.get(TRAIN, "use_fake_data")

  train_input_fn = dataloader.InputReader(
      train_file_pattern,
      is_training    = True,
      use_fake_data  = use_fake_data,
      max_instances_per_image = max_instances_per_image)
      
  eval_input_fn = dataloader.InputReader(
      valid_file_pattern,
      is_training       = False,
      use_fake_data     = use_fake_data,
      max_instances_per_image = max_instances_per_image)

  save_checkpoints_steps = parser.get(TRAIN, "save_checkpoints_steps")
  tf_random_seed         = parser.get(TRAIN, "tf_random_seed")
  if strategy == 'tpu':
    iterations_per_loop = parser.get(VALID, "iterations_per_loop")
    tpu_config = tf.estimator.tpu.TPUConfig(
        iterations_per_loop if strategy == 'tpu' else 1,
        num_cores_per_replica = num_cores_per_replica,
        input_partition_dims  = input_partition_dims,
        per_host_input_for_training = tf.estimator.tpu.InputPipelineConfig
        .PER_HOST_V2)
    run_config = tf.estimator.tpu.RunConfig(
        cluster    = tpu_cluster_resolver,
        model_dir  = model_dir,
        log_step_count_steps = iterations_per_loop,
        session_config = config_proto,
        tpu_config     = tpu_config,
        save_checkpoints_steps = save_checkpoints_steps,
        tf_random_seed = tf_random_seed,
    )
    # TPUEstimator can do both train and eval.
    train_est = tf.estimator.tpu.TPUEstimator(
        model_fn=model_fn_instance,
        train_batch_size = train_batch_size,
        eval_batch_size  = eval_batch_size,
        config           = run_config,
        params           = params)
    eval_est = train_est
  else:
    strategy = None
    if strategy == 'gpus':
        strategy = tf.distribute.MirroredStrategy()
    run_config       = tf.estimator.RunConfig(
        model_dir        = model_dir,
        train_distribute = strategy,
        log_step_count_steps = iterations_per_loop,
        session_config         = config_proto,
        save_checkpoints_steps = save_checkpoints_steps,
        tf_random_seed         = tf_random_seed,
    )

    def get_estimator(global_batch_size):
      params['num_shards'] = getattr(strategy, 'num_replicas_in_sync', 1)
      params['batch_size'] = global_batch_size // params['num_shards']
      params['eval_dir']   = eval_dir     #2021/11/14
      params['label_map']  = label_map          #2021/11/14
      params['disable_per_class_ap'] = False    #2021/11/15
      return tf.estimator.Estimator(
          model_fn = model_fn_instance, 
          config   = run_config, params=params)

    # train and eval need different estimator due to different batch size.
    train_est = get_estimator(train_batch_size)
    eval_est  = get_estimator(eval_batch_size)

  # start train/eval flow.
  eval_after_train = parser.get(TRAIN, "eval_after_train")
  if mode == 'train':
    train_est.train(input_fn=train_input_fn, max_steps=train_steps)
    if eval_after_train:
      eval_est.evaluate(input_fn = eval_input_fn, steps = eval_steps)

  elif mode == 'eval':
    min_eval_interval = parser.get(VALID, "min_eval_interval") 
    eval_timeout      = parser.get(VALID, "eval_timeout")
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        model_dir,
        min_interval_secs = min_eval_interval,
        timeout           = eval_timeout):

      logging.info('Starting to evaluate.')
      try:
        eval_results = eval_est.evaluate(
            eval_input_fn, steps = eval_steps, name = FLAGS.eval_name)
        # Terminate eval job when final checkpoint is reached.
        try:
          current_step = int(os.path.basename(ckpt).split('-')[1])
        except IndexError:
          logging.info('%s has no global step info: stop!', ckpt)
          break

        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
        if current_step >= train_steps:
          logging.info('Eval finished step %d/%d', current_step, train_steps)
          break

      except tf.errors.NotFoundError:
        # Checkpoint might be not already deleted by the time eval finished.
        # We simply skip ssuch case.
        logging.info('Checkpoint %s no longer exists, skipping.', ckpt)

  elif mode == 'train_and_eval':
    ckpt = tf.train.latest_checkpoint(model_dir)
    try:
      step = int(os.path.basename(ckpt).split('-')[1])
      current_epoch = (
          step * train_batch_size // num_examples_per_epoch)
      logging.info('found ckpt at step %d (epoch %d)', step, current_epoch)
    except (IndexError, TypeError):
      logging.info('Folder %s has no ckpt with valid step.', model_dir)
      current_epoch = 0

    def run_train_and_eval(e):
      print('\n   =====> Starting training, epoch: %d.' % e)
      # 2021/11/20
      os.environ['epoch'] = str(e)
      train_est.train(
          input_fn=train_input_fn,
          max_steps=e * num_examples_per_epoch // train_batch_size)
      print('\n   =====> Starting evaluation, epoch: %d.' % e)
      eval_results = eval_est.evaluate(input_fn=eval_input_fn, steps=eval_steps)

      ckpt = tf.train.latest_checkpoint(model_dir)
      utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)

      coco_metrics_writer.write(e, eval_results)
      train_losses_writer.write(e, eval_results)
      if "label_map" in hparams:      
        categorized_ap_writer.write(e, eval_results)
      
      early_stop = False
      if early_stopping != None:
        ap = eval_results['AP']
        ar = eval_results['ARmax1']
      
        early_stop = early_stopping.validate(e, ap, ar)
      return early_stop

    epochs_per_cycle = 1  # higher number has less graph construction overhead.
    run_epoch_in_child_process = parser.get(TRAIN, "run_epoch_in_child_process")
    for e in range(current_epoch + 1, config.num_epochs + 1, epochs_per_cycle):
      if run_epoch_in_child_process:
        p = multiprocessing.Process(target=run_train_and_eval, args=(e,))
        p.start()
        p.join()
        if p.exitcode != 0:
          return p.exitcode
      else:
        tf.reset_default_graph()
        early_stop = run_train_and_eval(e)
        if early_stop:
          print("==== EarlyStopping validated: break training loop.") 
          break
  else:
    logging.info('Invalid mode: %s', FLAGS.mode)


if __name__ == '__main__':
  app.run(main)
