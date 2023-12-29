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

# This code has been taken from main.py google/automl/efficientdet.

"""The main training script."""
import multiprocessing
import os
import platform

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import sys
from absl import app
#from absl import FLAGS
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import dataloader

#import det_model_fn
#2021/11/17
import det_model_fn2 as det_model_fn

import hparams_config
import utils

import pprint

#from io import StringIO 
from LabelMapReader          import LabelMapReader

from ConfigParser            import ConfigParser
from mAPEarlyStopping        import mAPEarlyStopping
from FvalueEarlyStopping     import FvalueEarlyStopping

from COCOMetricsWriter       import COCOMetricsWriter
from EpochChangeNotifier     import EpochChangeNotifier
from TrainingLossesWriter    import TrainingLossesWriter
from CategorizedAPWriter     import CategorizedAPWriter

from ConfigParser import *

class EfficientDetFinetuningModel(object):

  def __init__(self, config_file):
    self.TRAIN          = 'train'
    self.EVAL           = 'eval'
    self.TRAIN_AND_EVAL = 'train_and_eval'
    self.read_config_file(config_file)

    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)

    osname = platform.platform()
    print("=== Platform {}".format(osname))
    if "Windows" in osname:
      self.model_dir = self.model_dir.replace("/", "\\")
      print("--- Converted model_dir to:{}".format(self.model_dir))
      self.saved_model_dir = self.saved_model_dir.replace("/", "\\")
      print("--- Converted model_dir to:{}".format(self.saved_model_dir))

    if not os.path.exists(self.model_dir):
      raise Exception("FATAL ERROR: Not found model_dir {}".format(self.model_dir))

    if not os.path.exists(self.saved_model_dir):
      os.makedirs(self.saved_model_dir)

    print("=== eval_dir {}",format(self.eval_dir))
    if os.path.exists(self.eval_dir) == False:
      os.makedirs(self.eval_dir)
  
    labelMapReader          = LabelMapReader()
    self.label_map, classes = labelMapReader.read(self.label_map_pbtxt)
    print("=== label_map {}".format(self.label_map))

    self.training_losses_writer = TrainingLossesWriter(self.training_losses_file)    self.categorized_ap_writer  = CategorizedAPWriter(self.label_map_pbtxt, self.coco_ap_per_class_file)
        
    self.coco_metrics_writer    = COCOMetricsWriter(self.coco_metrics_file)

    self.early_stopping          = None

    if  self.patience > 0:
      if self.metrics   == "map":
        self.early_stopping = mAPEarlyStopping(patience=self.patience, verbose=1) 
      elif self.metrics == "fvalue":
        self.early_stopping = FvalueEarlyStopping(patience=self.patience, verbose=1)


    if self.strategy == 'tpu':
      tf.disable_eager_execution()
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          self.tpu, 
          zone    = self.tpu_zone, 
          project = self.gcp_project)
  
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)
    else:
      tpu_cluster_resolver = None

  def read_configfile(self, config_file):
    self.parser      = ConfigParser(config_file)

    #[project]
    self.name        = self.parser.get(PROJECT, "name") 
    self.owner       = self.parser.get(PROJECT, "owner")

    #[hardware]
    self.tpu         = self.parser.get(HARDWARE, "tpu") 
    self.tpu_zone    = self.parser.get(HARDWARE, "tpu_zone")
    self.gcp_project = self.parser.get(HARDWARE, "gcp_project")
    self.strategy    = self.parser.get(HARDWARE, "strategy")
    self.use_xla     = self.parser.get(HARDWARE, "use_xla")

    #[model]
    self.model_name    = self.parser.get(MODEL, "model_name") #"efficientdet-d0"
    self.classes_file  = self.parser.get(MODEL, "classes_file")
    self.model_dir     = self.parser.get(MODEL, "model_dir")
    self.ckpt          = self.parser.get(MODEL, "ckpt")  #../../efficientdet/efficientdet-d0"
    self.backbone_ckpt = self.parser.get(MODEL, "backbone_ckpt") #None
    self.tflite_path   = self.parser.get(MODEL, "tflite_path")
    self.tensorrt      = self.parser.get(MODEL, "tensorrt")
    self.export_ckpt   = self.parser.get(MODEL, "export_ckpt", dvalue=None)

    #[hyper_parameters]
    self.learning_rate= self.parser.get(HYPER_PARAMETERS,  "learning_rate", dvalue=0.08)
    self.num_classes  = self.parser.get(HYPER_PARAMETERS,  "num_classes")
    self.label_map    = self.parser.get(HYPER_PARAMETERS,  "label_map_yaml")
    self.augmentation = self.parser.get(HYPER_PARAMETERS,  "augmentation") #input_rand_hflip=False"
    self.image_size   = self.parser.get(HYPER_PARAMETERS,  "image_size")   #512x512"
    self.hparams      = "learning_rate=" + str(self.learning_rate) + "," \
                 + self.augmentation + \
                 ",image_size=" + self.image_size + \
                 ",num_classes=" + str(self.num_classes) + \
                 ",label_map=" + self.label_map 
    
    #[train]
    self.train_mode                = self.parser.get(TRAIN, "runmode")
    self.profile                   = self.parser.get(TRAIN, "profile")
    self.run_epoch_in_child_process= self.parser.get(TRAIN, "run_epoch_in_child_process")
    self.train_batch_size          = self.parser.get(TRAIN, "batch_size")
    self.epochs                    = self.parser.get(TRAIN, "epochs")
    self.save_checkpoints_steps    = self.parser.get(TRAIN, "save_checkpoints_steps")
    self.train_file_pattern        = self.parser.get(TRAIN, "file_pattern")
    self.label_map_pbtxt           = self.parser.get(TRAIN, "label_map_pbtxt")
    self.examples_per_epoch        = self.parser.get(TRAIN, "examples_per_epoch")
    self.cores                     = self.parser.get(TRAIN, "cores")
    self.use_spatial_partition     = self.parser.get(TRAIN, "use_spatial_partition")
    self.cores_per_replica         = self.parser.get(TRAIN, "cores_per_replica")
    self.input_partition_dims      = self.parser.get(TRAIN, "input_partition_dims")  
    self.tf_random_seed            = self.parser.get(TRAIN, "tf_random_seed")
    self.use_fake_data             = self.parser.get(TRAIN, "use_fake_data")
    self.training_losses_file      = self.parser.get(TRAIN, "training_losses_file")
    self.testdev_dir               = self.parser.get(TRAIN, "testdev_dir", dvalue=None)

    #[valid]
    self.val_name                = self.parser.get(VALID, "name")
    self.val_file_pattern        = self.parser.get(VALID, "file_pattern") 
    self.eval_dir                = self.parser.get(VALID, "eval_dir")
    self.val_batch_size          = self.parser.get(VALID, "batch_size")
    self.eval_samples            = self.parser.get(VALID, "eval_samples") 
    self.iterations_per_loop     = self.parser.get(VALID, "iterations_per_loop")

    self.val_json_file           = self.parser.get(VALID, "val_json_file")
    self.eval_after_train        = self.parser.get(VALID, "eval_after_train")
    self.min_eval_interval       = self.parser.get(VALID, "min_eval_internal") 
    self.timeout                 = self.parser.get(VALID, "timeout")
    self.coco_metrics_file       = self.parser.get(VALID, "coco_metrics_file")
    self.coco_ap_per_class_file  = self.parser.get(VALID, "coco_ap_per_class_file")
    self.disable_per_clsss_ap    = self.parser.get(VALID, "disable_per_clsss_ap")

    #[early_stopping]
    #;metrics     = "fvalue"
    self.metrics     = self.parser.get(EARLY_STOPPING, "metrics") 
    self.patience    = self.parser.get(EARLY_STOPPING, "patience")

    #[saved_model]
    #runmode        = "saved_model"
    self.saved_model_dir   = self.parser.get(SAVED_MODEL, "saved_model_dir")

    #[infer]
    self.infer_batch_size  = self.parser.get(INFER, "batch_size", dvalue=1)
    self.line_thickness    = self.parser.get(INFER, "line_thickness")
    self.max_boxes_to_draw = self.parser.get(INFER,  "max_boxes_to_draw")

    self.min_score_thresh  = self.parser.get(INFER, "min_score_thresh")  
    self.input_images      = self.parser.get(INFER, "input_images")       #"./realistic_test_dataset/*.jpg"
    self.ground_truth_json = self.parser.get(INFER, "ground_truth_json")  #"./realistic_test_dataset/annotation.json"
    self.output_images_dir = self.parser.get(INFER, "output_images_dir")  #"./realistic_test_dataset_outputs"
    #filters = [person, dog]
    self.filters           = self.parser.get(INFER, "filters")

    #[bench_mark]
    self.warmup_runs     = self.parser.get(BENCH_MARK "warmup_runs")
    self.bm_runs         = self.parser.get(BENCH_MARK "bm_runs")
    self.bm_threads      = self.parser.get(BENCH_MARK "bm_thread")
    self.trace_filename  = self.parser.get(BENCH_MARK "trace_filename")

  def train(self):
    # Check data path
    if self.train_mode in ('train', 'train_and_eval'):
      if self.train_file_pattern is None:
        raise RuntimeError('Must specify --train_file_pattern for train.')
    if self.train_mode in ('eval', 'train_and_eval'):
      if self.val_file_pattern is None:
        raise RuntimeError('Must specify --val_file_pattern for eval.')

    # Parse and override hparams
    config = hparams_config.get_detection_config(self.model_name )
    
    
    if self.hparams:
      config.override(self.hparams)
    if self.epochs:  # NOTE: remove this flag after updating all docs.
      config.num_epochs = self.epochs
    print("--- config {}".format(config))
    # Parse image size in case it is in string format.
    print("----- config.image_size()".format(config.image_size) )
    config.image_size = utils.parse_image_size(config.image_size)

    # The following is for spatial partitioning. `features` has one tensor while
    # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
    # partition is performed on `features` and all partitionable tensors of
    # `labels`, see the partition logic below.
    # In the TPUEstimator context, the meaning of `shard` and `replica` is the
    # same; follwing the API, here has mixed use of both.

    if self.use_spatial_partition:
      # Checks input_partition_dims agrees with num_cores_per_replica.
      if self.cores_per_replica != np.prod(self.input_partition_dims):
        raise RuntimeError('--num_cores_per_replica must be a product of array'
                           'elements in --input_partition_dims.')

      labels_partition_dims = {
          'mean_num_positives': None,
          'source_ids':         None,
          'groundtruth_data':   None,
          'image_scales':       None,
          'image_masks':        None,
      }
      # The Input Partition Logic: We partition only the partition-able tensors.
      feat_sizes = utils.get_feat_sizes(
          config.get('image_size'), config.get('max_level'))
      for level in range(config.get('min_level'), config.get('max_level') + 1):

        def _can_partition(spatial_dim):
          partitionable_index = np.where(
              spatial_dim % np.array(self.input_partition_dims ) == 0)
          return len(partitionable_index[0]) == len(self.input_partition_dims )

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
      input_partition_dims = [input_partition_dims, labels_partition_dims]
      num_shards = self.cores // self.cores_per_replica
    else:
      num_cores_per_replica = None
      input_partition_dims = None
      num_shards = self.cores

    params = dict(
        config.as_dict(),
        model_name = self.model_name,
        iterations_per_loop = self.iterations_per_loop,
        model_dir  = self.model_dir,
        num_shards = num_shards,
        num_examples_per_epoch = self.examples_per_epoch,
        strategy = self.strategy,
        backbone_ckpt = self.backbone_ckpt,
        ckpt = ckpt,
        val_json_file = self.val_json_file,
        testdev_dir = self.testdev_dir,
        profile = self.profile,
        mode =self.train_mode)
    
    config_proto = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    if self.strategy != 'tpu':
      if self.use_xla:
        config_proto.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
        config_proto.gpu_options.allow_growth = True

    model_fn_instance = det_model_fn.get_model_fn(self.model_name )
    max_instances_per_image = config.max_instances_per_image
    if self.eval_samples:
      eval_steps = int(self.eval_samples + self.val_batch_size - 1) // self.val_batch_size 
    else:
      eval_steps = None
    total_examples = int(self.epochs * self.examples_per_epoch)
    train_steps = total_examples // self.train_batch_size
    logging.info(params)

    if not tf.io.gfile.exists(self.model_dir):
      tf.io.gfile.makedirs(self.model_dir)

    config_file = os.path.join(self.model_dir, 'config.yaml')
    if not tf.io.gfile.exists(config_file):
      tf.io.gfile.GFile(config_file, 'w').write(str(config))

    train_input_fn = dataloader.InputReader(
        self.train_file_pattern,
        is_training  = True,
        use_fake_data= self.use_fake_data,
        max_instances_per_image=max_instances_per_image)
    eval_input_fn = dataloader.InputReader(
        self.val_file_pattern,
        is_training  = False,
        use_fake_data= self.use_fake_data,
        max_instances_per_image=max_instances_per_image)

    if self.strategy == 'tpu':
      tpu_config = tf.estimator.tpu.TPUConfig(
          self.iterations_per_loop if self.strategy == 'tpu' else 1,
          num_cores_per_replica = self.cores_per_replica,
          input_partition_dims  = input_partition_dims,
          per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
          .PER_HOST_V2)
      tpu_cluster_resolver = None
      run_config = tf.estimator.tpu.RunConfig(
          cluster  = tpu_cluster_resolver,
          model_dir= self.model_dir,
          log_step_count_steps = self.iterations_per_loop,
          session_config = config_proto,
          tpu_config = tpu_config,
          save_checkpoints_steps = self.save_checkpoints_steps,
          tf_random_seed = self.tf_random_seed,
      )
      # TPUEstimator can do both train and eval.
      train_est = tf.estimator.tpu.TPUEstimator(
          model_fn=model_fn_instance,
          train_batch_size = self.train_batch_size,
          eval_batch_size  = self.val_batch_size,
          config = run_config,
          params = params)
      eval_est = train_est
    else:
      strategy = None
      if self.strategy == 'gpus':
        strategy = tf.distribute.MirroredStrategy()
      run_config = tf.estimator.RunConfig(
          model_dir = self.model_dir,
          train_distribute = strategy,
          log_step_count_steps = self.iterations_per_loop,
          session_config = config_proto,
          save_checkpoints_steps = self.save_checkpoints_steps,
          tf_random_seed = self.tf_random_seed,
      )

      def get_estimator(global_batch_size):
        params['num_shards'] = getattr(strategy, 'num_replicas_in_sync', 1)
        params['batch_size'] = global_batch_size // params['num_shards']
        params['eval_dir']   = self.eval_dir()   
        params['label_map']  = self.label_map    
        params['disable_per_class_ap'] = self.disable_per_clsss_ap 
        print("---disable_per_class_ap {}".format(params['disable_per_class_ap']))
        return tf.estimator.Estimator(
            model_fn = model_fn_instance, 
            config = run_config, 
            params = params)

      # train and eval need different estimator due to different batch size.
      train_est = get_estimator(self.train_batch_size)
      eval_est  = get_estimator(self.val_batch_size)

    # start train/eval flow.
    if self.train_mode == 'train':
      train_est.train(input_fn = train_input_fn, max_steps = train_steps)
      if self.eval_after_train:
        eval_est.evaluate(input_fn = eval_input_fn, steps = eval_steps)

    elif self.train_mode == 'eval':
      # Run evaluation when there's a new checkpoint
      for ckpt in tf.train.checkpoints_iterator(
          self.model_dir,
          min_interval_secs = self.min_eval_interval,
          timeout           = self.timeout ):

        logging.info('Starting to evaluate.')
        try:
          eval_results = eval_est.evaluate(
              eval_input_fn, 
              steps = eval_steps, 
              name  = self.val_name )
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

    elif self.train_mode == 'train_and_eval':
      ckpt = tf.train.latest_checkpoint(self.model_dir)
      try:
        step = int(os.path.basename(ckpt).split('-')[1])
        current_epoch = (
            step * self.train_batch_size // self.examples_per_epoch)
        logging.info('found ckpt at step %d (epoch %d)', step, current_epoch)
      except (IndexError, TypeError):
        logging.info('Folder %s has no ckpt with valid step.', self.model_dir )
        current_epoch = 0

      def run_train_and_eval(e):
        print('\n   =====> Starting run_train_and_eval, epoch: %d.' % e)
        # 2021/11/15 
        os.environ['epoch'] = str(e)

        train_est.train(
            input_fn  = train_input_fn,
            max_steps = e * self.examples_per_epoch // self.train_batch_size )
        print('\n   =====> Starting evaluation, epoch: %d.' % e)
        eval_results = eval_est.evaluate(input_fn = eval_input_fn, steps = eval_steps)
        

        ckpt = tf.train.latest_checkpoint(self.model_dir )
        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
        map  = eval_results['AP']
        loss = eval_results['loss']

        #self.epoch_change_notifier.epoch_end(e, loss, map)
        
        self.coco_metrics_writer.write(e, eval_results)
        self.categorized_ap_writer.write(e, eval_results)
        self.training_losses_writer.write(e, eval_results)

        ckpt = tf.train.latest_checkpoint(self.model_dir )
        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)

        earlystopping = False
        if self.early_stopping != None:
          ap = eval_results['AP']
          ar = eval_results['ARmax1']
          earlystopping = self.early_stopping.validate(e, ap, ar)
          
        return earlystopping
      
      epochs_per_cycle = 1  # higher number has less graph construction overhead.
      for e in range(current_epoch + 1, config.num_epochs + 1, epochs_per_cycle):
        if self.run_epoch_in_child_process:
          p = multiprocessing.Process(target=run_train_and_eval, args=(e,))
          p.start()
          p.join()
          if p.exitcode != 0:
            return p.exitcode
        else:
          tf.reset_default_graph()
          earlystopping = run_train_and_eval(e)
          if earlystopping:
            print("=== Early_stopping validated at epoch {}".format(e))
            break
            
    else:
      logging.info('Invalid mode: %s', self.train_mode)

def main(_):
  config_file = ""
  if len(sys.argv)==2:
    config_file = sys.argv[1]
  else:
    raise Exception("Usage: python EfficientDetFinetuningModel.py config_file")

  if os.path.exists(config_file) == False:
    raise Exception("Not found train_config {}".format(config_file)) 
  
  model = EfficientDetFinetuningModel(config_file)
  model.train()
       

if __name__ == '__main__':
  app.run(main)
