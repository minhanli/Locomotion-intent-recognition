from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.applications.mobilenet import MobileNet,preprocess_input

import collections
import copy
import csv
import io
import json
import os
import re
import time

import numpy as np
import six

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.saving import checkpoint_options as checkpoint_options_lib
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
try:
  import requests
except ImportError:
  requests = None

class WeightAnnealing(tfk.callbacks.Callback):

    def __init__(self, total_iteration=1.0e5,num_cycle=8.0,ratio=0.5,beta=1):
        super(WeightAnnealing, self).__init__()
        self.total_iteration = tf.convert_to_tensor(total_iteration, tf.float32)
        self.num_cycle = tf.convert_to_tensor(num_cycle, tf.float32)
        self.ratio = tf.convert_to_tensor(ratio, tf.float32)
        self.weight = tf.convert_to_tensor(0.0, tf.float32)
        self.cur_iteration = tf.convert_to_tensor(0.0, tf.float32)
        self.beta = beta

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model, "kl_weight"):
            raise ValueError('Model must have a "kl_weight" attribute.')

        t_m = tf.math.divide(self.total_iteration,self.num_cycle)
        int_t_m = tf.math.floor(t_m)
        tau = tf.math.floormod(self.cur_iteration,int_t_m)/t_m
        if tau>self.ratio:
            self.weight  = self.beta*tf.convert_to_tensor(1.,tf.float32)
        else:
            self.weight  = self.beta*tf.math.divide(1,self.ratio)*tau

        self.model.kl_weight.assign(self.weight,use_locking=True)

        self.cur_iteration = self.cur_iteration+1

        # tf.print("\nBatch %06d: Learning rate is %6.4f." % (batch, self.weight))

class TrackMetrics(tfk.metrics.Metric):
    def __init__(self, name="track_metric", **kwargs):
        super(TrackMetrics, self).__init__(name=name, **kwargs)
        self.track_metric = self.add_weight(name="met", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

      self.track_metric.assign(tf.reduce_mean(y_pred))

    def result(self):

        return self.track_metric

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.track_metric.assign(0.0)

def prediction_loss(one_hot_label,prediction_logit):
    if tf.shape(prediction_logit)[-1]==5:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                       (labels=one_hot_label, logits=prediction_logit))
    else:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                       (labels=one_hot_label, logits=prediction_logit[:,:5]))
        cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                            (labels=one_hot_label, logits=prediction_logit[:, 5:10]))
        cross_entropy += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                            (labels=one_hot_label, logits=prediction_logit[:, 10:]))
    return cross_entropy

def build_pretrained_model(pretrained_layers=(85,),tunable=False):
    input_tensor = tf.keras.Input((224,224,3))
    MN_model = MobileNet(include_top=False, pooling=None, input_tensor=input_tensor)
    fea_outputs=[]
    for num, layer in enumerate(MN_model.layers):
        if num in pretrained_layers:
            fea_outputs.append(layer.output)
        if tunable and layer.name[-2:]!='bn':
            layer.trainable = True
        else:
            layer.trainable = False
    pretrained_model = tf.keras.Model(inputs=input_tensor, outputs=fea_outputs)
    return pretrained_model

################################################################################################################
def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        epochs=None,
                        steps_per_epoch=None,
                        samples=None,
                        verbose=1,
                        count_mode='steps',
                        mode=ModeKeys.TRAIN):
  """Configures callbacks for use in various training loops.
  Arguments:
      callbacks: List of Callbacks.
      model: Model being trained.
      do_validation: Whether or not validation loop will be run.
      batch_size: Number of samples per batch.
      epochs: Number of epoch to train.
      steps_per_epoch: Number of batches to run per training epoch.
      samples: Number of training samples.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
      count_mode: One of 'steps' or 'samples'. Per-batch or per-sample count.
      mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
        Which loop mode to configure callbacks for.
  Returns:
      Instance of CallbackList used to control all Callbacks.
  """
  # Check if callbacks have already been configured.
  if isinstance(callbacks, CallbackList):
    return callbacks

  if not callbacks:
    callbacks = []

  # Add additional callbacks during training.
  if mode == ModeKeys.TRAIN:
    model.history = History()
    callbacks = [BaseLogger()] + (callbacks or []) + [model.history]
    if verbose:
      callbacks.append(ProgbarLogger(count_mode))
  callback_list = CallbackList(callbacks)

  # Set callback model
  callback_model = model._get_callback_model()  # pylint: disable=protected-access
  callback_list.set_model(callback_model)

  set_callback_parameters(
      callback_list,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      samples=samples,
      verbose=verbose,
      mode=mode)

  callback_list.model.stop_training = False
  return callback_list

def set_callback_parameters(callback_list,
                            model,
                            do_validation=False,
                            batch_size=None,
                            epochs=None,
                            steps_per_epoch=None,
                            samples=None,
                            verbose=1,
                            mode=ModeKeys.TRAIN):
  """Sets callback parameters.
  Arguments:
      callback_list: CallbackList instance.
      model: Model being trained.
      do_validation: Whether or not validation loop will be run.
      batch_size: Number of samples per batch.
      epochs: Number of epoch to train.
      steps_per_epoch: Number of batches to run per training epoch.
      samples: Number of training samples.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
      mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
        Which loop mode to configure callbacks for.
  """
  metric_names = model.metrics_names
  for cbk in callback_list:
    if isinstance(cbk, (BaseLogger, ProgbarLogger)):
      cbk.stateful_metrics = metric_names[1:]  # Exclude `loss`

  # Set callback parameters
  callback_metrics = []
  # When we have deferred build scenario with iterator input, we will compile
  # when we standardize first batch of data.
  if mode != ModeKeys.PREDICT:
    callback_metrics = copy.copy(metric_names)
    if do_validation:
      callback_metrics += ['val_' + n for n in metric_names]
  callback_params = {
      'batch_size': batch_size,
      'epochs': epochs,
      'steps': steps_per_epoch,
      'samples': samples,
      'verbose': verbose,
      'do_validation': do_validation,
      'metrics': callback_metrics,
  }
  callback_list.set_params(callback_params)

def _is_generator_like(data):
  """Checks if data is a generator, Sequence, or Iterator."""
  return (hasattr(data, '__next__') or hasattr(data, 'next') or isinstance(
      data, (Sequence, iterator_ops.Iterator, iterator_ops.OwnedIterator)))

def make_logs(model, logs, outputs, mode, prefix=''):
  """Computes logs for sending to `on_batch_end` methods."""
  metric_names = model.metrics_names
  if mode in {ModeKeys.TRAIN, ModeKeys.TEST} and metric_names:
    for label, output in zip(metric_names, outputs):
      logs[prefix + label] = output
  else:
    logs['outputs'] = outputs
  return logs

@keras_export('keras.callbacks.Callback')
class Callback(object):
  """Abstract base class used to build new callbacks.
  Attributes:
      params: Dict. Training parameters
          (eg. verbosity, batch size, number of epochs...).
      model: Instance of `keras.models.Model`.
          Reference of the model being trained.
  The `logs` dictionary that callback methods
  take as argument will contain keys for quantities relevant to
  the current batch or epoch (see method-specific docstrings).
  """

  def __init__(self):
    self.validation_data = None  # pylint: disable=g-missing-from-attributes
    self.model = None
    # Whether this Callback should only run on the chief worker in a
    # Multi-Worker setting.
    # TODO(omalleyt): Make this attr public once solution is stable.
    self._chief_worker_only = None
    self._supports_tf_logs = False

  def set_params(self, params):
    self.params = params

  def set_model(self, model):
    self.model = model

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_batch_begin(self, batch, logs=None):
    """A backwards compatibility alias for `on_train_batch_begin`."""

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_batch_end(self, batch, logs=None):
    """A backwards compatibility alias for `on_train_batch_end`."""

  @doc_controls.for_subclass_implementers
  def on_epoch_begin(self, epoch, logs=None):
    """Called at the start of an epoch.
    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.
    Arguments:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_epoch_end(self, epoch, logs=None):
    """Called at the end of an epoch.
    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.
    Arguments:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`. For training epoch, the values of the
         `Model`'s metrics are returned. Example : `{'loss': 0.2, 'acc': 0.7}`.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_train_batch_begin(self, batch, logs=None):
    """Called at the beginning of a training batch in `fit` methods.
    Subclasses should override for any actions to run.
    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.
    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.train_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    # For backwards compatibility.
    self.on_batch_begin(batch, logs=logs)

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_train_batch_end(self, batch, logs=None):
    """Called at the end of a training batch in `fit` methods.
    Subclasses should override for any actions to run.
    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.
    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    # For backwards compatibility.
    self.on_batch_end(batch, logs=logs)

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_test_batch_begin(self, batch, logs=None):
    """Called at the beginning of a batch in `evaluate` methods.
    Also called at the beginning of a validation batch in the `fit`
    methods, if validation data is provided.
    Subclasses should override for any actions to run.
    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.
    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.test_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_test_batch_end(self, batch, logs=None):
    """Called at the end of a batch in `evaluate` methods.
    Also called at the end of a validation batch in the `fit`
    methods, if validation data is provided.
    Subclasses should override for any actions to run.
    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.
    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_predict_batch_begin(self, batch, logs=None):
    """Called at the beginning of a batch in `predict` methods.
    Subclasses should override for any actions to run.
    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.
    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.predict_step`,
          it typically returns a dict with a key 'outputs' containing
          the model's outputs.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_predict_batch_end(self, batch, logs=None):
    """Called at the end of a batch in `predict` methods.
    Subclasses should override for any actions to run.
    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.
    Arguments:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

  @doc_controls.for_subclass_implementers
  def on_train_begin(self, logs=None):
    """Called at the beginning of training.
    Subclasses should override for any actions to run.
    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_train_end(self, logs=None):
    """Called at the end of training.
    Subclasses should override for any actions to run.
    Arguments:
        logs: Dict. Currently the output of the last call to `on_epoch_end()`
          is passed to this argument for this method but that may change in
          the future.
    """

  @doc_controls.for_subclass_implementers
  def on_test_begin(self, logs=None):
    """Called at the beginning of evaluation or validation.
    Subclasses should override for any actions to run.
    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_test_end(self, logs=None):
    """Called at the end of evaluation or validation.
    Subclasses should override for any actions to run.
    Arguments:
        logs: Dict. Currently the output of the last call to
          `on_test_batch_end()` is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_predict_begin(self, logs=None):
    """Called at the beginning of prediction.
    Subclasses should override for any actions to run.
    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_predict_end(self, logs=None):
    """Called at the end of prediction.
    Subclasses should override for any actions to run.
    Arguments:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  def _implements_train_batch_hooks(self):
    """Determines if this Callback should be called for each train batch."""
    return (not generic_utils.is_default(self.on_batch_begin) or
            not generic_utils.is_default(self.on_batch_end) or
            not generic_utils.is_default(self.on_train_batch_begin) or
            not generic_utils.is_default(self.on_train_batch_end))

  def _implements_test_batch_hooks(self):
    """Determines if this Callback should be called for each test batch."""
    return (not generic_utils.is_default(self.on_test_batch_begin) or
            not generic_utils.is_default(self.on_test_batch_end))

  def _implements_predict_batch_hooks(self):
    """Determines if this Callback should be called for each predict batch."""
    return (not generic_utils.is_default(self.on_predict_batch_begin) or
            not generic_utils.is_default(self.on_predict_batch_end))

@keras_export('keras.callbacks.ModelCheckpoint')
class CustomizedModelCheckpoint(Callback):
  """Callback to save the Keras model or model weights at some frequency.
  `ModelCheckpoint` callback is used in conjunction with training using
  `model.fit()` to save a model or weights (in a checkpoint file) at some
  interval, so the model or weights can be loaded later to continue the training
  from the state saved.
  A few options this callback provides include:
  - Whether to only keep the model that has achieved the "best performance" so
    far, or whether to save the model at the end of every epoch regardless of
    performance.
  - Definition of 'best'; which quantity to monitor and whether it should be
    maximized or minimized.
  - The frequency it should save at. Currently, the callback supports saving at
    the end of every epoch, or after a fixed number of training batches.
  - Whether only weights are saved, or the whole model is saved.
  Note: If you get `WARNING:tensorflow:Can save best model only with <name>
  available, skipping` see the description of the `monitor` argument for
  details on how to get this right.
  Example:
  ```python
  model.compile(loss=..., optimizer=...,
                metrics=['accuracy'])
  EPOCHS = 10
  checkpoint_filepath = '/tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)
  # Model weights are saved at the end of every epoch, if it's the best seen
  # so far.
  model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
  # The model weights (that are considered the best) are loaded into the model.
  model.load_weights(checkpoint_filepath)
  ```
  Arguments:
      filepath: string or `PathLike`, path to save the model file. `filepath`
        can contain named formatting options, which will be filled the value of
        `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
        `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
        checkpoints will be saved with the epoch number and the validation loss
        in the filename.
      monitor: The metric name to monitor. Typically the metrics are set by the
        `Model.compile` method. Note:
        * Prefix the name with `"val_`" to monitor validation metrics.
        * Use `"loss"` or "`val_loss`" to monitor the model's total loss.
        * If you specify metrics as strings, like `"accuracy"`, pass the same
          string (with or without the `"val_"` prefix).
        * If you pass `metrics.Metric` objects, `monitor` should be set to
          `metric.name`
        * If you're not sure about the metric names you can check the contents
          of the `history.history` dictionary returned by
          `history = model.fit()`
        * Multi-output models set additional prefixes on the metric names.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, it only saves when the model
        is considered the "best" and the latest best model according to the
        quantity monitored will not be overwritten. If `filepath` doesn't
        contain formatting options like `{epoch}` then `filepath` will be
        overwritten by each new better model.
      mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
        decision to overwrite the current save file is made based on either
        the maximization or the minimization of the monitored quantity.
        For `val_acc`, this should be `max`, for `val_loss` this should be
        `min`, etc. In `auto` mode, the direction is automatically inferred
        from the name of the monitored quantity.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of this many batches. If the `Model` is compiled with
        `steps_per_execution=N`, then the saving criteria will be
        checked every Nth batch. Note that if the saving isn't aligned to
        epochs, the monitored metric may potentially be less reliable (it
        could reflect as little as 1 batch, since the metrics get reset every
        epoch). Defaults to `'epoch'`.
      options: Optional `tf.train.CheckpointOptions` object if
        `save_weights_only` is true or optional `tf.saved_model.SaveOptions`
        object if `save_weights_only` is false.
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  """

  def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               save_freq='epoch',
               options=None,
               **kwargs):
    super(CustomizedModelCheckpoint, self).__init__()
    self._supports_tf_logs = True
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = path_to_string(filepath)
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.save_freq = save_freq
    self.epochs_since_last_save = 0
    self._batches_seen_since_last_saving = 0
    self._last_batch_seen = 0

    if save_weights_only:
      if options is None or isinstance(
          options, checkpoint_options_lib.CheckpointOptions):
        self._options = options or checkpoint_options_lib.CheckpointOptions()
      else:
        raise TypeError('If save_weights_only is True, then `options` must be'
                        'either None or a tf.train.CheckpointOptions')
    else:
      if options is None or isinstance(options, save_options_lib.SaveOptions):
        self._options = options or save_options_lib.SaveOptions()
      else:
        raise TypeError('If save_weights_only is False, then `options` must be'
                        'either None or a tf.saved_model.SaveOptions')

    # Deprecated field `load_weights_on_restart` is for loading the checkpoint
    # file from `filepath` at the start of `model.fit()`
    # TODO(rchao): Remove the arg during next breaking release.
    if 'load_weights_on_restart' in kwargs:
      self.load_weights_on_restart = kwargs['load_weights_on_restart']
      logging.warning('`load_weights_on_restart` argument is deprecated. '
                      'Please use `model.load_weights()` for loading weights '
                      'before the start of `model.fit()`.')
    else:
      self.load_weights_on_restart = False

    # Deprecated field `period` is for the number of epochs between which
    # the model is saved.
    if 'period' in kwargs:
      self.period = kwargs['period']
      logging.warning('`period` argument is deprecated. Please use `save_freq` '
                      'to specify the frequency in number of batches seen.')
    else:
      self.period = 1

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

    if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
      raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def set_model(self, model):
    self.model = model

  def on_train_begin(self, logs=None):
    if self.load_weights_on_restart:
      filepath_to_load = (
          self._get_most_recently_modified_file_matching_pattern(self.filepath))
      if (filepath_to_load is not None and
          self._checkpoint_exists(filepath_to_load)):
        try:
          # `filepath` may contain placeholders such as `{epoch:02d}`, and
          # thus it attempts to load the most recently modified file with file
          # name matching the pattern.
          self.model.load_weights(filepath_to_load)
        except (IOError, ValueError) as e:
          raise ValueError('Error loading file from {}. Reason: {}'.format(
              filepath_to_load, e))

  def _implements_train_batch_hooks(self):
    # Only call batch hooks when saving on batch
    return self.save_freq != 'epoch'

  def on_train_batch_end(self, batch, logs=None):
    if self._should_save_on_batch(batch):
      self._save_model(epoch=self._current_epoch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    self._current_epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    self.epochs_since_last_save += 1
    # pylint: disable=protected-access
    if self.save_freq == 'epoch':
      self._save_model(epoch=epoch, logs=logs)

  def _should_save_on_batch(self, batch):
    """Handles batch-level saving logic, supports steps_per_execution."""
    if self.save_freq == 'epoch':
      return False

    if batch <= self._last_batch_seen:  # New epoch.
      add_batches = batch + 1  # batches are zero-indexed.
    else:
      add_batches = batch - self._last_batch_seen
    self._batches_seen_since_last_saving += add_batches
    self._last_batch_seen = batch

    if self._batches_seen_since_last_saving >= self.save_freq:
      self._batches_seen_since_last_saving = 0
      return True
    return False

  def _save_model(self, epoch, logs):
    """Saves the model.
    Arguments:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}

    if isinstance(self.save_freq,
                  int) or self.epochs_since_last_save >= self.period:
      # Block only when saving interval is reached.
      logs = tf_utils.to_numpy_or_python_type(logs)
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, logs)

      try:
        if self.save_best_only:
          current = logs.get(self.monitor)
          if current is None:
            logging.warning('Can save best model only with %s available, '
                            'skipping.', self.monitor)
          else:
            if self.monitor_op(current, self.best):
              if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' % (epoch + 1, self.monitor,
                                               self.best, current, filepath))
              self.best = current
              if self.save_weights_only:
                self.model.save_weights(
                    filepath, overwrite=True, options=self._options)
              else:
                self.model.save(filepath, overwrite=True, options=self._options)
            else:
              if self.verbose > 0:
                print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, self.monitor, self.best))
        else:
          if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
          if self.save_weights_only:
            self.model.save_weights(
                filepath, overwrite=True, options=self._options)
          else:
            self.model.save(filepath, overwrite=True, options=self._options)

        self._maybe_remove_file()
      except IOError as e:
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
        if 'is a directory' in six.ensure_str(e.args[0]).lower():
          raise IOError('Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        'directory: {}'.format(filepath))
        # Re-throw the error for any other causes.
        raise e

  def _get_file_path(self, epoch, logs):
    """Returns the file path for checkpoint."""
    # pylint: disable=protected-access
    try:
      # `filepath` may contain placeholders such as `{epoch:02d}` and
      # `{mape:.2f}`. A mismatch between logged metrics and the path's
      # placeholders can cause formatting to fail.
      file_path = self.filepath.format(epoch=epoch + 1, **logs)
    except KeyError as e:
      raise KeyError('Failed to format this callback filepath: "{}". '
                     'Reason: {}'.format(self.filepath, e))
    self._write_filepath = distributed_file_utils.write_filepath(
        file_path, self.model.distribute_strategy)
    return self._write_filepath

  def _maybe_remove_file(self):
    # Remove the checkpoint directory in multi-worker training where this worker
    # should not checkpoint. It is a dummy directory previously saved for sync
    # distributed training.
    distributed_file_utils.remove_temp_dir_with_filepath(
        self._write_filepath, self.model.distribute_strategy)

  def _checkpoint_exists(self, filepath):
    """Returns whether the checkpoint `filepath` refers to exists."""
    if filepath.endswith('.h5'):
      return file_io.file_exists_v2(filepath)
    tf_saved_model_exists = file_io.file_exists_v2(filepath)
    tf_weights_only_checkpoint_exists = file_io.file_exists_v2(
        filepath + '.index')
    return tf_saved_model_exists or tf_weights_only_checkpoint_exists

  def _get_most_recently_modified_file_matching_pattern(self, pattern):
    """Returns the most recently modified filepath matching pattern.
    Pattern may contain python formatting placeholder. If
    `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
    check for most recently modified one that matches the pattern.
    In the rare case where there are more than one pattern-matching file having
    the same modified time that is most recent among all, return the filepath
    that is largest (by `>` operator, lexicographically using the numeric
    equivalents). This provides a tie-breaker when multiple files are most
    recent. Note that a larger `filepath` can sometimes indicate a later time of
    modification (for instance, when epoch/batch is used as formatting option),
    but not necessarily (when accuracy or loss is used). The tie-breaker is
    put in the logic as best effort to return the most recent, and to avoid
    undeterministic result.
    Modified time of a file is obtained with `os.path.getmtime()`.
    This utility function is best demonstrated via an example:
    ```python
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      # Write something to each of the files
    self.assertEqual(
        _get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])
    ```
    Arguments:
        pattern: The file pattern that may optionally contain python placeholder
            such as `{epoch:02d}`.
    Returns:
        The most recently modified file's full filepath matching `pattern`. If
        `pattern` does not contain any placeholder, this returns the filepath
        that
        exactly matches `pattern`. Returns `None` if no match is found.
    """
    dir_name = os.path.dirname(pattern)
    base_name = os.path.basename(pattern)
    base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

    # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
    # use that as it is more robust than `os.path.getmtime()`.
    latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
    if latest_tf_checkpoint is not None and re.match(
        base_name_regex, os.path.basename(latest_tf_checkpoint)):
      return latest_tf_checkpoint

    latest_mod_time = 0
    file_path_with_latest_mod_time = None
    n_file_with_latest_mod_time = 0
    file_path_with_largest_file_name = None

    if file_io.file_exists_v2(dir_name):
      for file_name in os.listdir(dir_name):
        # Only consider if `file_name` matches the pattern.
        if re.match(base_name_regex, file_name):
          file_path = os.path.join(dir_name, file_name)
          mod_time = os.path.getmtime(file_path)
          if (file_path_with_largest_file_name is None or
              file_path > file_path_with_largest_file_name):
            file_path_with_largest_file_name = file_path
          if mod_time > latest_mod_time:
            latest_mod_time = mod_time
            file_path_with_latest_mod_time = file_path
            # In the case a file with later modified time is found, reset
            # the counter for the number of files with latest modified time.
            n_file_with_latest_mod_time = 1
          elif mod_time == latest_mod_time:
            # In the case a file has modified time tied with the most recent,
            # increment the counter for the number of files with latest modified
            # time by 1.
            n_file_with_latest_mod_time += 1

    if n_file_with_latest_mod_time == 1:
      # Return the sole file that has most recent modified time.
      return file_path_with_latest_mod_time
    else:
      # If there are more than one file having latest modified time, return
      # the file path with the largest file name.
      return file_path_with_largest_file_name

