# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def inception_v1_base(inputs, final_endpoint='Mixed_5c', scope='InceptionV1'):
  end_points = {}
  with variable_scope.variable_scope(scope, 'InceptionV1', [inputs]):
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected],
        weights_initializer=trunc_normal(0.01)):
      with arg_scope(
          [layers.conv2d, layers_lib.max_pool2d], stride=1, padding='SAME'):
        end_point = 'Conv2d_1a_7x7'
        net = layers.conv2d(inputs, 32, [7, 7], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
        end_point = 'MaxPool_2a_3x3'
        net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
        end_point = 'Conv2d_2b_1x1'
        net = layers.conv2d(net, 32, [1, 1], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
        end_point = 'Conv2d_2c_3x3'
        net = layers.conv2d(net, 96, [3, 3], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
        end_point = 'MaxPool_3a_3x3'
        net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_3b'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net, 8, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 16, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 16, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_3c'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_4b'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net,16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_4c'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_4d'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_4e'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_4f'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'MaxPool_5a_2x2'
        net = layers_lib.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_5b'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 64, [3, 3], scope='Conv2d_0a_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_5c'
        with variable_scope.variable_scope(end_point):
          with variable_scope.variable_scope('Branch_0'):
            branch_0 = layers.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with variable_scope.variable_scope('Branch_1'):
            branch_1 = layers.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = layers.conv2d(
                branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_2'):
            branch_2 = layers.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = layers.conv2d(
                branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with variable_scope.variable_scope('Branch_3'):
            branch_3 = layers_lib.max_pool2d(
                net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv2d(
                branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = array_ops.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v1(inputs,
                 num_classes=666,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=layers_lib.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1'):
  # Final pooling and prediction
  with variable_scope.variable_scope(
      scope, 'InceptionV1', [inputs, num_classes], reuse=reuse) as scope:
    with arg_scope(
        [layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
      net, end_points = inception_v1_base(inputs, scope=scope)
      with variable_scope.variable_scope('Logits'):
        net = layers_lib.avg_pool2d(
            net, [4, 4], stride=1, scope='MaxPool_0a_2x2')
        net = layers_lib.dropout(net, dropout_keep_prob, scope='Dropout_0b')
        logits = layers.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='Conv2d_0c_1x1')
        if spatial_squeeze:
          logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


inception_v1.default_image_size = 128


def inception_v1_arg_scope(weight_decay=0.00004,
                           use_batch_norm=True,
                           batch_norm_var_collection='moving_vars'):
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      }
  }
  if use_batch_norm:
    normalizer_fn = layers_lib.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope(
        [layers.conv2d],
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc


