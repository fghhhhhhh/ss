from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from frontends import resnet_utils

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):

  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

          net = slim.utils.collect_named_outputs(end_points_collection, 'pool2', net)
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        end_points['pool3'] = end_points[scope + '/block1']
        end_points['pool4'] = end_points[scope + '/block2']
        end_points['pool5'] = net
        return net, end_points
resnet_v2.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride):

  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])
resnet_v2.default_image_size = 224


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_50.default_image_size = resnet_v2.default_image_size


def resnet_v2_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_101.default_image_size = resnet_v2.default_image_size


def resnet_v2_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_152.default_image_size = resnet_v2.default_image_size


def resnet_v2_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_200.default_image_size = resnet_v2.default_image_size
