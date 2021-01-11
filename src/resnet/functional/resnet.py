import tensorflow as tf
from .residual_block import basic_block_layer, bottleneck_layer, bottleneck_layer_v2, bn_epsilon


def resnet_type_i(x, layer_params, filter_num=64, units=10):
    x = tf.keras.layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(x)
    x = tf.keras.layers.Conv2D(filter_num, kernel_size=(
        7, 7), strides=2, padding='valid', name='conv1_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name='conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)
    x = tf.keras.layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(
        3, 3), strides=2, name='pool1_pool')(x)

    x = basic_block_layer(x, filter_num, layer_params[0])
    x = basic_block_layer(x, filter_num * 2, layer_params[1], stride=2)
    x = basic_block_layer(x, filter_num * 4, layer_params[2], stride=2)
    x = basic_block_layer(x, filter_num * 8, layer_params[3], stride=2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        units=units, activation=tf.keras.activations.softmax)(x)
    return x


def resnet_type_ii(x, layer_params, filter_num=64, units=10):
    # ResNetV1
    x = tf.keras.layers.ZeroPadding2D(((3, 3), (3, 3)), name='conv1_pad')(x)
    x = tf.keras.layers.Conv2D(filter_num, 7, strides=2, name='conv1_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name='conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

    x = tf.keras.layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2, name='pool1_pool')(x)

    x = bottleneck_layer(x, filter_num=filter_num,
                         blocks=layer_params[0], stride=1, name='conv2')
    x = bottleneck_layer(x, filter_num=filter_num * 2,
                         blocks=layer_params[1], name='conv3')
    x = bottleneck_layer(x, filter_num=filter_num * 4,
                         blocks=layer_params[2], name='conv4')
    x = bottleneck_layer(x, filter_num=filter_num * 8,
                         blocks=layer_params[3], name='conv5')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(
        units=units, activation=tf.keras.activations.softmax, name='predictions')(x)
    return x


def resnet_type_ii_v2(x, layer_params, filter_num=64, units=10, spatial_sep_conv=False, depthwise_sep_conv=False):
    # ResNetV2
    x = tf.keras.layers.ZeroPadding2D(((3, 3), (3, 3)), name='conv1_pad')(x)
    x = tf.keras.layers.Conv2D(filter_num, 7, strides=2, name='conv1_conv')(x)

    x = tf.keras.layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2, name='pool1_pool')(x)

    x = bottleneck_layer_v2(x, filter_num=filter_num,
                            blocks=layer_params[0], spatial_sep_conv=spatial_sep_conv, depthwise_sep_conv=depthwise_sep_conv, name='conv2')
    x = bottleneck_layer_v2(x, filter_num=filter_num * 2,
                            blocks=layer_params[1], spatial_sep_conv=spatial_sep_conv, depthwise_sep_conv=depthwise_sep_conv, name='conv3')
    x = bottleneck_layer_v2(x, filter_num=filter_num * 4,
                            blocks=layer_params[2], spatial_sep_conv=spatial_sep_conv, depthwise_sep_conv=depthwise_sep_conv, name='conv4')
    x = bottleneck_layer_v2(x, filter_num=filter_num * 8, blocks=layer_params[3], stride=1,
                            spatial_sep_conv=spatial_sep_conv, depthwise_sep_conv=depthwise_sep_conv, name='conv5')

    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name='post_bn')(x)
    x = tf.keras.layers.Activation('relu', name='post_relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(
        units=units, activation=tf.keras.activations.softmax, name='predictions')(x)
    return x


def resnet_18(input_tensor, filters=64, units=10):
    input_layer = tf.keras.utils.get_source_inputs(input_tensor)
    output_layer = resnet_type_i(input_tensor, layer_params=[
                                 2, 2, 2, 2], filter_num=filters, units=units)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet18")


def resnet_34(input_tensor, filters=64, units=10):
    input_layer = tf.keras.utils.get_source_inputs(input_tensor)
    output_layer = resnet_type_i(input_tensor, layer_params=[
                                 3, 4, 6, 3], filter_num=filters, units=units)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet34")


def resnet_50(input_tensor, filters=64, units=10, v2=True, spatial_sep_conv=False, depthwise_sep_conv=False):
    input_layer = tf.keras.utils.get_source_inputs(input_tensor)
    layer_params = [3, 4, 6, 3]
    if v2:
        output_layer = resnet_type_ii_v2(
            input_tensor, layer_params, filters, units, spatial_sep_conv, depthwise_sep_conv)
    else:
        output_layer = resnet_type_ii(
            input_tensor, layer_params, filters, units)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name=("ResNet50" if not v2 else "ResNet50v2"))


def resnet_101(input_tensor, filters=64, units=10, v2=True):
    input_layer = tf.keras.utils.get_source_inputs(input_tensor)
    layer_params = [3, 4, 23, 3]
    if v2:
        output_layer = resnet_type_ii_v2(
            input_tensor, layer_params, filters, units)
    else:
        output_layer = resnet_type_ii(
            input_tensor, layer_params, filters, units)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name=("ResNet101" if not v2 else "ResNet101v2"))


def resnet_152(input_tensor, filters=64, units=10, v2=True):
    input_layer = tf.keras.utils.get_source_inputs(input_tensor)
    layer_params = [3, 8, 36, 3]
    if v2:
        output_layer = resnet_type_ii_v2(
            input_tensor, layer_params, filters, units)
    else:
        output_layer = resnet_type_ii(
            input_tensor, layer_params, filters, units)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name=("ResNet152" if not v2 else "ResNet152v2"))
