import tensorflow as tf

bn_epsilon = 1.001e-5


def basic_block(x, filter_num, stride=1, name=None):

    if stride != 1:
        route = tf.keras.layers.Conv2D(filter_num, kernel_size=(
            1, 1), strides=stride, name=name + '_0_conv')(x)
    else:
        route = x

    x = tf.keras.layers.Conv2D(filter_num, kernel_size=(
        3, 3), strides=stride, padding='same', name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', )(x)

    x = tf.keras.layers.Conv2D(filter_num, kernel_size=(
        3, 3), strides=1, padding='same', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name=name + '_1_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([route, x])
    x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
    return x


def bottleneck(x, filter_num, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    if conv_shortcut:
        route = tf.keras.layers.Conv2D(
            4 * filter_num, 1, strides=stride, name=name + '_0_conv')(x)
        route = tf.keras.layers.BatchNormalization(
            epsilon=bn_epsilon, name=name + '_0_bn')(route)
    else:
        route = x

    x = tf.keras.layers.Conv2D(
        filter_num, 1, strides=stride, name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = tf.keras.layers.Conv2D(
        filter_num, kernel_size, strides=1, padding='same', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name=name + '_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv2D(
        4 * filter_num, 1, strides=1, name=name + '_3_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name=name + '_3_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([route, x])
    x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
    return x

def conv(x, filters, kernel_size, spatial_sep_conv=False, depthwise_sep_conv=False, strides=1, name=None, **kwargs):
    if kernel_size == 1: # No point in doing two 1x1 convs
        spatial_sep_conv=False
        depthwise_sep_conv=False
    if depthwise_sep_conv and not spatial_sep_conv:
        return tf.keras.layers.SeparableConv2D(filters, kernel_size, strides=strides, name=name+'_dconv', **kwargs)(x)
    if not depthwise_sep_conv and spatial_sep_conv:
        x = tf.keras.layers.Conv2D(filters, (kernel_size, 1), strides=(strides, 1), name=name+'_aconv0', **kwargs)(x)
        x = tf.keras.layers.Conv2D(filters, (1, kernel_size), strides=(1, strides), name=name+'_aconv1', **kwargs)(x)
        return x
    if depthwise_sep_conv and spatial_sep_conv:
        x = tf.keras.layers.SeparableConv2D(filters, (kernel_size, 1), strides=(strides, 1), name=name+'_adconv0', **kwargs)(x)
        x = tf.keras.layers.SeparableConv2D(filters, (1, kernel_size), strides=(1, strides), name=name+'_adconv1', **kwargs)(x)
        return x
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, name=name+'_conv', **kwargs)(x)


def bottleneck_v2(x, filter_num, kernel_size=3, stride=1, conv_shortcut=False, spatial_sep_conv=False, depthwise_sep_conv=False, name=None):

    preact = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name=name + '_preact_bn')(x)
    preact = tf.keras.layers.Activation(
        'relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = conv(preact, 4 * filter_num, 1, spatial_sep_conv, depthwise_sep_conv, strides=stride, name=name + '_0')
    else:
        shortcut = tf.keras.layers.MaxPooling2D(
            1, strides=stride)(x) if stride > 1 else x

    x = conv(preact, filter_num, 1, spatial_sep_conv, depthwise_sep_conv, strides=1, use_bias=False, name=name + '_1')
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = tf.keras.layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = conv(x, filter_num, kernel_size, spatial_sep_conv, depthwise_sep_conv, strides=stride, use_bias=False, name=name + '_2')
    x = tf.keras.layers.BatchNormalization(
        epsilon=bn_epsilon, name=name + '_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = conv(x, 4 * filter_num, 1, spatial_sep_conv, depthwise_sep_conv, name=name + '_3')
    x = tf.keras.layers.Add(name=name + '_out')([shortcut, x])
    return x


def basic_block_layer(x, filter_num, blocks, stride=1, name=None):
    x = basic_block(x, filter_num, 
        stride=stride, 
        name=name+'_block1')
    for i in range(2, blocks + 1):
        x = basic_block(x, filter_num, 
            stride=1, 
            name=name+'_block'+str(i+1))
    return x


def bottleneck_layer(x, filter_num, blocks, stride=2, name=None):
    x = bottleneck(x, filter_num, 
        stride=stride, 
        name=name+'_block1')
    for i in range(2, blocks + 1):
        x = bottleneck(x, filter_num, 
            conv_shortcut=False,
            name=name + '_block' + str(i))
    return x


def bottleneck_layer_v2(x, filter_num, blocks, stride=2, spatial_sep_conv=False, depthwise_sep_conv=False, name=None):
    x = bottleneck_v2(x, filter_num, 
        conv_shortcut=True, 
        spatial_sep_conv=spatial_sep_conv,
        depthwise_sep_conv=depthwise_sep_conv, 
        name=name+'_block1')
    for i in range(2, blocks):
        x = bottleneck_v2(x, filter_num, 
            spatial_sep_conv=spatial_sep_conv,
            depthwise_sep_conv=depthwise_sep_conv, 
            name=name + '_block' + str(i))
    x = bottleneck_v2(x, filter_num, 
        stride=stride, 
        spatial_sep_conv=spatial_sep_conv,
        depthwise_sep_conv=depthwise_sep_conv, 
        name=name + '_block' + str(blocks))
    return x
