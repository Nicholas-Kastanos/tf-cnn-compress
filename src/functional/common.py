import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)


def darknet_conv(
    x,
    filters: int,
    kernel_size: int,
    downsample=False,
    activate=True,
    bn=True,
    activate_type="leaky",
    use_asymetric_conv=False
):
    if downsample:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    if use_asymetric_conv and kernel_size==3 and not downsample:
        conv = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=(kernel_size, 1), 
            strides=strides, 
            padding=padding, 
            use_bias=not bn, 
            kernel_regularizer=tf.keras.regularizers.l2(0.0005), 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01), 
            bias_initializer=tf.constant_initializer(0.)
        )(x)
        conv = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=(1, kernel_size), 
            strides=strides, 
            padding=padding, 
            use_bias=not bn, 
            kernel_regularizer=tf.keras.regularizers.l2(0.0005), 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01), 
            bias_initializer=tf.constant_initializer(0.)
        )(conv)
    else:
        conv = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            use_bias=not bn, 
            kernel_regularizer=tf.keras.regularizers.l2(0.0005), 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01), 
            bias_initializer=tf.constant_initializer(0.)
        )(x)

    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv


def darknet_residual(
    x, 
    filters_1: int, 
    filters_2: int, 
    activate_type="leaky",
    use_asymetric_conv=False
):
    short_cut = x
    x = darknet_conv(x=x, filters=filters_1, kernel_size=1, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
    x = darknet_conv(x=x, filters=filters_2, kernel_size=3, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)

    x = short_cut + x
    return x

# def residual_block(
#     x,
#     iterations: int,
#     filters_1: int,
#     filters_2: int,
#     activate_type="leaky",
#     use_asymetric_conv=False
# ):
#     for _ in range(iterations):
#         x = darknet_residual(x, filters_1, filters_2, activate_type, use_asymetric_conv)
#     return x

# def darknet_residual_block(
#     x,
#     iterations: int,
#     filters_1: int,
#     filters_2: int,
#     activate_type="leaky",
#     use_asymetric_conv=False
# ):
#     x = darknet_conv(x, filters=filters_2, kernel_size=3, downsample=True, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
#     x = residual_block(x, iterations, filters_1, filters_2, activate_type, use_asymetric_conv)
#     return x
    
# def block_tiny(input_layer, input_channel, filter_num1, activate_type='leaky'):
#     conv = convolutional(input_layer, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#     short_cut = input_layer
#     conv = convolutional(conv, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#
#     input_data = tf.concat([conv, short_cut], axis=-1)
#     return residual_output

def csp_darknet_residual_block(# Combination of CPSResidualBlock and CSPDarknetResidualBlock
    x,
    iterations: int,
    filters_1: int,
    filters_2: int,
    activate_type: str = "mish",
    use_asymetric_conv=False
):
    x = darknet_conv(x, filters=filters_2, kernel_size=3, downsample=True, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
    route = x
    route = darknet_conv(route, filters=filters_2, kernel_size=1, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
    x = darknet_conv(x, filters=filters_2, kernel_size=1, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
    for _ in range(iterations):
        x = darknet_residual(x,  filters_1=filters_1, filters_2=filters_2, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
    x = darknet_conv(x, filters=filters_2, kernel_size=1, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
    x = tf.concat([x, route], axis=-1)   
    x = darknet_conv(x, filters=filters_2, kernel_size=1, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
    return x



def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
