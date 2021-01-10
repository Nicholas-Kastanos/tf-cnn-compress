import tensorflow as tf 

def basic_block(x, filter_num, stride=1):
    route = x
    x = tf.keras.layers.Conv2D(filter_num, 
        kernel_size=(3, 3), 
        strides=stride,
        padding='same'
    )(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filter_num, 
        kernel_size=(3, 3), 
        strides=1,
        padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1:
        route = tf.keras.layers.Conv2D(filter_num,
            kernel_size=(1, 1),
            strides=stride
        )(route)

    x = tf.keras.layers.Add()([route, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def bottleneck(x, filter_num, stride=1):
    route = x

    x = tf.keras.layers.Conv2D(filter_num,
        kernel_size=(1, 1),
        strides=1,
        padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filter_num,
        kernel_size=(3, 3),
        strides=stride,
        padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(4 * filter_num,
        kernel_size=(1, 1),
        strides=1,
        padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    route = tf.keras.layers.Conv2D(4 * filter_num,
        kernel_size=(1, 1),
        strides=stride
    )(route)
    route = tf.keras.layers.BatchNormalization()(route)
    
    x = tf.keras.layers.Add()([route, x])
    x = tf.keras.layers.ReLU()(x)
    return x

def basic_block_layer(x, filter_num, blocks, stride=1):
    for _ in range(blocks):
        x = basic_block(x, filter_num, stride=stride)
    return x

def bottleneck_layer(x, filter_num, blocks, stride=1):
    x = bottleneck(x, filter_num, stride=stride)
    for _ in range(1, blocks):
        x = bottleneck(x, filter_num, stride=1)
    return x