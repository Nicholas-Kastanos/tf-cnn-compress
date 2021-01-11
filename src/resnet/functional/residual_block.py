import tensorflow as tf 

def basic_block(x, filter_num, stride=1, name=None):

    if stride != 1:
        route = tf.keras.layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride, name=name + '_0_conv')(x)
    else:
        route = x

    x = tf.keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=stride, padding='same', name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=1, padding='same', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([route, x])
    x = tf.keras.layers.ReLU(name=name + '_out')(x)
    return x

def bottleneck(x, filter_num, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    if conv_shortcut:
        route = tf.keras.layers.Conv2D(4 * filter_num, 1, strides=stride, name=name + '_0_conv')(x)
        route = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(route)
    else:
        route = x

    x = tf.keras.layers.Conv2D(filter_num, 1, strides=stride, name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x) 
    x = tf.keras.layers.ReLU(name=name + '_1_relu')(x)

    x = tf.keras.layers.Conv2D(filter_num, kernel_size, strides=1, padding='same', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = tf.keras.layers.ReLU(name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv2D(4 * filter_num, 1, strides=1, name=name + '_3_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_3_bn')(x)
    
    x = tf.keras.layers.Add(name=name + '_add')([route, x])
    x = tf.keras.layers.ReLU(name=name + '_out')(x)
    return x

def basic_block_layer(x, filter_num, blocks, stride=1, name=None):
    x = basic_block(x, filter_num, stride=stride, name=name+'_block1')
    for i in range(2, blocks + 1):
        x = basic_block(x, filter_num, stride=1, name=name+'_block'+str(i+1))
    return x

def bottleneck_layer(x, filter_num, blocks, stride=2, name=None):
    x = bottleneck(x, filter_num, stride=stride, name=name+'_block1')
    for i in range(2, blocks + 1):
        x = bottleneck(x, filter_num, conv_shortcut=False, name=name + '_block' + str(i))
    return x