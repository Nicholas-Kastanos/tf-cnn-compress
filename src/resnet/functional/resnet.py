import tensorflow as tf 
from .residual_block import basic_block_layer, bottleneck_layer

def resnet_type_i(x, layer_params, filter_num=64, units=10):
    x = tf.keras.layers.Conv2D(filter_num,
        kernel_size=(7, 7),
        strides=2,
        padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same'
    )(x)

    x = basic_block_layer(x, filter_num, layer_params[0])
    x = basic_block_layer(x, filter_num * 2, layer_params[1], stride=2)
    x = basic_block_layer(x, filter_num * 4, layer_params[2], stride=2)
    x = basic_block_layer(x, filter_num * 8, layer_params[3], stride=2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.softmax)(x)
    return x
    

def resnet_type_ii(x, layer_params, filter_num=64, units=10):
    x = tf.keras.layers.Conv2D(filter_num,
        kernel_size=(7,7),
        strides=2,
        padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=2,
        padding="same"
    )(x)

    x = bottleneck_layer(x,
        filter_num=filter_num,
        blocks=layer_params[0]
    )
    x = bottleneck_layer(x,
        filter_num=filter_num*2,
        blocks=layer_params[1]            
    )
    x = bottleneck_layer(x,
        filter_num=filter_num*4,
        blocks=layer_params[2]
    )
    x = bottleneck_layer(x,
        filter_num=filter_num*8,
        blocks=layer_params[3]
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        units=units,
        activation=tf.keras.activations.softmax
    )(x)
    return x

def resnet_18(input_shape, filters=64, units=10):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    output_layer = resnet_type_i(input_layer, layer_params=[2,2,2,2], filter_num=filters, units=units)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet18")

def resnet_34(input_shape, filters=64, units=10):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    output_layer = resnet_type_i(input_layer, layer_params=[3,4,6,3], filter_num=filters, units=units)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet34")

def resnet_50(input_shape, filters=64, units=10):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    output_layer = resnet_type_ii(input_layer, layer_params=[3,4,6,3], filter_num=filters, units=units)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet50")

def resnet_101(input_shape, filters=64, units=10):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    output_layer = resnet_type_ii(input_layer, layer_params=[3,4,23,3], filter_num=filters, units=units)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet101")

def resnet_152(input_shape, filters=64, units=10):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    output_layer = resnet_type_ii(input_layer, layer_params=[3,8,36,3], filter_num=filters, units=units)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="ResNet152")