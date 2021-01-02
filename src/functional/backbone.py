#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import functional.common as common

# def darknet53(input_data, use_asymetric_conv=False):

#     input_data = common.darknet_conv(
#         input_data, 
#         kernel_size=3, 
#         filters=32, 
#         use_asymetric_conv=use_asymetric_conv)

#     input_data = common.darknet_residual_block(
#         input_data, 
#         iterations=1, 
#         filters_1=32, 
#         filters_2=64, 
#         use_asymetric_conv=use_asymetric_conv)

#     input_data = common.darknet_residual_block(
#         input_data, 
#         iterations=2, 
#         filters_1=64, 
#         filters_2=128, 
#         use_asymetric_conv=use_asymetric_conv)

#     input_data = common.darknet_residual_block(
#         input_data, 
#         iterations=8, 
#         filters_1=128, 
#         filters_2=256, 
#         use_asymetric_conv=use_asymetric_conv)

#     route_1 = input_data

#     input_data = common.darknet_residual_block(
#         input_data, 
#         iterations=8, 
#         filters_1=256, 
#         filters_2=512, 
#         use_asymetric_conv=use_asymetric_conv)    

#     route_2 = input_data

#     input_data = common.darknet_residual_block(
#         input_data, 
#         iterations=4, 
#         filters_1=512, 
#         filters_2=1024, 
#         use_asymetric_conv=use_asymetric_conv) 

#     return route_1, route_2, input_data

def cspdarknet53(input_data, use_asymetric_conv=False):

    input_data = common.darknet_conv(input_data, filters=32, kernel_size=3, activate_type="mish", use_asymetric_conv=use_asymetric_conv)

    input_data = common.csp_darknet_residual_block(
        input_data,
        iterations=1,
        filters_1=32,
        filters_2=64,
        activate_type="mish",
        use_asymetric_conv=use_asymetric_conv
    )

    input_data = common.csp_darknet_residual_block(
        input_data,
        iterations=2,def darknet_residual(
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
        activate_type="mish",
        use_asymetric_conv=use_asymetric_conv
    )
    
    input_data = common.csp_darknet_residual_block(
        input_data,
        iterations=8,
        filters_1=128,
        filters_2=256,
        activate_type="mish",
        use_asymetric_conv=use_asymetric_conv
    )

    route_1 = input_data

    input_data = common.csp_darknet_residual_block(
        input_data,
        iterations=8,
        filters_1=256,
        filters_2=512,
        activate_type="mish",
        use_asymetric_conv=use_asymetric_conv
    )

    route_2 = input_data
    
    input_data = common.csp_darknet_residual_block(
        input_data,
        iterations=4,
        filters_1=512,
        filters_2=1024,
        activate_type="mish",
        use_asymetric_conv=use_asymetric_conv
    )
    
    input_data = common.darknet_conv(input_data, filters=512, kernel_size=1, use_asymetric_conv=use_asymetric_conv)
    input_data = common.darknet_conv(input_data, filters=1024, kernel_size=3, use_asymetric_conv=use_asymetric_conv)
    input_data = common.darknet_conv(input_data, filters=512, kernel_size=1, use_asymetric_conv=use_asymetric_conv)

    # SPP
    input_data = tf.concat([
        tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), 
        tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1), 
        tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), 
        input_data
    ], axis=-1)

    input_data = common.darknet_conv(input_data, filters=512, kernel_size=1, use_asymetric_conv=use_asymetric_conv)
    input_data = common.darknet_conv(input_data, filters=1024, kernel_size=3, use_asymetric_conv=use_asymetric_conv)
    input_data = common.darknet_conv(input_data, filters=512, kernel_size=1, use_asymetric_conv=use_asymetric_conv)

    return route_1, route_2, input_data

# def cspdarknet53_tiny(input_data):
#     input_data = common.convolutional(input_data, (3, 3, 3, 32), downsample=True)
#     input_data = common.convolutional(input_data, (3, 3, 32, 64), downsample=True)
#     input_data = common.convolutional(input_data, (3, 3, 64, 64))

#     route = input_data
#     input_data = common.route_group(input_data, 2, 1)
#     input_data = common.convolutional(input_data, (3, 3, 32, 32))
#     route_1 = input_data
#     input_data = common.convolutional(input_data, (3, 3, 32, 32))
#     input_data = tf.concat([input_data, route_1], axis=-1)
#     input_data = common.convolutional(input_data, (1, 1, 32, 64))
#     input_data = tf.concat([route, input_data], axis=-1)
#     input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

#     input_data = common.convolutional(input_data, (3, 3, 64, 128))
#     route = input_data
#     input_data = common.route_group(input_data, 2, 1)
#     input_data = common.convolutional(input_data, (3, 3, 64, 64))
#     route_1 = input_data
#     input_data = common.convolutional(input_data, (3, 3, 64, 64))
#     input_data = tf.concat([input_data, route_1], axis=-1)
#     input_data = common.convolutional(input_data, (1, 1, 64, 128))
#     input_data = tf.concat([route, input_data], axis=-1)
#     input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

#     input_data = common.convolutional(input_data, (3, 3, 128, 256))
#     route = input_data
#     input_data = common.route_group(input_data, 2, 1)
#     input_data = common.convolutional(input_data, (3, 3, 128, 128))
#     route_1 = input_data
#     input_data = common.convolutional(input_data, (3, 3, 128, 128))
#     input_data = tf.concat([input_data, route_1], axis=-1)
#     input_data = common.convolutional(input_data, (1, 1, 128, 256))
#     route_1 = input_data
#     input_data = tf.concat([route, input_data], axis=-1)
#     input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

#     input_data = common.convolutional(input_data, (3, 3, 512, 512))

#     return route_1, input_data

# def darknet53_tiny(input_data):
#     input_data = common.convolutional(input_data, (3, 3, 3, 16))
#     input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
#     input_data = common.convolutional(input_data, (3, 3, 16, 32))
#     input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
#     input_data = common.convolutional(input_data, (3, 3, 32, 64))
#     input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
#     input_data = common.convolutional(input_data, (3, 3, 64, 128))
#     input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
#     input_data = common.convolutional(input_data, (3, 3, 128, 256))
#     route_1 = input_data
#     input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
#     input_data = common.convolutional(input_data, (3, 3, 256, 512))
#     input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
#     input_data = common.convolutional(input_data, (3, 3, 512, 1024))

#     return route_1, input_data


