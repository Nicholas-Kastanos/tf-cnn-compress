import tensorflow as tf
import numpy as np


def yolov3_head(x, grid_size, NUM_CLASS, STRIDES, ANCHORS, XYSCALES):
    raw_s, raw_m, raw_l = x
    pred_s = decode(raw_s, grid_size[0], NUM_CLASS, STRIDES, ANCHORS, 0, XYSCALES)
    pred_m = decode(raw_m, grid_size[1], NUM_CLASS, STRIDES, ANCHORS, 1, XYSCALES)
    pred_l = decode(raw_l, grid_size[2], NUM_CLASS, STRIDES, ANCHORS, 2, XYSCALES)
    return pred_s, pred_m, pred_l


def decode(x, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALES=[1, 1, 1]):
    x = tf.reshape(
        x, (tf.shape(x)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(x, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1),
                             axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0),
                      [tf.shape(x)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) *
                XYSCALES[i]) - 0.5 * (XYSCALES[i] - 1) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
