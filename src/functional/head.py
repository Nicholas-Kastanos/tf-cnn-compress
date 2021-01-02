import tensorflow as tf
import numpy as np


def yolov3_head(x, grid_size, NUM_CLASS, STRIDES, ANCHORS, XYSCALES, IMAGE_WIDTH):
    raw_s, raw_m, raw_l = x
    pred_s = decode(raw_s, grid_size[0], NUM_CLASS, STRIDES, ANCHORS, 0, XYSCALES)
    pred_m = decode(raw_m, grid_size[1], NUM_CLASS, STRIDES, ANCHORS, 1, XYSCALES)
    pred_l = decode(raw_l, grid_size[2], NUM_CLASS, STRIDES, ANCHORS, 2, XYSCALES)
    return pred_s, pred_m, pred_l


def decode(x, grid_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALES=[1, 1, 1], IMAGE_WIDTH=416):
    batch_size=tf.shape(x)[0]
    x = tf.reshape(
        x, (batch_size, grid_size[0], grid_size[1], 3, 5 + NUM_CLASS))

    raw_dxdy, raw_dwdh, raw_conf, raw_prob = tf.split(x, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(grid_size[0]), tf.range(grid_size[1]))
    xy_grid = tf.stack(xy_grid, axis=-1)
    xy_grid = xy_grid[tf.newaxis, :, :, tf.newaxis, :]
    xy_grid = tf.tile(xy_grid, [1, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    a_half = tf.constant(0.5, dtype=tf.float32, shape=(grid_size[0], grid_size[1], 3, 2))

    txty = tf.sigmoid(raw_dxdy)
    txty = (txty - a_half) * XYSCALES[i] + a_half
    bxby = (txty + xy_grid) / grid_size

    conf = tf.sigmoid(raw_conf)
    prob = tf.sigmoid(raw_prob)

    bwbh = (ANCHORS[i] / IMAGE_WIDTH) * tf.exp(raw_dwdh)

    return tf.concat([bxby, bwbh, conf, prob], axis=-1)
