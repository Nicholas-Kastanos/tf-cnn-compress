import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, backend, layers, Layer

class YOLOv3Head(Layer):
    def __init__(self, anchors, num_classes, xysclaes):
        super(YOLOv3Head, self).__init__(name="YOLOv3Head")
        self.a_half = None
        self.anchors = anchors
        self.grid_coord = []
        self.grid_size = None
        self.image_width = None
        self.num_classes = num_classes
        self.scales = xysclaes

        self.reshape0 = layers.Reshape((-1,))
        self.reshape1 = layers.Reshape((-1,))
        self.reshape2 = layers.Reshape((-1,))

        self.concat0 = layers.Concatenate(axis=-1)
        self.concat1 = layers.Concatenate(axis=-1)
        self.concat2 = layers.Concatenate(axis=-1)

    def build(self, input_shape):
        # None, g_height, g_width,
        #       (xywh + conf + num_classes) * (# of anchors)

        # g_width, g_height
        _size = [(shape[2], shape[1]) for shape in input_shape]

        self.reshape0.target_shape = (
            _size[0][1],
            _size[0][0],
            3,
            5 + self.num_classes,
        )
        self.reshape1.target_shape = (
            _size[1][1],
            _size[1][0],
            3,
            5 + self.num_classes,
        )
        self.reshape2.target_shape = (
            _size[2][1],
            _size[2][0],
            3,
            5 + self.num_classes,
        )

        self.a_half = [
            tf.constant(
                0.5,
                dtype=tf.float32,
                shape=(1, _size[i][1], _size[i][0], 3, 2),
            )
            for i in range(3)
        ]

        for i in range(3):
            xy_grid = tf.meshgrid(tf.range(_size[i][0]), tf.range(_size[i][1]))
            xy_grid = tf.stack(xy_grid, axis=-1)
            xy_grid = xy_grid[tf.newaxis, :, :, tf.newaxis, :]
            xy_grid = tf.tile(xy_grid, [1, 1, 1, 3, 1])
            xy_grid = tf.cast(xy_grid, tf.float32)
            self.grid_coord.append(xy_grid)

        self.grid_size = tf.convert_to_tensor(_size, dtype=tf.float32)
        self.image_width = tf.convert_to_tensor(
            _size[0][0] * 8.0, dtype=tf.float32
        )

    def call(self, x):
        raw_s, raw_m, raw_l = x

        raw_s = self.reshape0(raw_s)
        raw_m = self.reshape1(raw_m)
        raw_l = self.reshape2(raw_l)

        txty_s, twth_s, conf_s, prob_s = tf.split(
            raw_s, (2, 2, 1, self.num_classes), axis=-1
        )
        txty_m, twth_m, conf_m, prob_m = tf.split(
            raw_m, (2, 2, 1, self.num_classes), axis=-1
        )
        txty_l, twth_l, conf_l, prob_l = tf.split(
            raw_l, (2, 2, 1, self.num_classes), axis=-1
        )

        txty_s = activations.sigmoid(txty_s)
        txty_s = (txty_s - self.a_half[0]) * self.scales[0] + self.a_half[0]
        bxby_s = (txty_s + self.grid_coord[0]) / self.grid_size[0]
        txty_m = activations.sigmoid(txty_m)
        txty_m = (txty_m - self.a_half[1]) * self.scales[1] + self.a_half[1]
        bxby_m = (txty_m + self.grid_coord[1]) / self.grid_size[1]
        txty_l = activations.sigmoid(txty_l)
        txty_l = (txty_l - self.a_half[2]) * self.scales[2] + self.a_half[2]
        bxby_l = (txty_l + self.grid_coord[2]) / self.grid_size[2]

        conf_s = activations.sigmoid(conf_s)
        conf_m = activations.sigmoid(conf_m)
        conf_l = activations.sigmoid(conf_l)

        prob_s = activations.sigmoid(prob_s)
        prob_m = activations.sigmoid(prob_m)
        prob_l = activations.sigmoid(prob_l)

        bwbh_s = (self.anchors[0] / self.image_width) * backend.exp(twth_s)
        bwbh_m = (self.anchors[1] / self.image_width) * backend.exp(twth_m)
        bwbh_l = (self.anchors[2] / self.image_width) * backend.exp(twth_l)

        pred_s = self.concat0([bxby_s, bwbh_s, conf_s, prob_s])
        pred_m = self.concat1([bxby_m, bwbh_m, conf_m, prob_m])
        pred_l = self.concat2([bxby_l, bwbh_l, conf_l, prob_l])

        return pred_s, pred_m, pred_l