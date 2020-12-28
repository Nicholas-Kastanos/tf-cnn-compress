#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import tensorboard
from typing import Tuple
import matplotlib.pyplot as plt
from src.yolov4.yolov4 import YOLOv4
# import src.dataset as dataset
import src.train as train
# import src.predict as predict
# import src.media as media
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras import backend, layers, optimizers, regularizers, callbacks
from tensorflow import keras
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    data_dir = 'gs://tfds-data/datasets'
else:
    data_dir = os.path.join('/', 'media', 'nicholas',
                            'Data', 'nicho', 'Documents', 'tensorflow_datasets')


(ds_train, ds_val), ds_info = tfds.load(
    'coco/2017',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
    data_dir=data_dir
)

anchors = np.array([
    [[12, 16], [19, 36], [40, 28]],
    [[36, 75], [76, 55], [72, 146]],
    [[142, 110], [192, 243], [459, 401]],
]).astype(np.float32).reshape(3, 3, 2)
strides = np.array([8, 16, 32])
xyscales = np.array([1.2, 1.1, 1.05])
input_size = (256, 256)  # (416, 416)
anchors_ratio = anchors / input_size[0]
batch_size = 3
grid_size = (input_size[1], input_size[0]) // np.stack(
    (strides, strides), axis=1
)
label_smoothing = 0.1
included_classes = ds_info.features["objects"]["label"].names[:20]
num_classes = len(included_classes)
class_dict = dict(
    zip(
        range(num_classes),
        included_classes
    )
)
included_classes_idxs = np.asarray(list(class_dict.keys()))

print(class_dict)
print(num_classes)

grid_xy = [
    np.tile(
        np.reshape(
            np.stack(
                np.meshgrid(
                    (np.arange(_size[0]) + 0.5) / _size[0],
                    (np.arange(_size[1]) + 0.5) / _size[1],
                ),
                axis=-1,
            ),
            (1, _size[0], _size[1], 1, 2),
        ),
        (1, 1, 1, 3, 1),
    ).astype(np.float32)
    for _size in grid_size  # (height, width)
]

chance_to_remove_person_only_images = 0.5


def bboxes_to_ground_truth(bboxes):
    """
        @param bboxes: [[b_x, b_y, b_w, b_h, class_id], ...]

        @return [s, m, l] or [s, l]
            Dim(1, grid_y, grid_x, anchors,
                                (b_x, b_y, b_w, b_h, conf, prob_0, prob_1, ...))
        """
    ground_truth = [
        np.zeros(
                (
                    1,
                    _size[0],
                    _size[1],
                    3,
                    5 + num_classes,
                ),
            dtype=np.float32,
        )
        for _size in grid_size
    ]

    for i, _grid in enumerate(grid_xy):
        ground_truth[i][..., 0:2] = _grid

    for bbox in bboxes:
        # [b_x, b_y, b_w, b_h, class_id]
        xywh = np.array(bbox[:4], dtype=np.float32)
        class_id = int(bbox[4])

        # smooth_onehot = [0.xx, ... , 1-(0.xx*(n-1)), 0.xx, ...]
        onehot = np.zeros(num_classes, dtype=np.float32)
        onehot[class_id] = 1.0
        uniform_distribution = np.full(
            num_classes, 1.0 / num_classes, dtype=np.float32
        )
        smooth_onehot = (
            1 - label_smoothing
        ) * onehot + label_smoothing * uniform_distribution

        ious = []
        exist_positive = False
        for i in range(len(grid_xy)):
            # Dim(anchors, xywh)
            anchors_xywh = np.zeros((3, 4), dtype=np.float32)
            anchors_xywh[:, 0:2] = xywh[0:2]
            anchors_xywh[:, 2:4] = anchors_ratio[i]
            iou = train.bbox_iou(xywh, anchors_xywh)
            ious.append(iou)
            iou_mask = iou > 0.3

            if np.any(iou_mask):
                xy_grid = xywh[0:2] * (
                    grid_size[i][1],
                    grid_size[i][0],
                )
                xy_index = np.floor(xy_grid)

                exist_positive = True
                for j, mask in enumerate(iou_mask):
                    if mask:
                        _x, _y = int(xy_index[0]), int(xy_index[1])
                        ground_truth[i][0, _y, _x, j, 0:4] = xywh
                        ground_truth[i][0, _y, _x, j, 4:5] = 1.0
                        ground_truth[i][0, _y, _x, j, 5:] = smooth_onehot

        if not exist_positive:
            index = np.argmax(np.array(ious))
            i = index // 3
            j = index % 3

            xy_grid = xywh[0:2] * (
                grid_size[i][1],
                grid_size[i][0],
            )
            xy_index = np.floor(xy_grid)

            _x, _y = int(xy_index[0]), int(xy_index[1])
            ground_truth[i][0, _y, _x, j, 0:4] = xywh
            ground_truth[i][0, _y, _x, j, 4:5] = 1.0
            ground_truth[i][0, _y, _x, j, 5:] = smooth_onehot

    return ground_truth


def resize_image(
    image,
    ground_truth
):
    image = tf.cast(image, tf.float32) / 255.0
    height, width, _ = image.shape

    if width / height >= input_size[0] / input_size[1]:
        scale = input_size[0] / width
    else:
        scale = input_size[1] / height

    # Resize
    if scale != 1:
        width = int(round(width * scale))
        height = int(round(height * scale))
        padded_image = tf.image.resize_with_pad(
            image, input_size[1], input_size[0])
    else:
        padded_image = np.copy(image)

    # Resize ground truth
    dw = input_size[0] - width
    dh = input_size[1] - height

    ground_truth = np.copy(ground_truth)

    if dw > dh:
        scale = width / input_size[0]
        ground_truth[:, 0] = scale * (ground_truth[:, 0] - 0.5) + 0.5
        ground_truth[:, 2] = scale * ground_truth[:, 2]
    elif dw < dh:
        scale = height / input_size[1]
        ground_truth[:, 1] = scale * (ground_truth[:, 1] - 0.5) + 0.5
        ground_truth[:, 3] = scale * ground_truth[:, 3]

    return padded_image, ground_truth


def exclude_classes(modified_bboxes):
    return modified_bboxes[np.isin(modified_bboxes[:, -1], included_classes_idxs)]

@tf.function
def coco_to_yolo(features):
    objects = features["objects"]
    bboxes: tf.Tensor = objects["bbox"]
    labels: tf.Tensor = objects["label"]

    x_center = tf.math.reduce_mean(
        tf.concat(
            [
                tf.reshape(bboxes[:, 3], (tf.shape(bboxes)[0], 1)),
                tf.reshape(bboxes[:, 1], (tf.shape(bboxes)[0], 1))
            ],
            axis=1
        ),
        axis=1,
        keepdims=True
    )
    y_center = tf.math.reduce_mean(
        tf.concat(
            [
                tf.reshape(bboxes[:, 2], (tf.shape(bboxes)[0], 1)),
                tf.reshape(bboxes[:, 0], (tf.shape(bboxes)[0], 1))
            ],
            axis=1
        ),
        axis=1,
        keepdims=True
    )

    width = tf.subtract(
        tf.reshape(bboxes[:, 3], (tf.shape(bboxes)[0], 1)),
        tf.reshape(bboxes[:, 1], (tf.shape(bboxes)[0], 1))
    )

    height = tf.subtract(
        tf.reshape(bboxes[:, 2], (tf.shape(bboxes)[0], 1)),
        tf.reshape(bboxes[:, 0], (tf.shape(bboxes)[0], 1))
    )

    labels = tf.reshape(labels, (tf.shape(labels)[0], 1))
    position = tf.concat([x_center, y_center, width, height], axis=1)
    modified_bboxes = tf.concat(
        [position, tf.cast(labels, tf.float32)], axis=1)

    # Filters bboxes to included classes
    modified_bboxes = tf.numpy_function(
        func=exclude_classes,
        inp=[modified_bboxes],
        Tout=tf.float32
    )

    # Random chance to remove pictures with only people in them
    is_all_people = modified_bboxes[:, 4] == tf.zeros(
        tf.shape(modified_bboxes[:, 4]))

    if tf.reduce_all(is_all_people):
        tf.print("is all people")
        if tf.random.uniform(shape=()) < tf.constant(chance_to_remove_person_only_images):
            tf.print("removing")
            modified_bboxes = tf.reshape(tf.convert_to_tensor(()), (0, 5))


    image = features["image"]
    image, modified_bboxes = tf.numpy_function(
        func=resize_image,
        inp=[image, modified_bboxes],
        Tout=[tf.float32, tf.float32]
    )

    ground_truth = tf.numpy_function(
        func=bboxes_to_ground_truth,
        inp=[modified_bboxes],
        Tout=[tf.float32 for _size in grid_size]
    )

    image.set_shape([input_size[0], input_size[1], 3])

    ground_truth[0].set_shape((1, 52, 52, 3, 85))
    ground_truth[1].set_shape((1, 26, 26, 3, 85))
    ground_truth[2].set_shape((1, 13, 13, 3, 85))

    return (image, (ground_truth[0], ground_truth[1], ground_truth[2]))


def filter_fn(image, ground_truth):
    if(tf.size(ground_truth[0]) == 0):
        return False
    return True


for example in ds_train.skip(5).take(1):

    mapped = coco_to_yolo(example)
    image = mapped[0]
    ground_truth = mapped[1]
    # for gt in ground_truth:
    # print(tf.shape(gt))
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()

ds_train = ds_train.map(coco_to_yolo, num_parallel_calls=tf.data.experimental.AUTOTUNE).filter(
    filter_fn).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

ds_val = ds_val.map(coco_to_yolo, num_parallel_calls=tf.data.experimental.AUTOTUNE).filter(
    filter_fn).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

epochs = 400
lr = 1e-4


def lr_scheduler(epoch):
    if epoch < int(epochs * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1
    return lr * 0.01


backend.clear_session()
inputs = layers.Input([input_size[0], input_size[1], 3])
yolo = YOLOv4(
    anchors=anchors,
    num_classes=num_classes,
    xyscales=xyscales,
    kernel_regularizer=regularizers.l2(0.0005)
)
yolo(inputs)

optimizer = optimizers.Adam(learning_rate=lr)
loss_iou_type = "ciou"
loss_verbose = 0

yolo.compile(
    optimizer=optimizer,
    loss=train.YOLOv4Loss(
        batch_size=batch_size,
        iou_type=loss_iou_type,
        verbose=loss_verbose
    )
)

# print("Tensorboard Version: ", tensorboard.__version__)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=logdir, histogram_freq=10)

callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    tensorboard_callback
]

steps_per_epoch = 100
validation_steps = 50
validation_freq = 5

yolo.summary()

# verbose = 0
yolo.fit(
    ds_train,
    epochs=epochs,
    # verbose=verbose,
    callbacks=callbacks,
    validation_data=ds_val,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_freq=validation_freq
)
