#!/usr/bin/env python
# coding: utf-8

import os
from datetime import datetime
import argparse
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
import tensorboard
import tensorflow as tf
import tensorflow_datasets as tfds

# import src.predict as predict
import src.media as media
# import src.dataset as dataset
import src.train as train
from src.functional.config import cfg
from src.functional.yolov4 import YOLOv4
import yolo

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--folder_name', default=datetime.now().strftime("%Y%m%d-%H%M%S"))
parser.add_argument('-q', '--quantized_training', action='store_true', default=False)
parser.add_argument('-a', '--asymetric', action='store_true', default=False)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-i', '--input_size', type=int, default=416)
parser.add_argument('--data_dir', default=os.path.join('/', 'media', 'nicholas', 'Data', 'nicho', 'Documents', 'tensorflow_datasets'))
args = parser.parse_args()

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)

try:
    import google.colab  # pylint: disable=no-name-in-module,import-error
    IN_COLAB = True
except:
    IN_COLAB = False

data_dir = args.data_dir
epochs = args.epochs
batch_size = args.batch_size
input_size = (args.input_size, args.input_size)
quantized_training = args.quantized_training
use_asymetric_conv = args.asymetric
folder_name = args.folder_name

use_yolo=True
if use_yolo:
    (ds_train, ds_val), ds_info = tfds.load(
        'yolo',
        split=['train', 'validation'],
        with_info=True,
        data_dir=data_dir
    )
    num_classes=20
else:
    (ds_train, ds_val), ds_info = tfds.load(
        'coco/2017',
        split=['train', 'validation'],
        with_info=True,
        data_dir=data_dir,
        try_gcs=IN_COLAB
    )
    included_classes = ds_info.features["objects"]["label"].names[:20]
    num_classes = len(included_classes)
    class_dict = dict(
        zip(
            range(num_classes),
            included_classes
        )
    )
    included_classes_idxs = np.asarray(list(class_dict.keys()))

model_dir = 'models/' + folder_name
if not os.path.isdir(model_dir):
   os.makedirs(model_dir)

log_dir = "logs/fit/" + folder_name
if not os.path.isdir(log_dir):
   os.makedirs(log_dir)

checkpoint_dir = model_dir + '/checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

copyfile('./train_coco.py', model_dir + '/train_coco.py')

lr = cfg.TRAIN.LR
def lr_scheduler(epoch):
    if epoch < int(epochs * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1
    return lr * 0.01

anchors = np.asarray(cfg.YOLO.ANCHORS).astype(np.float32).reshape(3, 3, 2)
strides = np.asarray(cfg.YOLO.STRIDES)
xyscales = np.asarray(cfg.YOLO.XYSCALE)
label_smoothing = cfg.YOLO.LABEL_SMOOTHING
anchors_ratio = anchors / input_size[0]
grid_size = (input_size[1], input_size[0]) // np.stack(
    (strides, strides), axis=1
)


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

chance_to_remove_person_only_images = 0.5 # Attempt to balance the dataset


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


def resize_image(image, ground_truth):
    image = tf.cast(image, tf.float32) / 255.0
    height, width, _ = image.shape

    if width / height >= input_size[0] / input_size[1]:
        scale = input_size[0] / width
    else:
        scale = input_size[1] / height

    # Resize

    width = int(round(width * scale))
    height = int(round(height * scale))
    padded_image = tf.image.resize_with_pad(
        image, input_size[1], input_size[0])


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

    # tf.print(tf.shape(bboxes), tf.shape(labels))
    # plt.imshow(media.draw_bboxes(features["image"], modified_bboxes, dict(
    #     zip(
    #         range(len(ds_info.features["objects"]["label"].names)),
    #         ds_info.features["objects"]["label"].names
    #     )
    # )))
    # plt.show()

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
        if tf.random.uniform(shape=()) < tf.constant(chance_to_remove_person_only_images):
            modified_bboxes = tf.reshape(tf.convert_to_tensor(()), (0, 5))

    # Remove invalid images
    if tf.shape(modified_bboxes)[0] == 0:
        image = tf.constant(np.nan, shape=(input_size[0], input_size[1], 3))
        # tf.print("Removed")
    else:
        image = features["image"]
        # plt.imshow(media.draw_bboxes(image, modified_bboxes, class_dict))
        # plt.show()
        image, modified_bboxes = tf.numpy_function(
            func=resize_image,
            inp=[image, modified_bboxes],
            Tout=[tf.float32, tf.float32]
        )
        # plt.imshow(media.draw_bboxes(image, modified_bboxes, class_dict))
        # plt.show()

    # Convert from xywhc to yolo ground truth
    ground_truth = tf.numpy_function(
        func=bboxes_to_ground_truth,
        inp=[modified_bboxes],
        Tout=[tf.float32 for _size in grid_size]
    )

    image.set_shape([input_size[0], input_size[1], 3])

    for i, _size in enumerate(grid_size):
        ground_truth[i].set_shape((1, _size[0], _size[1], 3, num_classes + 5))

    return (image, (ground_truth[0], ground_truth[1], ground_truth[2]))

def filter_fn(image, ground_truth):
    image_ok = tf.reduce_all(tf.math.is_finite(image))
    gt_0_ok = tf.reduce_all(tf.math.is_finite(ground_truth[0]))
    gt_1_ok = tf.reduce_all(tf.math.is_finite(ground_truth[1]))
    gt_2_ok = tf.reduce_all(tf.math.is_finite(ground_truth[2]))
    return tf.reduce_all([image_ok, gt_0_ok, gt_1_ok, gt_2_ok])


# for example in ds_val.take(10):
#     mapped = coco_to_yolo(example)

def yolo_to_yolo(features):
    image = tf.cast(features['image'] / 255 , dtype=tf.float32)
    gt_s = tf.cast(features['gt_s'] / 255 , dtype=tf.float32)
    gt_m = tf.cast(features['gt_m'] / 255 , dtype=tf.float32)
    gt_l = tf.cast(features['gt_l'] / 255 , dtype=tf.float32)
    return (image, (gt_s, gt_m, gt_l))

if use_yolo:
    ds_train = ds_train.map(yolo_to_yolo, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(yolo_to_yolo, num_parallel_calls=tf.data.experimental.AUTOTUNE)
else:
    ds_train = ds_train.map(coco_to_yolo, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .filter(filter_fn)
    ds_val = ds_val.map(coco_to_yolo, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .filter(filter_fn) \

ds_train = ds_train.shuffle(1000) \
    .batch(batch_size) \
    .cache() \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_val = ds_val.batch(batch_size) \
    .cache() \
    .prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()

def get_compiled_model():
    input_layer = tf.keras.layers.Input([input_size[0], input_size[1], 3])
    output_layer = YOLOv4(input_layer, grid_size, num_classes, strides, anchors, xyscales, use_asymetric_conv, input_size[0])
    yolo = tf.keras.Model(input_layer, output_layer)

    if quantized_training:
        import tensorflow_model_optimization as tfmot

        def apply_quantization(layer):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            return layer

        yolo = tf.keras.models.clone_model(yolo, clone_function=apply_quantization)

        with tfmot.quantization.keras.quantize_scope({}):
            # Use `quantize_apply` to actually make the model quantization aware.
            yolo = tfmot.quantization.keras.quantize_apply(yolo)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
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
    return yolo

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint, custom_objects={'YOLOv4Loss': train.YOLOv4Loss})
    print("Creating a new model")
    return get_compiled_model()

yolo = make_or_restore_model()


callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        profile_batch=(2, 22)
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + 'model_{epoch}',
        save_freq='epoch',
        period=5
    )
]

steps_per_epoch = 100
validation_steps = 50
validation_freq = 5

# yolo.summary()

# tf.keras.utils.plot_model(yolo, show_shapes=True, show_layer_names=True, to_file='model.png')

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

yolo.save('./models/' + folder_name + '/model')