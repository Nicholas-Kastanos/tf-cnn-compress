#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import os
# import urllib.request
# urllib.request.urlretrieve('https://github.com/Nicholas-Kastanos/tf-yolov4-compress/archive/main.zip','tf-yolov4-compress.zip')


# In[ ]:


# get_ipython().run_line_magic('rm', '-r sample_data')
# get_ipython().system('unzip tf-yolov4-compress')
# get_ipython().run_line_magic('cd', 'tf-yolov4-compress-main/')
# get_ipython().run_line_magic('pwd', '')


# In[ ]:


# get_ipython().run_line_magic('tensorflow_version', '2.x')
# print("Using TensorFlow version", tf.__version__)


# In[ ]:


# get_ipython().system('nvidia-smi')


# In[ ]:


# This wont run on Colab. There is not enough storage on the kernel
# get_ipython().system('mkdir -p dataset/archives/')
# get_ipython().system('curl http://images.cocodataset.org/zips/train2017.zip --output dataset/archives/train2017.zip')
# get_ipython().system('curl http://images.cocodataset.org/zips/val2017.zip --output dataset/archives/val2017.zip')
# get_ipython().system('unzip dataset/archives/train2017.zip')
# get_ipython().system('unzip dataset/archives/val2017.zip')


# In[1]:

from datetime import datetime
import tensorboard
from typing import Tuple
import matplotlib.pyplot as plt
from src.yolov4.yolov4 import YOLOv4
import src.dataset as dataset
import src.train as train
import src.predict as predict
import src.media as media
import numpy as np
from tensorflow_datasets.core.features import FeaturesDict, BBoxFeature
from tensorflow_datasets.core.dataset_info import DatasetInfo
import tensorflow_datasets as tfds
from tensorflow.keras import backend, layers, optimizers, regularizers, callbacks
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# class_names_path = os.path.join(os.getcwd(), "dataset", "coco.names")
# classes = media.read_classes_names(class_names_path)


# # In[7]:


# def load_dataset(
#     dataset_path,
#     dataset_type="converted_coco",
#     label_smoothing=0.1,
#     image_path_prefix=None,
#     training=True,
# ):
#     return dataset.Dataset(
#         anchors=anchors,
#         batch_size=batch_size,
#         dataset_path=dataset_path,
#         dataset_type=dataset_type,
#         data_augmentation=training,
#         input_size=input_size,
#         label_smoothing=label_smoothing,
#         num_classes=len(classes),
#         image_path_prefix=image_path_prefix,
#         strides=strides,
#         xyscales=xyscales,
#     )


# # In[8]:


# train_data_set = load_dataset(
#     os.path.join(os.getcwd(), "dataset", "train2017.txt"),
#     # image_path_prefix=os.path.join(os.getcwd(), 'archives', 'train2017.zip'),
#     image_path_prefix=os.path.join('/', 'media', 'nicholas', 'Data','nicho','Documents','ACS','L46','Project', 'train2017'),
#     label_smoothing=0.05
# )

# print(type(train_data_set.dataset))
# old_example = train_data_set.dataset[0]
# print(type(old_example))
# print(old_example)

# # In[9]:


# val_data_set = load_dataset(
#     os.path.join(os.getcwd(), "dataset", "val2017.txt"),
#     # image_path_prefix=os.path.join(os.getcwd(), 'dataset', 'archives', 'val2017.zip'),
#     image_path_prefix=os.path.join('/', 'media', 'nicholas', 'Data','nicho','Documents','ACS','L46','Project', 'val2017'),
#     label_smoothing=0.05
# )

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
input_size = (416, 416)

anchors_ratio = anchors / input_size[0]
batch_size = 32
grid_size = (input_size[1], input_size[0]) // np.stack(
    (strides, strides), axis=1
)
label_smoothing = 0.1
num_classes = ds_info.features["objects"]["label"].num_classes
class_dict = dict(
    zip(
        range(ds_info.features["objects"]["label"].num_classes),
        ds_info.features["objects"]["label"].names
    )
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


def bboxes_to_ground_truth(bboxes):
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

    return (image, [ground_truth[0], ground_truth[1], ground_truth[2]])


for example in ds_train.skip(5).take(1):

    mapped = coco_to_yolo(example)
    image = mapped[0]
    ground_truth = mapped[1]
    for gt in ground_truth:
        print(tf.shape(gt))
    print(image.shape)
    plt.imshow(image)
    plt.show()

ds_train = ds_train.map(
    coco_to_yolo, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(1000)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_val = ds_val.map(
    coco_to_yolo, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.batch(batch_size)
ds_val = ds_val.cache()
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

# # In[10]:


epochs = 1
lr = 1e-4


def lr_scheduler(epoch):
    if epoch < int(epochs * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1
    return lr * 0.01


# # # In[13]:


backend.clear_session()
inputs = layers.Input([input_size[0], input_size[1], 3])
yolo = YOLOv4(
    anchors=anchors,
    num_classes=num_classes,
    xyscales=xyscales,
    kernel_regularizer=regularizers.l2(0.0005)
)
yolo(inputs)


# # # In[14]:


optimizer = optimizers.Adam(learning_rate=lr)
loss_iou_type = "ciou"
loss_verbose = 1

yolo.compile(
    optimizer=optimizer,
    loss=train.YOLOv4Loss(
        batch_size=batch_size,
        iou_type=loss_iou_type,
        verbose=loss_verbose
    )
)

print("Tensorboard Version: ", tensorboard.__version__)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

verbose = 2
callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    tensorboard_callback
]
initial_epoch = 0
steps_per_epoch = 10  # 100
validation_steps = 5  # 50
validation_freq = 1  # 5


# # # In[ ]:

yolo.fit(
    ds_train,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_data=ds_val,
    # initial_epoch=initial_epoch,
    # steps_per_epoch=steps_per_epoch,
    # validation_steps=validation_steps,
    # validation_freq=validation_freq
)


# # # In[ ]:
