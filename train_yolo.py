import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend, layers, optimizers, regularizers, callbacks

import numpy as np

import src.media as media
import src.predict as predict
import src.train as train
import src.dataset as dataset

from src.yolov4.yolov4 import YOLOv4

anchors = np.array([
    [[12, 16], [19, 36], [40, 28]],
    [[36, 75], [76, 55], [72, 146]],
    [[142, 110], [192, 243], [459, 401]],
]).astype(np.float32).reshape(3, 3, 2)
strides = np.array([8, 16, 32])
xyscales = np.array([1.2, 1.1, 1.05])
input_size = (608, 416)

class_names_path = os.path.join(os.getcwd(), "dataset", "coco.names")
classes = media.read_classes_names(class_names_path)

train_data_set = load_dataset(
    os.path.join(os.getcwd(), "dataset", "train2017.txt"),
    image_path_prefix="/content/train2017",
    label_smoothing=0.05
)

val_data_set = load_dataset(
    os.path.join(os.getcwd(), "dataset", "val2017.txt"),
    image_path_prefix="/content/val2017",
    label_smoothing=0.05
)

batch_size = 32


backend.clear_session()
inputs = layers.Input([input_size[1], input_size[0], 3])
yolo = YOLOv4(
    anchors=anchors,
    num_classes=len(classes),
    xyscales=xyscales,
    kernel_regularizer=regularizers.l2(0.0005)
)
model = keras.Sequential()
model.add(inputs)
model.add(yolo)

lr=1e-4

optimizer = optimizers.Adam(learning_rate=lr)
loss_iou_type = "ciou"
loss_verbose = 1

model.compile(
    optimizer=optimizer,
    loss=train.YOLOv4Loss(
        batch_size=batch_size,
        iou_type=loss_iou_type,
        verbose=loss_verbose
    )
)

data_set = train_data_set
epochs = 400
verbose = 2
callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(
        log_dir=os.path.join(os.getcwd(), "logs")
    )
]
validation_data = val_data_set
initial_epoch = 0
steps_per_epoch = 100
validation_steps = 50
validation_freq = 5
model.fit(
    data_set,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    callbacks=callbacks,
    validation_data=validation_data,
    initial_epoch=initial_epoch,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    validation_freq=validation_freq
)


def lr_scheduler(epoch):
    if epoch < int(epochs * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1
    return lr * 0.01


def resize_image(image, ground_truth=None):
    """
    @param image:        Dim(height, width, channels)
    @param ground_truth: [[center_x, center_y, w, h, class_id], ...]

    @return resized_image or (resized_image, resized_ground_truth)

    Usage:
        image = yolo.resize_image(image)
        image, ground_truth = yolo.resize_image(image, ground_truth)
    """
    return media.resize_image(
        image, target_size=input_size, ground_truth=ground_truth
    )


def candidates_to_pred_bboxes(
    candidates,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.25,
):
    """
    @param candidates: Dim(-1, (x, y, w, h, conf, prob_0, prob_1, ...))

    @return Dim(-1, (x, y, w, h, class_id, probability))
    """
    return predict.candidates_to_pred_bboxes(
        candidates,
        input_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )


def fit_pred_bboxes_to_original(pred_bboxes, original_shape):
    """
    @param pred_bboxes:    Dim(-1, (x, y, w, h, class_id, probability))
    @param original_shape: (height, width, channels)
    """
    # pylint: disable=no-self-use
    return predict.fit_pred_bboxes_to_original(
        pred_bboxes, input_size, original_shape
    )


def draw_bboxes(image, bboxes):
    """
    @parma image:  Dim(height, width, channel)
    @param bboxes: (candidates, 4) or (candidates, 5)
            [[center_x, center_y, w, h, class_id], ...]
            [[center_x, center_y, w, h, class_id, propability], ...]

    @return drawn_image

    Usage:
        image = yolo.draw_bboxes(image, bboxes)
    """
    return media.draw_bboxes(image, bboxes, classes)


def load_dataset(
    dataset_path,
    dataset_type="converted_coco",
    label_smoothing=0.1,
    image_path_prefix=None,
    training=True,
):
    return dataset.Dataset(
        anchors=anchors,
        batch_size=batch_size,
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        data_augmentation=training,
        input_size=input_size,
        label_smoothing=label_smoothing,
        num_classes=len(classes),
        image_path_prefix=image_path_prefix,
        strides=strides,
        xyscales=xyscales,
    )
