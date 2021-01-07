"""yolo dataset."""

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import numpy as np
import os
from src.functional.config import cfg
import src.train as train

# TODO(yolo): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(yolo): BibTeX citation
_CITATION = """
"""


class Yolo(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for yolo dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, **kwargs):
        super(Yolo, self).__init__(**kwargs)
        

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        (ds_train, ds_val), ds_info = tfds.load(
            'coco/2017',
            split=['train', 'validation'],
            with_info=True,
            data_dir=os.path.join(
                '/', 'media', 'nicholas', 'Data', 'nicho', 'Documents', 'tensorflow_datasets')
        )
        self.coco_ds_train = ds_train
        self.coco_ds_val = ds_val
        self.coco_ds_info = ds_info

        self.chance_to_remove_person_only_images = 0.5  # Attempt to balance the dataset
        self.anchors = np.asarray(cfg.YOLO.ANCHORS).astype(
            np.float32).reshape(3, 3, 2)
        self.strides = np.asarray(cfg.YOLO.STRIDES)
        self.xyscales = np.asarray(cfg.YOLO.XYSCALE)
        self.label_smoothing = cfg.YOLO.LABEL_SMOOTHING

        self.input_size = (cfg.YOLO.INPUT_SIZE, cfg.YOLO.INPUT_SIZE, 3)

        self.anchors_ratio = self.anchors / self.input_size[0]
        self.grid_size = (self.input_size[1], self.input_size[0]) // np.stack(
            (self.strides, self.strides), axis=1
        )

        self.num_classes = 20

        self.included_classes = ds_info.features["objects"]["label"].names[:self.num_classes]
        self.class_dict = dict(
            zip(
                range(self.num_classes),
                self.included_classes
            )
        )
        self.included_classes_idxs = np.asarray(list(self.class_dict.keys()))

        self.grid_xy = [
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
            for _size in self.grid_size  # (height, width)
        ]

        
        # TODO(yolo): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=self.input_size, dtype=tf.uint8),
                'xywhc': tfds.features.Tensor(shape=(None, 5), dtype=tf.float32),
                'gt_s': tfds.features.Tensor(shape=(1, self.grid_size[0][0], self.grid_size[0][0], 3, self.num_classes + 5), dtype=tf.float32),
                'gt_m': tfds.features.Tensor(shape=(1, self.grid_size[1][0], self.grid_size[1][1], 3, self.num_classes + 5), dtype=tf.float32),
                'gt_l': tfds.features.Tensor(shape=(1, self.grid_size[2][0], self.grid_size[2][1], 3, self.num_classes + 5), dtype=tf.float32),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # e.g. ('image', 'label')
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # TODO(yolo): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(self.coco_ds_train),
            'validation': self._generate_examples(self.coco_ds_val)
        }

    def _generate_examples(self, dataset: tf.data.Dataset):
        """Yields examples."""
        # TODO(yolo): Yields (key, example) tuples from the dataset
        dataset = dataset.map(self._coco_to_yolo).filter(self._filter_fn)
        for example in dataset:
            yield str(example['image_id'].numpy()), {
                'image': tf.cast(example['image'] * 255, tf.uint8).numpy(),
                'xywhc': example['xywhc'].numpy(),
                'gt_s': example['gt_s'].numpy(),
                'gt_m': example['gt_m'].numpy(),
                'gt_l': example['gt_l'].numpy(),
            }

    def _bboxes_to_ground_truth(self, bboxes):
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
                    5 + self.num_classes,
                ),
            dtype=np.float32,
          )
          for _size in self.grid_size
        ]

        for i, _grid in enumerate(self.grid_xy):
            ground_truth[i][..., 0:2] = _grid

        for bbox in bboxes:
            # [b_x, b_y, b_w, b_h, class_id]
            xywh = np.array(bbox[:4], dtype=np.float32)
            class_id = int(bbox[4])

            # smooth_onehot = [0.xx, ... , 1-(0.xx*(n-1)), 0.xx, ...]
            onehot = np.zeros(self.num_classes, dtype=np.float32)
            onehot[class_id] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes, dtype=np.float32
            )
            smooth_onehot = (
                1 - self.label_smoothing
            ) * onehot + self.label_smoothing * uniform_distribution

            ious = []
            exist_positive = False
            for i in range(len(self.grid_xy)):
                # Dim(anchors, xywh)
                anchors_xywh = np.zeros((3, 4), dtype=np.float32)
                anchors_xywh[:, 0:2] = xywh[0:2]
                anchors_xywh[:, 2:4] = self.anchors_ratio[i]
                iou = train.bbox_iou(xywh, anchors_xywh)
                ious.append(iou)
                iou_mask = iou > 0.3

                if np.any(iou_mask):
                    xy_grid = xywh[0:2] * (
                        self.grid_size[i][1],
                        self.grid_size[i][0],
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
                    self.grid_size[i][1],
                    self.grid_size[i][0],
                )
                xy_index = np.floor(xy_grid)

                _x, _y = int(xy_index[0]), int(xy_index[1])
                ground_truth[i][0, _y, _x, j, 0:4] = xywh
                ground_truth[i][0, _y, _x, j, 4:5] = 1.0
                ground_truth[i][0, _y, _x, j, 5:] = smooth_onehot

        return ground_truth


    def _resize_image(self, image, ground_truth):
        image = tf.cast(image, tf.float32) / 255.0
        height, width, _ = image.shape

        if width / height >= self.input_size[0] / self.input_size[1]:
            scale = self.input_size[0] / width
        else:
            scale = self.input_size[1] / height

        # Resize

        width = int(round(width * scale))
        height = int(round(height * scale))
        padded_image = tf.image.resize_with_pad(
            image, self.input_size[1], self.input_size[0])


        # Resize ground truth
        dw = self.input_size[0] - width
        dh = self.input_size[1] - height

        ground_truth = np.copy(ground_truth)

        if dw > dh:
            scale = width / self.input_size[0]
            ground_truth[:, 0] = scale * (ground_truth[:, 0] - 0.5) + 0.5
            ground_truth[:, 2] = scale * ground_truth[:, 2]
        elif dw < dh:
            scale = height / self.input_size[1]
            ground_truth[:, 1] = scale * (ground_truth[:, 1] - 0.5) + 0.5
            ground_truth[:, 3] = scale * ground_truth[:, 3]

        return padded_image, ground_truth


    def _exclude_classes(self, modified_bboxes):
        return modified_bboxes[np.isin(modified_bboxes[:, -1], self.included_classes_idxs)]

    @tf.function
    def _coco_to_yolo(self, features):
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
            func=self._exclude_classes,
            inp=[modified_bboxes],
            Tout=tf.float32
        )

        # Random chance to remove pictures with only people in them
        is_all_people = modified_bboxes[:, 4] == tf.zeros(
            tf.shape(modified_bboxes[:, 4]))

        if tf.reduce_all(is_all_people):
            if tf.random.uniform(shape=()) < tf.constant(self.chance_to_remove_person_only_images):
                modified_bboxes = tf.reshape(tf.convert_to_tensor(()), (0, 5))

        # Remove invalid images
        if tf.shape(modified_bboxes)[0] == 0:
            image = tf.constant(np.nan, shape=self.input_size, dtype=tf.float32)
            # tf.print("Removed")
        else:
            image = features["image"]
            # plt.imshow(media.draw_bboxes(image, modified_bboxes, class_dict))
            # plt.show()
            image, modified_bboxes = tf.numpy_function(
                func=self._resize_image,
                inp=[image, modified_bboxes],
                Tout=[tf.float32, tf.float32]
            )
            # plt.imshow(media.draw_bboxes(image, modified_bboxes, class_dict))
            # plt.show()

        # Convert from xywhc to yolo ground truth
        ground_truth = tf.numpy_function(
            func=self._bboxes_to_ground_truth,
            inp=[modified_bboxes],
            Tout=[tf.float32 for _size in self.grid_size]
        )

        image.set_shape([self.input_size[0], self.input_size[1], 3])

        for i, _size in enumerate(self.grid_size):
            ground_truth[i].set_shape((1, _size[0], _size[1], 3, self.num_classes + 5))

        modified_bboxes.set_shape([None, 5])

        return {
            'image_id': features['image/id'],
            'image': image,
            'xywhc': modified_bboxes,
            'gt_s': ground_truth[0],
            'gt_m': ground_truth[1],
            'gt_l': ground_truth[2]
        }

    def _filter_fn(self, d: dict):
        image_ok = tf.reduce_all(tf.math.is_finite(d['image']))
        bboxes_ok = tf.reduce_all(tf.math.is_finite(d['xywhc']))
        gt_0_ok = tf.reduce_all(tf.math.is_finite(d['gt_s']))
        gt_1_ok = tf.reduce_all(tf.math.is_finite(d['gt_m']))
        gt_2_ok = tf.reduce_all(tf.math.is_finite(d['gt_l']))
        return tf.reduce_all([image_ok, bboxes_ok, gt_0_ok, gt_1_ok, gt_2_ok])
