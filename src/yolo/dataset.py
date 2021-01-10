"""
MIT License

Copyright (c) 2019 YangYun
Copyright (c) 2020 Việt Hùng
Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from os import path
import random
from typing import Union

import cv2
import numpy as np

from . import train
from . import media

    def _next_data(self):
        for _ in range(5):
            _dataset = self.dataset[self.count]
            self.count += 1
            if self.count == len(self.dataset):
                if self.data_augmentation:
                    np.random.shuffle(self.dataset)
                self.count = 0

            ret = self.load_image_then_resize(_dataset)
            if ret is not None:
                return ret

        raise FileNotFoundError("Failed to find images")

    def _next_random_augmentation_data(self):
        if random.random() < 0.2:
            _prob = random.random()
            if _prob < 0.25:
                _dataset = cut_out(self._next_data())
            elif _prob < 0.5:
                _dataset = mix_up(self._next_data(), self._next_data())
            else:
                _dataset = mosaic(*[self._next_data() for _ in range(4)])
        else:
            _dataset = self._next_data()

        return _dataset

    def __iter__(self):
        self.count = 0
        if self.data_augmentation:
            np.random.shuffle(self.dataset)
        return self

    def __next__(self):
        """
        @return image, ground_truth
            ground_truth == (s_truth, m_truth, l_truth) or (s_truth, l_truth)
        """
        if self.batch_size > 1:
            batch_x = []
            _batch_y = [[] for _ in range(len(self.grid_size))]
            for _ in range(self.batch_size):
                if self.data_augmentation:
                    _dataset = self._next_random_augmentation_data()
                else:
                    _dataset = self._next_data()
                x = _dataset[0]
                y = self.bboxes_to_ground_truth(_dataset[1])
                batch_x.append(x)
                for i, _y in enumerate(y):
                    _batch_y[i].append(_y)
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = [np.concatenate(b_y, axis=0) for b_y in _batch_y]
        else:
            if self.data_augmentation:
                _dataset = self._next_random_augmentation_data()
            else:
                _dataset = self._next_data()
            batch_x = _dataset[0]
            batch_y = self.bboxes_to_ground_truth(_dataset[1])

        # batch_x == Dim(batch, input_size, input_size, channels)
        # batch_y[0] == Dim(batch, grid_size, grid_size, anchors, bboxes)
        return batch_x, batch_y

    def __len__(self):
        return len(self.dataset)


def cut_out(dataset):
    """
    @parma `dataset`: [image(float), bboxes]
            bboxes = [image_path, [[x, y, w, h, class_id], ...]]
    """
    _size = dataset[0].shape[1:3]  # height, width
    for bbox in dataset[1]:
        if random.random() < 0.5:
            _pixel_bbox = [
                int(pos * _size[(i + 1) % 2]) for i, pos in enumerate(bbox[0:4])
            ]
            _x_min = _pixel_bbox[0] - (_pixel_bbox[2] // 2)
            _y_min = _pixel_bbox[1] - (_pixel_bbox[3] // 2)
            _cut_out_width = _pixel_bbox[2] // 4
            _cut_out_height = _pixel_bbox[3] // 4
            _x_offset = (
                int((_pixel_bbox[2] - _cut_out_width) * random.random())
                + _x_min
            )
            _y_offset = (
                int((_pixel_bbox[3] - _cut_out_height) * random.random())
                + _y_min
            )
            dataset[0][
                :,
                _y_offset : _y_offset + _cut_out_height,
                _x_offset : _x_offset + _cut_out_width,
                :,
            ] = 0.5

    return dataset


def mix_up(dataset0, dataset1, alpha=0.2):
    return (
        (dataset0[0] * alpha + dataset1[0] * (1 - alpha)),
        (np.concatenate((dataset0[1], dataset1[1]), axis=0)),
    )


def mosaic(dataset0, dataset1, dataset2, dataset3):
    size = dataset0[0].shape[1:3]  # height, width
    image = np.empty((1, size[0], size[1], 3))
    bboxes = []

    partition_x = int((random.random() * 0.6 + 0.2) * size[1])
    partition_y = int((random.random() * 0.6 + 0.2) * size[0])

    x_offset = [0, partition_x, 0, partition_x]
    y_offset = [0, 0, partition_y, partition_y]

    left = [
        (size[1] - partition_x) // 2,
        partition_x // 2,
        (size[1] - partition_x) // 2,
        partition_x // 2,
    ]
    right = [
        left[0] + partition_x,
        left[1] + size[1] - partition_x,
        left[2] + partition_x,
        left[3] + size[1] - partition_x,
    ]
    top = [
        (size[0] - partition_y) // 2,
        (size[0] - partition_y) // 2,
        partition_y // 2,
        partition_y // 2,
    ]
    down = [
        top[0] + partition_y,
        top[1] + partition_y,
        top[2] + size[0] - partition_y,
        top[3] + size[0] - partition_y,
    ]

    image[:, :partition_y, :partition_x, :] = dataset0[0][
        :,
        top[0] : down[0],
        left[0] : right[0],
        :,
    ]
    image[:, :partition_y, partition_x:, :] = dataset1[0][
        :,
        top[1] : down[1],
        left[1] : right[1],
        :,
    ]
    image[:, partition_y:, :partition_x, :] = dataset2[0][
        :,
        top[2] : down[2],
        left[2] : right[2],
        :,
    ]
    image[:, partition_y:, partition_x:, :] = dataset3[0][
        :,
        top[3] : down[3],
        left[3] : right[3],
        :,
    ]

    for i, _bboxes in enumerate(
        (dataset0[1], dataset1[1], dataset2[1], dataset3[1])
    ):
        for bbox in _bboxes:
            pixel_bbox = [
                int(pos * size[(i + 1) % 2]) for i, pos in enumerate(bbox[0:4])
            ]
            x_min = int(pixel_bbox[0] - pixel_bbox[2] // 2)
            y_min = int(pixel_bbox[1] - pixel_bbox[3] // 2)
            x_max = int(pixel_bbox[0] + pixel_bbox[2] // 2)
            y_max = int(pixel_bbox[1] + pixel_bbox[3] // 2)

            class_id = bbox[4]

            if x_min > right[i]:
                continue
            if y_min > down[i]:
                continue
            if x_max < left[i]:
                continue
            if y_max < top[i]:
                continue

            if x_max > right[i]:
                x_max = right[i]
            if y_max > down[i]:
                y_max = down[i]
            if x_min < left[i]:
                x_min = left[i]
            if y_min < top[i]:
                y_min = top[i]

            x_min -= left[i]
            x_max -= left[i]
            y_min -= top[i]
            y_max -= top[i]

            if x_min + 3 > x_max:
                continue

            if y_min + 3 > y_max:
                continue

            bboxes.append(
                np.array(
                    [
                        [
                            ((x_min + x_max) / 2 + x_offset[i]) / size[1],
                            ((y_min + y_max) / 2 + y_offset[i]) / size[0],
                            (x_max - x_min) / size[1],
                            (y_max - y_min) / size[0],
                            class_id,
                        ],
                    ]
                )
            )

    if len(bboxes) == 0:
        return dataset0

    return image, np.concatenate(bboxes, axis=0)
