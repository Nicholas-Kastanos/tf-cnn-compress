import numpy as np
import tensorlfow as tf
import functional.utils as utils
import functional.common as common
import functional.backbone as backbone
from functional.config import cfg

def panet(x, NUM_CLASS, use_asymetric_conv=False, activate_type="leaky"):
    route_1, route_2, conv = x

    route = conv
    conv = common.darknet_conv(conv, filters=256, kernel_size=1)
    conv = common.upsample(conv)
    route_2 = common.darknet_conv(route_2, filters=256, kernel_size=1)
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.darknet_conv(conv, filters=256, kernel_size=1)
    conv = common.darknet_conv(conv, filters=512, kernel_size=3)
    conv = common.darknet_conv(conv, filters=256, kernel_size=1)
    conv = common.darknet_conv(conv, filters=512, kernel_size=3)
    conv = common.darknet_conv(conv, filters=256, kernel_size=1)

    route_2 = conv
    conv = common.darknet_conv(conv, filters=128, kernel_size=1)
    conv = common.upsample(conv)
    route_1 = common.darknet_conv(route_1, filters=128, kernel_size=1)
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.darknet_conv(conv, filters=128, kernel_size=1)
    conv = common.darknet_conv(conv, filters=256, kernel_size=3)
    conv = common.darknet_conv(conv, filters=128, kernel_size=1)
    conv = common.darknet_conv(conv, filters=256, kernel_size=3)
    conv = common.darknet_conv(conv, filters=128, kernel_size=1)

    route_1 = conv
    conv = common.darknet_conv(conv, filters=256, kernel_size=3)
    conv_sbbox = common.darknet_conv(conv, filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)

    conv = common.darknet_conv(route_1, filters=256, kernel_size=3, downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.darknet_conv(conv, filters=256, kernel_size=1)
    conv = common.darknet_conv(conv, filters=512, kernel_size=3)
    conv = common.darknet_conv(conv, filters=256, kernel_size=1)
    conv = common.darknet_conv(conv, filters=512, kernel_size=3)
    conv = common.darknet_conv(conv, filters=256, kernel_size=1)

    route_2 = conv
    conv = common.darknet_conv(conv, filters=512, kernel_size=3)
    conv_mbbox = common.darknet_conv(conv, filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)

    conv = common.darknet_conv(route_2, filters=512, kernel_size=3, downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.darknet_conv(conv, filters=512, kernel_size=1)
    conv = common.darknet_conv(conv, filters=1024, kernel_size=3)
    conv = common.darknet_conv(conv, filters=512, kernel_size=1)
    conv = common.darknet_conv(conv, filters=1024, kernel_size=3)
    conv = common.darknet_conv(conv, filters=512, kernel_size=1)

    conv = common.darknet_conv(conv, filters=1024, kernel_size=3)
    conv_lbbox = common.darknet_conv(conv, filters=3 * (NUM_CLASS + 5), kernel_size=1, activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]
