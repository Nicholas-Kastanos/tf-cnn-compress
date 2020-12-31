import tensorflow as tf
from tensorflow.keras import layers

from .common import DarknetConv


class PANet(layers.Layer):
    def __init__(
        self,
        num_classes: int,
        activation: str = "leaky",
        kernel_regularizer=None,
        use_asymetrical_conv=False
    ):
        super(PANet, self).__init__(name="PANet")
        self.conv_0 = DarknetConv(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.upSampling_0 = layers.UpSampling2D(interpolation="bilinear")
        self.conv_1 = DarknetConv(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.concat_0_1 = layers.Concatenate(axis=-1)

        self.conv_2 = DarknetConv(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_3 = DarknetConv(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_4 = DarknetConv(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_5 = DarknetConv(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_6 = DarknetConv(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )

        self.conv_7 = DarknetConv(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.upSampling_7 = layers.UpSampling2D(interpolation="bilinear")
        self.conv_8 = DarknetConv(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.concat_7_8 = layers.Concatenate(axis=-1)

        self.conv_9 = DarknetConv(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_10 = DarknetConv(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_11 = DarknetConv(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_12 = DarknetConv(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_13 = DarknetConv(
            filters=128,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )

        self.conv_14 = DarknetConv(
            filters=256,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_15 = DarknetConv(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )

        self.conv_16 = DarknetConv(
            filters=256,
            kernel_size=3,
            strides=2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.concat_6_16 = layers.Concatenate(axis=-1)

        self.conv_17 = DarknetConv(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_18 = DarknetConv(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_19 = DarknetConv(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_20 = DarknetConv(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_21 = DarknetConv(
            filters=256,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )

        self.conv_22 = DarknetConv(
            filters=512,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_23 = DarknetConv(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )

        self.conv_24 = DarknetConv(
            filters=512,
            kernel_size=3,
            strides=2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.concat_input3_24 = layers.Concatenate(axis=-1)

        self.conv_25 = DarknetConv(
            filters=512,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_26 = DarknetConv(
            filters=1024,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_27 = DarknetConv(
            filters=512,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_28 = DarknetConv(
            filters=1024,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_29 = DarknetConv(
            filters=512,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )

        self.conv_30 = DarknetConv(
            filters=1024,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_31 = DarknetConv(
            filters=3 * (num_classes + 5),
            kernel_size=1,
            activation=None,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )

    def call(self, x):
        route_1, route_2, route_3 = x

        x1 = self.conv_0(route_3)
        part2 = self.upSampling_0(x1)
        part1 = self.conv_1(route_2)
        x1 = self.concat_0_1([part1, part2])

        x1 = self.conv_2(x1)
        x1 = self.conv_3(x1)
        x1 = self.conv_4(x1)
        x1 = self.conv_5(x1)
        x1 = self.conv_6(x1)

        x2 = self.conv_7(x1)
        part2 = self.upSampling_7(x2)
        part1 = self.conv_8(route_1)
        x2 = self.concat_7_8([part1, part2])

        x2 = self.conv_9(x2)
        x2 = self.conv_10(x2)
        x2 = self.conv_11(x2)
        x2 = self.conv_12(x2)
        x2 = self.conv_13(x2)

        pred_s = self.conv_14(x2)
        pred_s = self.conv_15(pred_s)

        x2 = self.conv_16(x2)
        x2 = self.concat_6_16([x2, x1])

        x2 = self.conv_17(x2)
        x2 = self.conv_18(x2)
        x2 = self.conv_19(x2)
        x2 = self.conv_20(x2)
        x2 = self.conv_21(x2)

        pred_m = self.conv_22(x2)
        pred_m = self.conv_23(pred_m)

        x2 = self.conv_24(x2)
        x2 = self.concat_input3_24([x2, route_3])

        x2 = self.conv_25(x2)
        x2 = self.conv_26(x2)
        x2 = self.conv_27(x2)
        x2 = self.conv_28(x2)
        x2 = self.conv_29(x2)

        pred_l = self.conv_30(x2)
        pred_l = self.conv_31(pred_l)

        return pred_s, pred_m, pred_l

