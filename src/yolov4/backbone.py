import tensorflow as tf
from tensorflow.keras import Layer, layers

from .common import DarknetConv, DarknetResidualBlock, CSPDarknetResidualBlock

class CSPDarknet53(Layer):
    def __init__(
        self,
        activation0: str = "mish",
        activation1: str = "leaky",
        kernel_regularizer=None
    ):
        super(CSPDarknet53, self).__init__()
        self.conv0 = DarknetConv(
            filters=32,
            kernel_size=3,
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
            strides=1
        )
        self.res0 = CSPDarknetResidualBlock(
            iterations=1,
            filters_1=32,
            filters_2=64,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )
        self.res1 = CSPDarknetResidualBlock(
            iterations=2,
            filters_1=64,
            filters_2=128,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )
        self.res2 = CSPDarknetResidualBlock(
            iterations=8,
            filters_1=128,
            filters_2=256,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )
        self.res3 = CSPDarknetResidualBlock(
            iterations=8,
            filters_1=256,
            filters_2=512,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )
        self.res4 = CSPDarknetResidualBlock(
            iterations=4,
            filters_1=512,
            filters_2=1024,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )

        self.SPP_pre_conv0 = DarknetConv(filters=512, kernel_size=1, activation=activation1, kernel_regularizer=kernel_regularizer)
        self.SPP_pre_conv1 = DarknetConv(filters=1024, kernel_size=3, activation=activation1, kernel_regularizer=kernel_regularizer)
        self.SPP_pre_conv2 = DarknetConv(filters=512, kernel_size=1, activation=activation1, kernel_regularizer=kernel_regularizer)

        self.spp = SPP()

        self.SPP_post_conv0 = DarknetConv(filters=512, kernel_size=1, activation=activation1, kernel_regularizer=kernel_regularizer)
        self.SPP_post_conv1 = DarknetConv(filters=1024, kernel_size=3, activation=activation1, kernel_regularizer=kernel_regularizer)
        self.SPP_post_conv2 = DarknetConv(filters=512, kernel_size=1, activation=activation1, kernel_regularizer=kernel_regularizer)

    def call(self, x):
        x = self.conv0(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)

        route1 = x

        x = self.res3(x)

        route2 = x

        x = self.res4(x)

        x = self.SPP_pre_conv0(x)
        x = self.SPP_pre_conv1(x)
        x = self.SPP_pre_conv2(x)

        x = self.spp(x)

        x = self.SPP_post_conv0(x)
        x = self.SPP_post_conv1(x)
        x = self.SPP_post_conv2(x)

        return route1, route2, x


class SPP(Layer):
    def __init__(
        self
    ):
        super(SPP, self).__init__()
        self.pool1 = layers.MaxPool2D(pool_size=(13, 13), strides=(1, 1), padding="same")
        self.pool2 = layers.MaxPool2D(pool_size=(9, 9), strides=(1, 1), padding="same")
        self.pool3 = layers.MaxPool2D(pool_size=(5, 5), strides=(1, 1), padding="same")
        self.concat = layers.Concatenate(axis=-1)

    def call(self, x):
        return self.concat([self.pool1(x), self.pool2(x), self.pool3(x), x])