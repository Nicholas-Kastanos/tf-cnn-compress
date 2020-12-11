import tensorflow as tf
from tensorflow.keras import Layer

from .common import DarknetConv, DarknetResidualBlock, CSPDarknetResidualBlock


class Darknet53(Layer):
    def __init__(
        self,
        activation0: str = "mish",
        activation1: str = "leaky",
        kernel_regularizer=None
    ):
        super(Darknet53, self).__init__(name="Darknet53")

        self.conv0 = DarknetConv(
            filters=32,
            kernel_size=3,
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
            strides=1
        )
        self.res0 = DarknetResidualBlock(
            iterations=1,
            filters_1=32,
            filters_2=64,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )
        self.res1 = DarknetResidualBlock(
            iterations=2,
            filters_1=64,
            filters_2=128,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )
        self.res2 = DarknetResidualBlock(
            iterations=8,
            filters_1=128,
            filters_2=256,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )
        self.res3 = DarknetResidualBlock(
            iterations=8,
            filters_1=256,
            filters_2=512,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )
        self.res4 = DarknetResidualBlock(
            iterations=4,
            filters_1=512,
            filters_2=1024,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
        )

    def call(self, x):
        x = self.conv0(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)

        route1 = x

        x = self.res3(x)

        route2 = x

        x = self.res4(x)

        return route1, route2, x

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

    def call(self, x):
        x = self.conv0(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)

        route1 = x

        x = self.res3(x)

        route2 = x

        x = self.res4(x)

        return route1, route2, x
