import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, ReLU, Add, ZeroPadding2D, Concatenate
from tensorflow.keras.regularizers import l2

from typing import Union

from ..common import Mish


class DarknetConv(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        activation: str = "mish",
        kernel_regularizer=l2(0.0005),
        strides: int = 1,
        **kwargs
    ):
        super(DarknetConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.activation = activation
        self.strides = (strides, strides)
        self.sequential = Sequential()

        if self.strides[0] == 2:
            self.sequential.add(ZeroPadding2D(((1, 0), (1, 0))))

        self.sequential.add(Conv2D(
            filters=self.filters,
            kernel_size=kernel_size,
            strides=self.strides,
            padding="same" if self.strides[0] == 1 else "valid",
            kernel_regularizer=kernel_regularizer
        ))

        if self.activation is not None:
            self.sequential.add(BatchNormalization())

        if self.activation == "mish":
            self.sequential.add(Mish())
        elif self.activation == "leaky":
            self.sequential.add(LeakyReLU(alpha=0.1))
        elif self.activation == "relu":
            self.sequential.add(ReLU())

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

    def call(self, x):
        return self.sequential(x)


class DarknetResidual(Layer):
    def __init__(
        self,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None
    ):
        super(DarknetResidual, self).__init__()
        self.conv_1 = DarknetConv(
            filters=filters_1,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )
        self.conv_2 = DarknetConv(
            filters=filters_2,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )
        self.add = Add()

    def call(self, x):
        prev = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.add([prev, x])
        return x


class ResidualBlock(Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None
    ):
        super(ResidualBlock, self).__init__()
        self.sequential = Sequential()
        for _ in range(iterations):
            self.sequential.add(
                DarknetResidual(
                    filters_1=filters_1,
                    filters_2=filters_2,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer
                )
            )

    def call(self, x):
        return self.sequential(x)


class DarknetResidualBlock(Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None
    ):
        super(DarknetResidualBlock, self).__init__()

        self.sequential = Sequential()

        self.sequential.add(
            DarknetConv(
                filters=filters_2,
                kernel_size=3,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                strides=2
            )
        )

        self.sequential.add(
            ResidualBlock(
                iterations=iterations,
                filters_1=filters_1,
                filters_2=filters_2,
                activation=activation,
                kernel_regularizer=kernel_regularizer
            )
        )

    def call(self, x):
        return self.sequential(x)


class CSPResidualBlock(Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None
    ):
        super(CSPResidualBlock, self).__init__()
        self.part1_conv = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )
        self.part2_conv1 = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )
        self.part2_res = ResidualBlock(
            iterations=iterations,
            filters_1=filters_1,
            filters_2=filters_2,
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )
        self.part2_conv2 = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )
        self.concat1_2 = Concatenate(axis=-1)
    
    def call(self, x):
        part1 = self.part1_conv(x)

        part2 = self.part2_conv1(x)
        part2 = self.part2_res(part2)
        part2 = self.part2_conv2(part2)

        x = self.concat1_2([part2, part1])

class CSPDarknetResidualBlock(Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None
    ):
        super(CSPDarknetResidualBlock, self).__init__()

        self.sequential = Sequential()

        self.sequential.add(
            DarknetConv(
                filters=filters_1,
                kernel_size=3,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                strides=2
            )
        )

        self.sequential.add(
            CSPResidualBlock(
                iterations=iterations,
                filters_1=filters_1,
                filters_2=filters_2,
                activation=activation,
                kernel_regularizer=kernel_regularizer
            )
        )

        self.sequential.add(
            DarknetConv(
                filters=filters_2,
                kernel_size=1,
                activation=activation,
                kernel_regularizer=kernel_regularizer
            )
        )
    
    def call(self, x):
        return self.sequential(x)

