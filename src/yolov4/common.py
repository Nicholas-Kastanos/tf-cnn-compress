import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers

from typing import Union

from ..common import Mish


class DarknetConv(layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        activation: str = "mish",
        kernel_regularizer=regularizers.l2(0.0005),
        strides: int = 1,
        use_asymetrical_conv=False,
        **kwargs
    ):
        super(DarknetConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.activation = activation
        self.strides = (strides, strides)
        self.sequential = Sequential()

        if self.strides[0] == 2:
            self.sequential.add(layers.ZeroPadding2D(((1, 0), (1, 0))))

        if use_asymetrical_conv and self.kernel_size == (3, 3) and self.strides == (1, 1):
            self.sequential.add(layers.Conv2D(
                filters=self.filters,
                kernel_size=(self.kernel_size[0], 1),
                strides=self.strides,
                padding="same",
                kernel_regularizer=kernel_regularizer
            ))
            self.sequential.add(layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, self.kernel_size[1]),
                strides=self.strides,
                padding="same",
                kernel_regularizer=kernel_regularizer
            ))
        else:
            self.sequential.add(layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding="same" if self.strides[0] == 1 else "valid",
                kernel_regularizer=kernel_regularizer
            ))

        if self.activation is not None:
            self.sequential.add(layers.BatchNormalization())

        if self.activation == "mish":
            self.sequential.add(Mish())
        elif self.activation == "leaky":
            self.sequential.add(layers.LeakyReLU(alpha=0.1))
        elif self.activation == "relu":
            self.sequential.add(layers.ReLU())

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

    def call(self, x):
        return self.sequential(x)


class DarknetResidual(layers.Layer):
    def __init__(
        self,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None,
        use_asymetrical_conv=False,
        **kwargs
    ):
        super(DarknetResidual, self).__init__(**kwargs)
        self.conv_1 = DarknetConv(
            filters=filters_1,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.conv_2 = DarknetConv(
            filters=filters_2,
            kernel_size=3,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.add = layers.Add()

    def call(self, x):
        prev = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.add([prev, x])
        return x


class ResidualBlock(layers.Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None,
        use_asymetrical_conv=False,
        **kwargs
    ):
        super(ResidualBlock, self).__init__(**kwargs)
        self.sequential = Sequential()
        for _ in range(iterations):
            self.sequential.add(
                DarknetResidual(
                    filters_1=filters_1,
                    filters_2=filters_2,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                    use_asymetrical_conv=use_asymetrical_conv
                )
            )

    def call(self, x):
        return self.sequential(x)


class DarknetResidualBlock(layers.Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None,
        use_asymetrical_conv=False,
        **kwargs
    ):
        super(DarknetResidualBlock, self).__init__(**kwargs)

        self.sequential = Sequential()

        self.sequential.add(
            DarknetConv(
                filters=filters_2,
                kernel_size=3,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                strides=2,
                use_asymetrical_conv=use_asymetrical_conv
            )
        )

        self.sequential.add(
            ResidualBlock(
                iterations=iterations,
                filters_1=filters_1,
                filters_2=filters_2,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                use_asymetrical_conv=use_asymetrical_conv
            )
        )

    def call(self, x):
        return self.sequential(x)


class CSPResidualBlock(layers.Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None,
        use_asymetrical_conv=False,
        **kwargs
    ):
        super(CSPResidualBlock, self).__init__(**kwargs)
        self.part1_conv = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.part2_conv1 = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.part2_res = ResidualBlock(
            iterations=iterations,
            filters_1=filters_1,
            filters_2=filters_2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.part2_conv2 = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetrical_conv=use_asymetrical_conv
        )
        self.concat1_2 = layers.Concatenate(axis=-1)
    
    def call(self, x):
        part1 = self.part1_conv(x)

        part2 = self.part2_conv1(x)
        part2 = self.part2_res(part2)
        part2 = self.part2_conv2(part2)

        x = self.concat1_2([part2, part1])
        return x

class CSPDarknetResidualBlock(layers.Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        kernel_regularizer=None,
        use_asymetrical_conv=False,
        **kwargs
    ):
        super(CSPDarknetResidualBlock, self).__init__(**kwargs)

        self.sequential = Sequential()

        self.sequential.add(
            DarknetConv(
                filters=filters_1,
                kernel_size=3,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                strides=2,
                use_asymetrical_conv=use_asymetrical_conv
            )
        )

        self.sequential.add(
            CSPResidualBlock(
                iterations=iterations,
                filters_1=filters_1,
                filters_2=filters_2,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                use_asymetrical_conv=use_asymetrical_conv
            )
        )

        self.sequential.add(
            DarknetConv(
                filters=filters_2,
                kernel_size=1,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                use_asymetrical_conv=use_asymetrical_conv
            )
        )
    
    def call(self, x):
        return self.sequential(x)

