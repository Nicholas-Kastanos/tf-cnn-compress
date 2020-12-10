import tensorflow as tf
from tensorflow import keras

from typing import Union

from ..common import Mish

class DarknetConv(keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, tuple],
        activation: str = "mish",
        kernel_regularizer=keras.regularizers.l2(0.0005),
        **kwargs
    ):
        super(DarknetConv, self).__init__(**kwargs)
        self.filters = filters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.activation = activation
        
        self.sequential = keras.Sequential()

        self.sequential.add(keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=kernel_size,
            padding="same",
            strides=(1, 1),
            kernel_regularizer=kernel_regularizer
        ))

        self.sequential.add(keras.layers.BatchNormalization())

        if self.activation == "mish":
            self.sequential.add(Mish())
        elif self.activation == "leaky":
            self.sequential.add(keras.layers.LeakyReLU(alpha=0.1))
        elif self.activation == "relu":
            self.sequential.add(keras.layers.ReLU())

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

    def call(self, x):
        return self.sequential(x)

class DarknetResidual(keras.Model):
    def __init__(
        self,
        filters_1: Union[int, tuple] = 1,
        filters_2: Union[int, tuple] = 3,
        activation: str = "mish",
        kernel_regularizer = None
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
        self.add = keras.layers.Add()

    def call(self, x):
        prev = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.add([prev, x])
        return x
    
class ResidualBlock(keras.Model):
    def __init__(
        self,
        iterations: int,
        filters_1: Union[int, tuple] = 1,
        filters_2: Union[int, tuple] = 3,
        activation: str = "mish",
        kernel_regularizer=None,
    ):
        super(ResidualBlock, self).__init__()
        self.sequential = keras.Sequential()
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