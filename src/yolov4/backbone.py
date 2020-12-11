import tensorflow as tf
from tensorflow import keras

from .common import DarknetConv, ResidualBlock

class Darknet53(keras.Model):
    def __init__(
        self,
        activation0: str = "mish",
        activation1: str = "leaky",
        kernel_regularizer = None
    ):
        super(Darknet53, self).__init__(name="Darknet53")
        
        self.conv0 = DarknetConv(
            filters=32, 
            kernel_size=3, 
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
            strides=1
            )
        self.conv1 = DarknetConv(
            filters=64, 
            kernel_size=3, 
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
            strides=2
            )
        self.res0 = ResidualBlock(
            iterations=1,
            filters_1=32, 
            filters_2=64,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
            )
        self.conv2 = DarknetConv(
            filters=128, 
            kernel_size=3, 
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
            strides=2
            )
        self.res1 = ResidualBlock(
            iterations=2,
            filters_1=64, 
            filters_2=128,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
            )
        self.conv3 = DarknetConv(
            filters=256, 
            kernel_size=3, 
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
            strides=2
            )
        self.res2 = ResidualBlock(
            iterations=8,
            filters_1=128, 
            filters_2=256,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
            )
        self.conv4 = DarknetConv(
            filters=512, 
            kernel_size=3, 
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
            strides=2
            )
        self.res3 = ResidualBlock(
            iterations=8,
            filters_1=256, 
            filters_2=512,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
            )
        self.conv5 = DarknetConv(
            filters=1024, 
            kernel_size=3, 
            activation=activation0,
            kernel_regularizer=kernel_regularizer,
            strides=2
            )
        self.res4 = ResidualBlock(
            iterations=4,
            filters_1=512, 
            filters_2=1024,
            activation=activation0,
            kernel_regularizer=kernel_regularizer
            )
    
    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.res0(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)

        route1 = x

        x = self.conv4(x)
        x = self.res3(x)

        route2 = x

        x = self.conv5(x)
        x = self.res4(x)

        return route1, route2, x