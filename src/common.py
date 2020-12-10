import tensorflow as tf
from tensorflow import keras

from typing import Union

class Mish(keras.layers.Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    '''
    def call(self, x):
        return x * keras.backend.tanh(keras.backend.softplus(x))