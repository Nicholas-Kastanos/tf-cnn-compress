import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers, backend

class Mish(layers.Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    '''
    def call(self, x):
        return x * backend.tanh(backend.softplus(x))


class DarknetConv(layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        downsample: bool=False,
        activate: bool=True, 
        bn: bool=True,
        activate_type: str = "mish",
        use_asymetric_conv: bool=False,
        **kwargs
    ):
        super(DarknetConv, self).__init__(**kwargs)
        self.sequential = Sequential()

        if downsample:
            self.sequential.add(layers.ZeroPadding2D(((1, 0), (1, 0))))
            padding = 'valid'
            strides = 2
        else:
            padding = 'same'
            strides = 1

        if use_asymetric_conv and kernel_size==3 and not downsample:
            self.sequential.add(layers.Conv2D(
                filters=filters,
                kernel_size=(kernel_size, 1),
                strides=strides,
                padding=padding,
                use_bias=not bn,
                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.constant_initializer(0.)
            ))
            self.sequential.add(layers.Conv2D(
                filters=filters,
                kernel_size=(1, kernel_size),
                strides=strides,
                padding=padding,
                use_bias=not bn,
                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.constant_initializer(0.)
            ))
        else:
            self.sequential.add(layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=not bn,
                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.constant_initializer(0.)
            ))

        if bn:
            self.sequential.add(layers.BatchNormalization())

        if activate:
            if activate_type == "mish":
                self.sequential.add(Mish())
            elif activate_type == "leaky":
                self.sequential.add(layers.LeakyReLU(alpha=0.1))
            elif activate_type == "relu":
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
        activate_type: str = "mish",
        use_asymetric_conv=False,
        **kwargs
    ):
        super(DarknetResidual, self).__init__(**kwargs)
        self.conv_1 = DarknetConv(filters=filters_1, kernel_size=1, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
        self.conv_2 = DarknetConv(filters=filters_2, kernel_size=3, activate_type=activate_type, use_asymetric_conv=use_asymetric_conv)
        self.add = layers.Add()

    def call(self, x):
        short_cut = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.add([short_cut, x])
        return x


class ResidualBlock(layers.Layer):
    def __init__(
        self,
        iterations: int,
        filters_1: int,
        filters_2: int,
        activation: str = "mish",
        use_asymetric_conv=False,
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
                    use_asymetric_conv=use_asymetric_conv
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
        use_asymetric_conv=False,
        **kwargs
    ):
        super(DarknetResidualBlock, self).__init__(**kwargs)

        self.sequential = Sequential()

        self.sequential.add(
            DarknetConv(
                filters=filters_2,
                kernel_size=3,
                activation=activation,
                strides=2,
                use_asymetric_conv=use_asymetric_conv
            )
        )

        self.sequential.add(
            ResidualBlock(
                iterations=iterations,
                filters_1=filters_1,
                filters_2=filters_2,
                activation=activation,
                use_asymetric_conv=use_asymetric_conv
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
        use_asymetric_conv=False,
        **kwargs
    ):
        super(CSPResidualBlock, self).__init__(**kwargs)
        self.part1_conv = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetric_conv=use_asymetric_conv
        )
        self.part2_conv1 = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetric_conv=use_asymetric_conv
        )
        self.part2_res = ResidualBlock(
            iterations=iterations,
            filters_1=filters_1,
            filters_2=filters_2,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetric_conv=use_asymetric_conv
        )
        self.part2_conv2 = DarknetConv(
            filters=filters_2,
            kernel_size=1,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_asymetric_conv=use_asymetric_conv
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
        use_asymetric_conv=False,
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
                use_asymetric_conv=use_asymetric_conv
            )
        )

        self.sequential.add(
            CSPResidualBlock(
                iterations=iterations,
                filters_1=filters_1,
                filters_2=filters_2,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                use_asymetric_conv=use_asymetric_conv
            )
        )

        self.sequential.add(
            DarknetConv(
                filters=filters_2,
                kernel_size=1,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                use_asymetric_conv=use_asymetric_conv
            )
        )
    
    def call(self, x):
        return self.sequential(x)

