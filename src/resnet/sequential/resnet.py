import tensorflow as tf 
from resnet.residual_block import make_basic_block_layer, make_bottleneck_layer

class ResNetTypeI(tf.keras.Model):
    def __init__(self,layer_params, filter_num=64):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(7,7),
            strides=2,
            padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(3,3),
            strides=2,
            padding="same"
        )
        self.layer1 = make_basic_block_layer(
            filter_num=filter_num,
            blocks=layer_params[0]
        )
        self.layer2 = make_basic_block_layer(
            filter_num=filter_num*2,
            blocks=layer_params[1],
            stride=2
        )
        self.layer3 = make_basic_block_layer(
            filter_num=filter_num*4,
            blocks=layer_params[2],
            stride=2
        )
        self.layer4 = make_basic_block_layer(
            filter_num=filter_num*8,
            blocks=layer_params[3],
            stride=2
        )

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output 

class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params, filter_num=64):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(7,7),
            strides=2,
            padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(3,3),
            strides=2,
            padding="same"
        )

        self.layer1 = make_bottleneck_layer(
            filter_num=filter_num,
            blocks=layer_params[0]
        )
        self.layer2 = make_bottleneck_layer(
            filter_num=filter_num*2,
            blocks=layer_params[1]            
        )
        self.layer3 = make_bottleneck_layer(
            filter_num=filter_num*4,
            blocks=layer_params[2]
        )
        self.layer4 = make_bottleneck_layer(
            filter_num=filter_num*8,
            blocks=layer_params[3]
        )

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(
            units=10,
            activation=tf.keras.activations.softmax
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output 

def resnet_18(filters):
    return ResNetTypeI(layer_params=[2,2,2,2], filter_num=filters)

def resnet_34(filters):
    return ResNetTypeI(layer_params=[3,4,6,3], filter_num=filters)

def resnet_50(filters):
    return ResNetTypeII(layer_params=[3,4,6,3], filter_num=filters)

def resnet_101(filters):
    return ResNetTypeII(layer_params=[3,4,23,3], filter_num=filters)

def resnet_152(filters):
    return ResNetTypeII(layer_params=[3,8,36,3], filter_num=filters)