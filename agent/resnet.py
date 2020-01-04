import tensorflow as tf


class ResNetBlock(tf.keras.layers.Layer):
    '''
    Implements a single residual block of:
        input -> conv -> batch norm -> relu -> conv --> bn + input --> relu
    '''
    def __init__(self, n_filters):
        super(ResNetBlock, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1), padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1), padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        # Apply residual link to inputs.
        # This block assumes inputs supplied have already
        # undergone a convolution to correct dimensions.
        x = x + inputs

        return tf.keras.activations.relu(x)


class ResNet(tf.keras.layers.Layer):
    '''
    Implements a ResNet to encode game state.
    Output is not subject to activation
    - treat as a layer, not a full model.
    '''
    def __init__(self, n_filters=32):
        super(ResNet, self).__init__()
        self.conv_input = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1), padding='same')
        self.block_1 = ResNetBlock(n_filters)
        self.block_2 = ResNetBlock(n_filters)


    def call(self, inputs, task=None):
        x = self.conv_input(inputs)
        x = self.block_1(x)
        x = self.block_2(x)

        # No tail/body activation when passing to ActorHead and CriticHead
        return x
