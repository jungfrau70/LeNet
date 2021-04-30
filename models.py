from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, AveragePooling2D, Flatten, Dense


class FeatureExtractor(Layer):
    def __init__(self, filters1, filters2):
        super(FeatureExtractor, self).__init__()

        self.conv1 = Conv2D(filters=filters1, kernel_size=5, padding='valid',
                            strides=1, activation='tanh')
        self.conv1_pool = AveragePooling2D(pool_size=2, strides=2)

        self.conv2 = Conv2D(filters=filters2, kernel_size=5, padding='valid',
                            strides=1, activation='tanh')
        self.conv2_pool = AveragePooling2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)
        return x


class LeNet1(Model):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.feature_extractor = FeatureExtractor(4, 12)

        self.flatten = Flatten()
        self.dense1 = Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.feature_extractor(x)

        x = self.flatten(x)
        x = self.dense1(x)
        return x


class LeNet4(Model):
    def __init__(self):
        super(LeNet4, self).__init__()
        self.feature_extractor = FeatureExtractor(4, 16)

        self.flatten = Flatten()
        self.dense1 = Dense(units=120, activation='tanh')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.feature_extractor(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class LeNet5(Model):
    def __init__(self, activation='tanh'):
        super(LeNet5, self).__init__()
        self.feature_extractor = FeatureExtractor(6, 16)

        self.flatten = Flatten()
        self.dense1 = Dense(units=140, activation=activation)
        self.dense2 = Dense(units=84, activation=activation)
        self.dense3 = Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.feature_extractor(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
