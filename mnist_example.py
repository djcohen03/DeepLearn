from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical

# Load Data:
(trainimages, trainlabels), (testimages, testlabels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax', input_shape=(28 * 28,)))
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Reshape
trainimages = trainimages.reshape((60000, 28 * 28)).astype('float32') / 255.
testimages = testimages.reshape((10000, 28 * 28)).astype('float32') / 255.

# Categorically Encode:
trainlabels = to_categorical(trainlabels)
testlabels = to_categorical(testlabels)

# Train:
network.fit(trainimages, trainlabels, epochs=5, batch_size=128)


# Test:
loss, acc = network.evaluate(testimages, testlabels)
