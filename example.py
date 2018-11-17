from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import numpy as np

import random
# def testfn(v):
#     return int(((((v % 3439) + 341) / 22.5 - 23413) ** 2) % 50)

def testfn(v):
    return int(v % 100)


trainvalues = np.array([random.randint(0, 10000000) for _ in range(100000)])
trainlabels = np.array([testfn(v) for v in trainvalues])
testvalues = np.array([random.randint(0, 10000000) for _ in range(10000)])
testlabels = np.array([testfn(v) for v in testvalues])


network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(1,)))
network.add(layers.Dense(100, activation='softmax'))
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Reshape
trainvalues = trainvalues.astype('float32') / 1000000.
testvalues = testvalues.astype('float32') / 1000000.

# Categorically Encode:
trainlabels = to_categorical(trainlabels)
testlabels = to_categorical(testlabels)

# Train:
network.fit(trainvalues, trainlabels, epochs=5, batch_size=128)


# Test:
loss, acc = network.evaluate(testvalues, testlabels)
