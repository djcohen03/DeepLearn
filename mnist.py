import time
import random
from PIL import Image
from graph import Graph
from keras.datasets import mnist as mnist_
from keras import models, layers
from keras.utils import to_categorical
import numpy as np

class Helpers(object):
    @classmethod
    def importimage(cls, path):
        ''' Import a 28*28 image as a grayscale integer bitmap
        '''
        image = Image.open(path)
        bitmap = np.zeros((28,28))
        for i in range(28):
            for j in range(28):
                r, g, b, s = image.getpixel((i,j))
                bitmap[j, i] = (r + g + b) / 3
        return bitmap


if __name__ == '__main__':
    # Load Data:
    start = time.time()
    print("Loading MNIST Image Dataset")
    (trainimages, trainlabels), (testimages, testlabels) = mnist_.load_data()
    print("Loaded MNIST Image Dataset in %.2fs" % (time.time() - start))

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Reshape Input Data from (28 * 28) matricies with values from 0 to 255 to
    # a (1 * 784)
    trainimages = trainimages.reshape((60000, 28 * 28)).astype('float32') / 255.
    testimages = testimages.reshape((10000, 28 * 28)).astype('float32') / 255.

    # Categorically Encode Output Digits (convert from [0,9] to, for example,
    # [0,0,0,0,0,1,0,0,0] for the digit "5")
    vectorized_trainlabels = to_categorical(trainlabels, num_classes=10)
    vectorized_testlabels = to_categorical(testlabels, num_classes=10)

    # Cut out Some Validation Data:
    validationimages = trainimages[:10000]
    trainimages = trainimages[10000:]
    validationlabels = vectorized_trainlabels[:10000]
    vectorized_trainlabels = vectorized_trainlabels[10000:]

    # Train Network:
    metadata = network.fit(
        trainimages,
        vectorized_trainlabels,
        epochs=5,
        batch_size=128,
        validation_data=(validationimages, validationlabels)
    )

    # Graph Training Progress:
    Graph.metadata(metadata)


    # Show Some Predictions:
    print("MNIST Samples:")
    count = 5
    testindicies = random.sample(range(len(testimages)), k=count)
    tests = testimages[testindicies]
    predictions = network.predict(tests)
    for i, prediction in enumerate(predictions):
        print("Expected: %s" % testlabels[testindicies[i]])
        for digit in range(10):
            print("\t%s: %.1f%%" % (digit, prediction[digit] * 100.))
        Graph.digit(tests[i])



    # Get Overall Accuracy of Testing Data Set:
    loss, acc = network.evaluate(testimages, vectorized_testlabels)
    print("Testing Data Set:")
    print("\tLoss: %.2f\n\tAccuracy: %.2f%%" % (loss, acc * 100.))



    # Test Some Handmade (low-quality) Samples:
    sampledata = np.array([
        Helpers.importimage('./img/seven.png'),
        Helpers.importimage('./img/three.png'),
        Helpers.importimage('./img/two.png'),
    ])
    sampledata = sampledata.reshape(len(sampledata), 28 * 28).astype('float32') / 255.
    samplelabels = [7, 3, 2]
    predictions = network.predict(sampledata)
    for i, prediction in enumerate(predictions):
        print("Expected: %s" % samplelabels[i])
        for digit in range(10):
            print("\t%s: %.1f%%" % (digit, prediction[digit] * 100.))
        Graph.digit(sampledata[i])
