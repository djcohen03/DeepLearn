import numpy as np
from keras.datasets import imdb
from keras import models, layers

# Import IMDB Training Data
(traindata, trainlabels), (testdata, testlabels) = imdb.load_data(num_words=10000)


wordindex = imdb.get_word_index()
wordmapping = dict([(v, k) for (k,v) in wordindex.items()])
def decode(words):
    return ' '.join([wordmapping.get(i - 3, '?') for i in words])


def vectorize(samples, dimensions=10000):
    results = np.zeros((len(samples), dimensions))
    for i, sample in enumerate(samples):
        results[i, sample] = 1.
    return results

traindata = vectorize(traindata)
trainlabels = np.array(trainlabels)


network = models.Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Get Validation Samples:
trainvalidatedata = traindata[:10000]
trainvalidatelabels = trainlabels[:10000]
traindata = traindata[10000:]
trainlabels = trainlabels[10000:]

metadata = network.fit(
    traindata,
    trainlabels,
    epochs=20,
    batch_size=512,
    validation_data=(trainvalidatedata, trainvalidatelabels)
)

import matplotlib.pyplot as plt

history = metadata.history
loss = history.get('loss')
val_loss = history.get('val_loss')
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend()
plt.show()

plt.clf()
acc = history.get('acc')
val_acc = history.get('val_acc')
plt.plot(epochs, acc, 'bo', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.legend()
plt.show()


def prediction(value):
    return "Positive" if val > 0.95 else "Negative" if val < 0.05 else "Unsure"

testlabels

# Show some Predicttions:
count = 10
predictions = network.predict(vectorize(testdatacopy[:count]))
for i in range(count):
    prediction()
    print "Prediction: %s\n%s" % (prediction, decode(testdatacopy[i])[:500])
