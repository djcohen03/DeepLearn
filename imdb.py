import time
import random
import numpy as np
from keras.datasets import imdb as imdb_
from keras import models, layers
from graph import Graph

# Dictionary Cutoff
MAX_WORDS = 10000


class Methods(object):

    wordindex = None
    wordmapping = None
    @classmethod
    def loadwords(cls):
        start = time.time()
        print("Loading IMDB Word Bank")
        cls.wordindex = imdb_.get_word_index()
        print("Loaded IMDB Word Bank in %.2fs" % (time.time() - start))
        cls.wordmapping = dict([(v, k) for (k,v) in cls.wordindex.items()])

    @classmethod
    def decode(cls, words):
        if not cls.wordindex or not cls.wordmapping:
            cls.loadwords()
        # Decode a word index to the string representation:
        return ' '.join([cls.wordmapping.get(i - 3, '?') for i in words])

    @classmethod
    def encode(cls, words):
        if not cls.wordindex or not cls.wordmapping:
            cls.loadwords()
        wordlist = words.lower().split(' ')
        indexlist = [cls.wordindex.get(word) for word in wordlist]
        return [x for x in indexlist if x is not None]

    @classmethod
    def ergprediction(cls, val):
        return "Positive" if val > 0.8 else "Negative" if val < 0.2 else "Unsure"

    @classmethod
    def vectorize(cls, samples, dimensions=MAX_WORDS):
        ''' Words are represented as indicies from 1 to MAX_WORDS, so this will
            transform an array of word indicies (eg [543, 33, 1435, 341, 2,...]) to a
            binary boolean array representing the the presense of the word in the
            array of indicies (eg [0, 1, 0, 0,...])
            This is done to the input data because it standardizes the shape/size of the
            input arrays to be (MAX_WORDS,)

            My Note: This might be improved by _not_ destroying the information
            corresponding to duplicate words in one array.  One might solve this by
            modifying the input above to use "counts". This would mean instead of
            using a simple "1" for presence and "0" for absense, we would use "n"
            for the number of times the word occurs in the sample text- this way we
            would know how many times each word appears.  Doesn't seem important
            enough to have been metioned in the book, however, so might just be a
            negligable improvement
        '''
        # Initialize an all-zeros matrix to represent the entire sample set
        vectorized = np.zeros((len(samples), dimensions))
        for i, sample in enumerate(samples):
            # Vectorize the ith sample:
            vectorized[i, sample] = 1.
        return vectorized




if __name__ == '__main__':

    # Load IMDB Training Data (With 10,000 most common words)
    print("Loading IMDB Datasset")
    start = time.time()
    (traindata, trainlabels), (testdata, testlabels) = imdb_.load_data(num_words=MAX_WORDS)
    print("Loaded IMDB Dataset (in %.2fs)" % (time.time() - start))
    # Vectorize the input data (see explanation above),
    vectortized_traindata = Methods.vectorize(traindata)
    trainlabels = np.array(trainlabels)
    # Construct Network
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
    validationdata = vectortized_traindata[:10000]
    validationlabels = trainlabels[:10000]
    vectortized_traindata = vectortized_traindata[10000:]
    trainlabels = trainlabels[10000:]
    print("Fitting Network:")
    metadata = network.fit(
        vectortized_traindata,
        trainlabels,
        epochs=20,
        batch_size=512,
        validation_data=(validationdata, validationlabels)
    )




    # MARK: Graph Validation Set Performance:
    print("Graphing Learning Performance:")
    Graph.epochs(metadata)



    # MARK: Testing Predictions from Training Data:
    print("IMDB Samples:")
    count = 10
    tests = np.random.choice(testdata, size=count)
    vectorized_tests = Methods.vectorize(tests)
    predictions = network.predict(vectorized_tests)
    for i in range(count):
        prediction = Methods.ergprediction(predictions[i])
        expected = Methods.ergprediction(testlabels[i])
        words = Methods.decode(tests[i])
        print("Predicted: %s\tActual: %s\tSample: %s" % (
            prediction,
            expected,
            words[:100]
        ))


    # MARK: Testing some hand-typed sample reviews:
    print("Handmade Samples:")
    samples = [
        ("This movie was fantastic I loved the surprise ending but don't go see it with your mom", 1),
        ("The plot was very intricate and well thought out the animation was a work of art Go see this movie right now", 1),
        ("I was not very impressed with the actors of this one they didn't really seem to embrace the characters the way that I would have expected which is a shame", 0),
        ("The plot was too easy to follow and lost my interest", 0),
    ]
    vectorized_tests = Methods.vectorize(
        np.array([Methods.encode(sample[0]) for sample in samples])
    )
    predictions = network.predict(vectorized_tests)
    for i in range(len(samples)):
        prediction = Methods.ergprediction(predictions[i])
        expected = Methods.ergprediction(samples[i][1])
        words = samples[i][0]
        print("Predicted: %s\tActual: %s\tSample: %s" % (
            prediction,
            expected,
            words[:100]
        ))
