import matplotlib.pyplot as plt


class Graph(object):
    @classmethod
    def metadata(cls, metadata):
        ''' Graph the Training Progress vs Validation Loss/Accuraccy in the
            training set and validation set
        '''

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

    @classmethod
    def digit(cls, data):
        ''' Plot a 28 by 28 pixel (grayscale) image
        '''
        if data.shape == (28 * 28, ):
            datacopy = data.copy().reshape((28, 28))
            plt.imshow(datacopy, cmap=plt.cm.binary)
        else:
            plt.imshow(data, cmap=plt.cm.binary)
        plt.show()
