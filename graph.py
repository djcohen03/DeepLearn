import matplotlib.pyplot as plt


class Graph(object):
    @classmethod
    def epochs(cls, metadata):
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
