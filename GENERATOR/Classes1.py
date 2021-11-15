import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, n_classes=10, shuffle=True, data=None):
        'Initialization'
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data = data
        self.num_cols = data.shape[1]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'

        filtered = self.data[np.ix_(indexes, list(range(self.num_cols)))]
        X = filtered[:, :-1]
        y = filtered[:, -1]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)