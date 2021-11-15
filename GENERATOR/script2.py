from tensorflow.keras.datasets import mnist
from pyspark import RDD, SparkContext
import numpy as np
import pandas as pd
import tensorflow as tf
from Classes2 import DataGenerator


def add_label(x, y):
    df = pd.DataFrame(x)
    df['label'] = y.tolist()
    # df.insert(0, 'person_id', df.index)
    # df = df.set_index('person_id')
    return df

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

train = add_label(x_train, y_train)

test = add_label(x_test, y_test)

# partition -> Can get this from spark dataframe
partition = {
    'train': train.index.tolist(),
    'validation': test.index.tolist()
}

# labels -> Can get this dictionary from the task gold standard.
labels = train['label'].to_dict()
val_labels = test['label'].to_dict()


# Parameters
params = {'dim': 784,
          'batch_size': 100,
          'n_classes': 10,
          'shuffle': True,
          'data': train}


# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)


model = tf.keras.Sequential([
       tf.keras.layers.Dense(128, input_dim=784, activation='relu'),
       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy'])


model.fit(x=training_generator, validation_data=validation_generator, verbose=0,
                    # validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6, epochs=20)
