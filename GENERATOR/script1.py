from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from Classes1 import DataGenerator
from tensorflow.keras import metrics
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

x_train = np.hstack((x_train, y_train))
x_test = np.hstack((x_test, y_test))

print(x_train.shape)
print(x_test.shape)
# exit()

# Parameters
params = {'batch_size': 100,
          'n_classes': 10,
          'shuffle': True,
          'data': x_train}

val_params = {'batch_size': 100,
          'n_classes': 10,
          'shuffle': True,
          'data': x_test}


# Generators
training_generator = DataGenerator(**params)
validation_generator = DataGenerator(**val_params)

metrics = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'),
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
      metrics.AUC(name='auc'),
      metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

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
    metrics=metrics)


history = model.fit(x=training_generator, validation_data=validation_generator,
            verbose=1, use_multiprocessing=True, workers=6, epochs=2)




def plot_metrics(history):
    mets = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'prc']
    fig, ax = plt.subplots(3, 2, constrained_layout=True, figsize=(18, 12))
    for i, j in enumerate(mets):
        row = int(i/2)
        col = i%2

        ax[row][col].plot(history.history[j], label="Training "+ j)
        ax[row][col].plot(history.history["val_"+j], label="Validation "+j)
        ax[row][col].set(ylabel=j, xlabel='epoch', title=j)
        ax[row][col].legend(['train', 'val'], loc='upper left')
    plt.show()

plot_metrics(history)