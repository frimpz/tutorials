## Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import Loss

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops

latent_dim = 64
class Autoencoder(Model):
  def __init__(self, decoder_shape):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(1024, activation='relu'),
      layers.Dense(32, activation='relu')
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(1024, activation='relu'),
      layers.Dense(decoder_shape, activation='sigmoid')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class MeanSquaredError(Loss):
  def call(self, y_true, y_pred):
    # y_pred = tf.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
    # return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred, axis=-1, name=None)





autoencoder = Autoencoder(latent_dim)
# cost = tf.reduce_mean(tf.square(tf.sub(y, y_)))
# cost = losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
cost = MeanSquaredError()
autoencoder.compile(optimizer=opt, loss=cost)


(x_train, _), (x_test, _) = fashion_mnist.load_data()


# exit()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

history = autoencoder.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))

print(autoencoder.encoder.summary())
print(autoencoder.decoder.summary())


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# plt.plot(x_test[0], 'b')
# plt.plot(decoded_imgs[0], 'r')
# plt.fill_between(np.arange(140), decoded_imgs[0], x_test[0], color='lightcoral')
# plt.legend(labels=["Input", "Reconstruction", "Error"])
# plt.show()


# n = 20
# plt.figure(figsize=(20, 4))
# for i in range(n):
#   # display original
#   ax = plt.subplot(2, n, i + 1)
#   plt.imshow(x_test[i])
#   plt.title("ori")
#   plt.gray()
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
#
#   # display reconstruction
#   ax = plt.subplot(2, n, i + 1 + n)
#   plt.imshow(decoded_imgs[i])
#   plt.title("rec")
#   plt.gray()
#   ax.get_xaxis().set_visible(False)
#   ax.get_yaxis().set_visible(False)
# plt.show()


exit()



