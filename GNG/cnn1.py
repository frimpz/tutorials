import os

import torch
from matplotlib import pyplot as plt
from skimage import data
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import MyGNG
from utils import image_to_point, draw_image2, draw_image_points, get_model, pickle_save, pickle_load

'''
    In this script we compare how reconstructions are formed varying 
    1. Max number of units
    2. Max age of edges
    3. 
'''


def de_enum(data):
    _data = []
    for (x, y), value in np.ndenumerate(data):
        if value == 1:
            _data.append([y, -x])
    return _data


def pic2grph(data):
    data = data.reshape(28, 28)
    data[data > 0] = 1
    data = np.array(de_enum(data))
    kwargs = {'probability': 'shuffle', 'max_units': 50}
    gng = MyGNG.MyGrowingNeuralGas(**kwargs)
    for train_epoch in range(20):
        gng.train(data)

    output = np.zeros((28, 28))
    for i, ix in enumerate(data):
        output[-ix[1], ix[0]] = 1
    output = output.reshape(28, 28, 1)
    return output


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

train_numpy = tfds.as_numpy(ds_train)
test_numpy = tfds.as_numpy(ds_test)


# Example Image
nrows, ncols = 2, 2
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 16))

data = next(iter(train_numpy))[0].reshape(28, 28)
axs[0, 0].imshow(data, aspect="auto")
axs[0, 0].title.set_text("Original Image")
data[data > 0] = 1

data = np.array(de_enum(data))
axs[0, 1].scatter(*data.T)
axs[0, 1].title.set_text("Image Converted to points")
axs[0, 1].axis('off')

kwargs = {'probability': 'shuffle', 'max_units': 50}
gng = MyGNG.MyGrowingNeuralGas(**kwargs)
for train_epoch in range(100):
    gng.train(data)
draw_image_points(gng.graph, "Image from GNG", ax=axs[1, 0])

output = np.zeros((28, 28))
for i, ix in enumerate(data):
    output[-ix[1], ix[0]] = 1


axs[1, 1].imshow(output, aspect="auto")
axs[1, 1].title.set_text("Reconstruction from GNG")

plt.suptitle("MNIST Image to Points to GNG to Reconstructed Image",
            fontsize=24, ha='center', fontweight="bold")

# plt.show()
cnt = 0
X_test = []
Y_test = []
for i, j in test_numpy:
    print("{} out of {}".format(cnt, len(ds_test)))
    output = pic2grph(i)
    X_test.append(output)
    Y_test.append(j)
    cnt+=1
gng_test_numpy = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(len(ds_test))
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

gng_test = gng_test_numpy.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
gng_test = gng_test.batch(len(gng_test))
gng_test = gng_test.cache()
gng_test = gng_test.prefetch(tf.data.AUTOTUNE)
#

model = get_model()
model.fit(ds_train, epochs=10)

ds_test = list(ds_test)
prediction = model(ds_test[0][0])
prediction = tf.math.argmax(prediction, axis=1)
confusion = confusion_matrix(ds_test[0][1], prediction)
print(ds_test[0][1])

gng_test = list(gng_test)
prediction1 = model(gng_test[0][0])
prediction1 = tf.math.argmax(prediction1, axis=1)
confusion1 = confusion_matrix(gng_test[0][1], prediction)

index = [i for i in "0123456789"]
df_cm = pd.DataFrame(np.asarray(confusion),
                     index=index, columns=index)

df_cm2 = pd.DataFrame(np.asarray(confusion1),
                     index=index, columns=index)


nrows = 1
ncols = 2
fig, axs = plt.subplots(
    nrows, ncols, figsize=(20, 16),
)

plt.title("Confusion matrix for validation")
sn.heatmap(df_cm, annot=True, ax=axs[0])
sn.heatmap(df_cm, annot=True, ax=axs[1])
plt.show()


