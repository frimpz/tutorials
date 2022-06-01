from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage._shared.filters import gaussian
from skimage.filters import rank
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import disk
import tensorflow as tf
import pickle


def load_image(file_name):
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def image_to_point(img):
    data = []
    for (x, y), value in np.ndenumerate(img):
        if value == 1:
            data.append([y, -x])
    return data


def draw_image_points(graph, title, ax=None, x_axis=0, y_axis=1):
    for node in graph.nodes(data=True):
        if ax:
            ax.scatter(node[1]['weight'][x_axis], node[1]['weight'][y_axis], s=2, c='b')
            ax.axis('off')
            ax.title.set_text(title)
            continue
        plt.scatter(node[1]['weight'][x_axis], node[1]['weight'][y_axis], s=2, c='b')
    if not ax:
        plt.title(title)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()


def draw_image2(graph, title, ax=None):
    for edge in graph.edges():
        pos = np.vstack(np.array([graph.nodes[edge[0]]['weight'], graph.nodes[edge[1]]['weight']]
                                 , dtype=np.float))
        if ax:
            ax.plot(*pos.T, color='black')
            ax.axis('off')
            ax.title.set_text(title)
            continue
        line, = plt.plot(*pos.T, color='black')
        plt.setp(line, linewidth=0.2, color='black')
    if not ax:
        plt.title(title)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()


def binarize_image(img, sigma=1.0, thresh=None, threshold_adjustment=0, to_gray=True, up=None):
    if to_gray:
        img = color.rgb2gray(img)
    img = gaussian(img, sigma=sigma)
    if thresh is None:
        thresh = threshold_otsu(img) + threshold_adjustment
    if up:
        return up * img < thresh
    else:
        return img < thresh


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def pickle_save(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)
