import networkx as nx
from matplotlib import pyplot as plt
from skimage import data, img_as_float, color
from skimage._shared.filters import gaussian
from skimage.filters.thresholding import threshold_otsu
import numpy as np
import MyGNG
import time








astro = img_as_float(data.astronaut())
print(astro.shape)
astro = astro[30:180, 150:300]
print(astro.shape)
astro_grey = color.rgb2gray(astro)
print(astro_grey.shape)
astro_grey = gaussian(astro_grey, sigma=0.6)
print(astro_grey.shape)
threshold_adjustment = 0.1
thresh = threshold_otsu(astro_grey) + threshold_adjustment
binary_astro = astro_grey < thresh

print(binary_astro.shape)

exit()

def image_to_data(img):
    data = []
    for (x, y), value in np.ndenumerate(img):
        if value == 1:
            data.append([y, -x])
    return data

data = np.array(image_to_data(binary_astro))
print(data.shape)
exit()
# exit()
#
# import cv2
# im = cv2.imread("abc.tiff",mode='RGB')
# print type(im)

def create_gng(max_nodes, step=0.2, n_start_nodes=2, max_edge_age=50):
    return algorithms.GrowingNeuralGas(
        n_inputs=2,
        n_start_nodes=n_start_nodes,

        shuffle_data=True,
        verbose=True,

        step=step,
        neighbour_step=0.005,

        max_edge_age=max_edge_age,
        max_nodes=max_nodes,

        n_iter_before_neuron_added=100,
        after_split_error_decay_rate=0.5,
        error_decay_rate=0.995,
        min_distance_for_update=0.01,
    )


from neupy import algorithms, utils


from neupy import algorithms, utils

utils.reproducible()

kwargs = {
    'max_units': 2400,
    'probability': 'shuffle'
}

gng = MyGNG.MyGrowingNeuralGas(**kwargs)

for epoch in range(2):
    #print("Training Epoch is {}".format(epoch))
    t = time.time()
    gng.train(data)
    #print('time: {:.4f}s'.format(time.time() - t))


    # Plot images after each iteration in order to see training progress
    # plt.figure(figsize=(5.5, 6))
    # draw_image(gng.graph)
# print(gng.graph)

plt.figure(figsize=(5.5, 6))
draw_image(gng.graph)