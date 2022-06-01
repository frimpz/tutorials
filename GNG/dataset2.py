from matplotlib import pyplot as plt
from skimage import data, img_as_float, color
from skimage._shared.filters import gaussian
from skimage.filters.thresholding import threshold_otsu
import numpy as np
import MyGNG
import time
from utils import image_to_point, draw_image2, draw_image_points

nrows = 2
ncols = 3
fig, axs = plt.subplots(
    nrows, ncols, figsize=(20, 16),
)

data = img_as_float(data.astronaut())
# crop image to reduce data points
data = data[30:180, 150:300]

axs[0, 0].imshow(data, interpolation='nearest')
axs[0, 0].title.set_text("Original Image")

gray = color.rgb2gray(data)
gaussian = gaussian(gray, sigma=0.6)
axs[0, 1].title.set_text("Gray Scale Image with Gaussian Smoothing")
axs[0, 1].imshow(gaussian, cmap='gray')

threshold_adjustment = 0.15
thresh = threshold_otsu(gaussian) + threshold_adjustment
binary = gaussian < thresh
axs[0, 2].title.set_text("Binarized image")
axs[0, 2].imshow(gaussian > thresh, cmap='gray');

image_point = image_to_point(binary)
axs[1, 0].title.set_text("Binarized image")
axs[1, 0].scatter(*np.array(image_point).T, alpha=1)

data = np.array(image_point)

kwargs = {
    'max_units': 2400,
    'probability': 'shuffle'
}

gng = MyGNG.MyGrowingNeuralGas(**kwargs)

for epoch in range(50):
    print("Training Epoch is {}".format(epoch))
    t = time.time()
    gng.train(data)
    print('time: {:.4f}s'.format(time.time() - t))

draw_image_points(gng.graph, "Reconstruction of Image with neurons", ax=axs[1, 1])
draw_image2(gng.graph, "Reconstruction of Image with edges", ax=axs[1, 2])

plt.suptitle("Reconstruction of dataset with growing neural gas",
                     fontsize=24, ha='center', fontweight="bold")
plt.savefig("images/astronaut.png")
