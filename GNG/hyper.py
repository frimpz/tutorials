from matplotlib import pyplot as plt
from skimage import data, img_as_float, color
from skimage._shared.filters import gaussian
from skimage.filters.thresholding import threshold_otsu
import numpy as np
import MyGNG
import time
from utils import image_to_point, draw_image2, draw_image_points

nrows = 3
ncols = 3
fig, axs = plt.subplots(
    nrows, ncols, figsize=(20, 18),
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
image_point = image_to_point(binary)
axs[0, 2].title.set_text("Image to points")
axs[0, 2].scatter(*np.array(image_point).T, alpha=1)

data = np.array(image_point)
print(data.shape)

col_count = 0
kwargs = {'probability': 'shuffle'}
gng_shuffle = MyGNG.MyGrowingNeuralGas(**kwargs)
for epoch in range(25):
    print("Training Epoch is {}".format(epoch))
    t = time.time()
    gng_shuffle.train(data)
    print('time: {:.4f}s'.format(time.time() - t))

    if epoch == 1 or epoch == 15 or epoch == 24:
        draw_image_points(gng_shuffle.graph, "Reconstruction from {} epoch".format(epoch)
                          , ax=axs[1, col_count])
        col_count += 1

kwargs = {'probability': 'random'}
gng_random = MyGNG.MyGrowingNeuralGas(**kwargs)
cnt = 1000
col_count = 0
for epoch in range(400001):
    print("Training Epoch is {}".format(epoch))
    t = time.time()
    gng_random.train(data)
    print('time: {:.4f}s'.format(time.time() - t))

    if epoch == 10000 or epoch == 250000 or epoch == 400000:
        draw_image_points(gng_random.graph, "Reconstruction from {} iterations".format(epoch)
                          , ax=axs[2, col_count])
        col_count += 1

plt.suptitle("Plot showing how recontsructions are formed for various iterations.",
                     fontsize=24, ha='center', fontweight="bold")
plt.savefig("images/hyper.png")
