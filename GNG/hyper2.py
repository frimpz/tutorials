from matplotlib import pyplot as plt
from skimage import data, img_as_float, color
from skimage._shared.filters import gaussian
from skimage.filters.thresholding import threshold_otsu
import numpy as np
import MyGNG
from utils import image_to_point, draw_image2, draw_image_points

'''
    In this script we compare how reconstructions are formed varying 
    1. Max number of units
    2. Max age of edgesg
    3. 
'''

data = img_as_float(data.astronaut())
data = data[30:180, 150:300]
gray = color.rgb2gray(data)
gaussian = gaussian(gray, sigma=0.6)
threshold_adjustment = 0.15
thresh = threshold_otsu(gaussian) + threshold_adjustment
binary = gaussian < thresh
image_point = image_to_point(binary)
data = np.array(image_point)

# # Holding all parameters the same, how does the GNG with respect to changing eb
# parameters = [(("eb", 0.3), ), (("eb", 0.4),), (("eb", 0.5),), (("eb", 0.7),),
#               (("en", 0.005),), (("en", 0.1),), (("en", 0.3),), (("en", 0.5),),
#               (("lambda", 10),), (("lambda", 50),), (("lambda", 250),), (("lambda", 500),)]
#
#
# epochs = [20000, 100000, 250000]
# row_count = 0
# col_count = 0
# for epoch in epochs:
#     nrows = 3
#     ncols = 4
#     fig, axs = plt.subplots(
#         nrows, ncols, figsize=(20, 16),
#     )
#     for param in parameters:
#         for _param in param:
#             kwargs = {
#                 _param[0]: _param[1]
#             }
#             print("Epoch is {} and params are {}".format(epoch, kwargs))
#             gng_shuffle = MyGNG.MyGrowingNeuralGas(**kwargs)
#             for train_epoch in range(epoch):
#                 gng_shuffle.train(data)
#             draw_image_points(gng_shuffle.graph, "{}"
#                               .format(str(kwargs).strip("{}")),
#                               ax=axs[row_count % 3, col_count])
#             col_count += 1
#             if col_count == 4:
#                 col_count = 0
#                 row_count += 1
#
#     plt.suptitle("Plot showing reconstructions total number of iterations is -- {}.".format(epoch),
#                  fontsize=14, ha='center', fontweight="bold")
#
#     plt.savefig("images/hyper_a_{}.png".format(epoch))
#     plt.show()


###############
## Next batch of hyperparameters
# Holding all parameters the same, how does the GNG with respect to changing eb
parameters = [(("max_units", 100), ), (("max_units", 500),), (("max_units", 1000),), (("max_units", 2000),),
              (("max_age", 10),), (("max_age", 25),), (("max_age", 50),), (("max_age", 100),),
              (("alpha", 0.1),), (("alpha", 0.2),), (("alpha", 0.8),), (("alpha", 0.9),)]


epochs = [20000, 100000, 250000]
row_count = 0
col_count = 0
for epoch in epochs:
    nrows = 3
    ncols = 4
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(20, 16),

    )
    for param in parameters:
        for _param in param:
            kwargs = {
                _param[0]: _param[1]
            }
            print("Epoch is {} and params are {}".format(epoch, kwargs))
            gng_shuffle = MyGNG.MyGrowingNeuralGas(**kwargs)
            for train_epoch in range(epoch):
                gng_shuffle.train(data)
            draw_image_points(gng_shuffle.graph, "{}"
                              .format(str(kwargs).strip("{}")),
                              ax=axs[row_count % 3, col_count])
            col_count += 1
            if col_count == 4:
                col_count = 0
                row_count += 1

    plt.suptitle("Plot showing reconstructions total number of iterations is -- {}.".format(epoch),
                 fontsize=14, ha='center', fontweight="bold")

    plt.savefig("images/hyper_b_{}.png".format(epoch))
    plt.show()