from matplotlib import pyplot as plt
from skimage import data as ski_data, img_as_float
import numpy as np
from skimage.transform import resize
from sklearn.datasets import make_moons

import MyGNG
import time
from utils import image_to_point, draw_image_points, binarize_image

'''
    In this script we compare how reconstructions are formed varying 
    1. Max number of units
    2. Max age of edges
    3. 
'''

nrows = 2
ncols = 2
fig2, axs2 = plt.subplots(
    nrows, ncols, figsize=(12, 8),
)
fig2.suptitle('Original Images', fontsize=14, ha='center', fontweight="bold")

moon_image, _ = make_moons(10000, noise=0.06, random_state=0)
axs2[0, 0].scatter(*moon_image.T)
axs2[0, 0].title.set_text("Moon Image")

astro_image = img_as_float(ski_data.astronaut())
astro_image = astro_image[30:180, 150:300]
axs2[0, 1].imshow(astro_image)
axs2[0, 1].title.set_text("Astronaut Image")
astro_image = binarize_image(astro_image, sigma=0.6)
astro_image = image_to_point(astro_image)
astro_image = np.array(astro_image)

camera_image = img_as_float(ski_data.camera())
camera_image = resize(camera_image, (256, 256, 3))
axs2[1, 0].imshow(camera_image)
axs2[1, 0].title.set_text("Camera Image")
camera_image = binarize_image(camera_image, sigma=0.6, threshold_adjustment=0.05)
camera_image = image_to_point(camera_image)
camera_image = np.array(camera_image)

horse_image = np.bitwise_not(ski_data.horse())
horse_image = resize(horse_image, (230, 280))
axs2[1, 1].imshow(horse_image, cmap='binary')
axs2[1, 1].title.set_text("Horse Image")
horse_image = image_to_point(horse_image)
horse_image = np.array(horse_image)

fig2.savefig("images/original_image.png")

max_units = [500, 2000]
stream_types = ['single', 'mix']

for stream_type in stream_types:
    for no_unit in max_units:
        print("Generating for {} stream with max number of units {}".format(stream_type, no_unit))

        nrows = 4
        ncols = 4
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(20, 16),
        )
        fig.suptitle("Generating for {} stream with max number of units {}".format(stream_type, no_unit),
                     fontsize=14, ha='center', fontweight="bold")

        axs[0, 0].scatter(*moon_image.T, alpha=1)
        axs[0, 0].title.set_text("Moon 2 points")

        axs[0, 1].scatter(*astro_image.T, alpha=1)
        axs[0, 1].title.set_text("Astronaut 2 points")

        axs[0, 2].scatter(*camera_image.T, alpha=1)
        axs[0, 2].title.set_text("Astronaut 2 points")

        axs[0, 3].scatter(*horse_image.T, alpha=1)
        axs[0, 3].title.set_text("Horse 2 points")

        kwargs = {
            'max_units': no_unit,
            'probability': 'shuffle'
        }

        gng = MyGNG.MyGrowingNeuralGas(**kwargs)

        row_count = 1
        col_count = 0


        for epoch in range(60):
            print("Training Epoch is {}".format(epoch))
            t = time.time()
            if stream_type == 'mix':
                if epoch % 4 == 0:
                    gng.train(moon_image)
                elif epoch % 4 == 1:
                    gng.train(astro_image)
                elif epoch % 4 == 2:
                    gng.train(camera_image)
                elif epoch % 4 == 3:
                    gng.train(horse_image)
            elif stream_type == 'single':
                if epoch < 15:
                    gng.train(moon_image)
                elif epoch < 30:
                    gng.train(astro_image)
                elif epoch < 45:
                    gng.train(camera_image)
                elif epoch < 60:
                    gng.train(horse_image)

            if epoch % 5 == 4:
                draw_image_points(gng.graph, "{}"
                                  .format("{} EPOCHS".format(epoch)),
                                  ax=axs[row_count, col_count])
                col_count += 1

            if col_count == 4:
                col_count = 0
                row_count += 1

        fig.savefig("images/{}_streaming_{}.png".format(stream_type, no_unit))