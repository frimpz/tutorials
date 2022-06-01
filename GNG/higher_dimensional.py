import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import MyGNG
from utils import draw_image_points


nrows = 4
ncols = 3
fig, axs = plt.subplots(
    nrows, ncols, figsize=(12, 8),
)
fig.suptitle('Clustering 3D feature space with GNG', fontsize=12, ha='center', fontweight="bold")


X1, Y1 = make_blobs(n_features=3, centers=10)
axs[0, 0].scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")
axs[0, 1].scatter(X1[:, 0], X1[:, 2], marker="o", c=Y1, s=25, edgecolor="k")
axs[0, 2].scatter(X1[:, 1], X1[:, 2], marker="o", c=Y1, s=25, edgecolor="k")
axs[0, 0].axis('off')
axs[0, 0].title.set_text("X1--X2 space")
axs[0, 1].axis('off')
axs[0, 1].title.set_text("X1--X3 space")
axs[0, 2].axis('off')
axs[0, 2].title.set_text("X2--X3 space")

parameters = [5, 10, 20]

for i in parameters:
    kwargs = {
                'max_units': i,
                'probability': 'shuffle'
            }

    gng = MyGNG.MyGrowingNeuralGas(**kwargs)

    for epoch in range(50):
        gng.train(X1)

    if i == 5:
        draw_image_points(gng.graph, "Max Units {} X1--X2".format(i),
                         ax=axs[1, 0])
        draw_image_points(gng.graph, "Max Units {} X1--X3".format(i),
                         ax=axs[1, 1], x_axis=0, y_axis=2)
        draw_image_points(gng.graph, "Max Units {} X2--X3".format(i),
                         ax=axs[1, 2], x_axis=1, y_axis=2)

    elif i == 10:
        draw_image_points(gng.graph, "Max Units {} X1--X2".format(i),
                          ax=axs[2, 0])
        draw_image_points(gng.graph, "Max Units {} X1--X3".format(i),
                          ax=axs[2, 1], x_axis=0, y_axis=2)
        draw_image_points(gng.graph, "Max Units {} X2--X3".format(i),
                          ax=axs[2, 2], x_axis=1, y_axis=2)

    elif i == 20:
        draw_image_points(gng.graph, "Max Units {} X1--X2".format(i),
                          ax=axs[3, 0])
        draw_image_points(gng.graph, "Max Units {} X1--X3".format(i),
                          ax=axs[3, 1], x_axis=0, y_axis=2)
        draw_image_points(gng.graph, "Max Units {} X2--X3".format(i),
                          ax=axs[3, 2], x_axis=1, y_axis=2)

plt.savefig("images/3D.png")