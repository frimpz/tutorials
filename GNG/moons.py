'''
    Experiment for Dataset 1.
'''

from matplotlib import pyplot as plt
import MyGNG
import time
from utils import draw_image_points, draw_image2

from sklearn.datasets import make_moons
data, _ = make_moons(10000, noise=0.06, random_state=0)


plt.scatter(*data.T)
plt.title("Moon dataset with noise {}".format(0.06))
plt.savefig("images/moon.png")

kwargs = {
    'max_units': 1000,
    'probability': 'random'
}

gng = MyGNG.MyGrowingNeuralGas(**kwargs)

nrows = 4
ncols = 4
fig, axs = plt.subplots(
    nrows, ncols, figsize=(20, 16),
)
nrows = 4
ncols = 4
fig2, axs2 = plt.subplots(
    nrows, ncols, figsize=(20, 16),
)
for epoch in range(15001):
    print("Training Epoch is {}".format(epoch))
    t = time.time()
    gng.train(data)
    print('time: {:.4f}s'.format(time.time() - t))

    if epoch % 1000 == 0:
        draw_image_points(gng.graph, "Iteration {}".format(epoch),
                         ax=axs[int(epoch/1000/4), int(epoch/1000)%4])
        plt.suptitle("Reconstruction of moon dataset with growing neural gas",
                     fontsize=24, ha='center', fontweight="bold")
        draw_image2(gng.graph, "Iteration {}".format(epoch),
                          ax=axs2[int(epoch / 1000 / 4), int(epoch / 1000) % 4])
        plt.suptitle("Reconstruction of moon dataset with growing neural gas with edges",
                     fontsize=24, ha='center', fontweight="bold")

fig.savefig("images/moon_units_only.png")
fig2.savefig("images/moon_edges.png")

