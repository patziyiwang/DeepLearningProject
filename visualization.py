import matplotlib.pyplot as plt
import matplotlib as mpl


def visualize(image, name):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    plt.savefig(name + ".png")