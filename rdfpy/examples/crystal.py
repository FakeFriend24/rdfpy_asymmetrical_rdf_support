import sys
import os
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdfpy import rdf3d

from matplotlib import rc
from multiprocessing import freeze_support

#added support for IDEs, Windows and just general multiprocessing 'endless loop error' protection
def main():
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    #adding folder_path of script to construct absolute pathing for relative file access
    folder_path = os.path.dirname(os.path.abspath(__file__))
    particles = np.load(os.path.join(folder_path,'./crystal.npy'))

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], color='r', alpha=0.4, edgecolors='k', s=25)
    plt.axis('off')
    plt.savefig(os.path.join(folder_path,'./crystal-particles.png'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    g_r, radii = rdf3d(particles, dr=0.1)

    image = plt.imread(os.path.join(folder_path,'./crystal-particles.png'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 2]})

    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(r'Crystal')

    axes[1].plot(radii, g_r, color='k', alpha=0.75)
    axes[1].hlines(y=1.0, xmin=0.0, xmax=max(radii), color='r', linestyle='--', alpha=0.4)
    axes[1].set_ylabel(r'g(r)')
    axes[1].set_xlabel(r'r')
    axes[1].set_xlim(0.0, max(radii))
    plt.savefig(os.path.join(folder_path,'./crystal.png'), bbox_inches='tight', pad_inches=0.25)
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()
