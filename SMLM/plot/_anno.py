import math
import numpy as np
import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
import trackpy as tp
from skimage.morphology import binary_dilation, binary_erosion, disk


def set_ylim_reverse(ax):
    """
    This function is needed for annotation. Since ax.imshow(img) display
    the img in a different manner comparing with traditional axis.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    """
    bottom, top = ax.get_ylim()
    if top > bottom:
        ax.set_ylim(top, bottom)

def anno_blob(ax, blob_df,
            marker='o',
            markersize=5,
            plot_r=True,
            color=(0,1,0,0.8)):
    """
    Annotate blob in matplotlib axis.
    The blob parameters are obtained from blob_df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    blob_df : DataFrame
        bolb_df has columns of 'x', 'y', 'r'.
    makers: string, optional
        The marker for center of the blob.
    plot_r: bool, optional
        If True, plot the circle.
    color: tuple, optional
        Color of the marker and circle.

    Returns
    -------
    Annotate center and the periphery of the blob.
    """

    ax.scatter(blob_df['y'],blob_df['x'],marker=marker,facecolors='none', edgecolors=color)

def anno_scatter(ax, scatter_df, marker = 'o', color=(0,1,0,0.8)):
    """
    Annotate scatter in matplotlib axis.
    The scatter parameters are obtained from scatter_df.

    Parameters
    ----------
    ax : object
        matplotlib axis to annotate ellipse.
    scatter_df : DataFrame
        scatter_df has columns of 'x', 'y'.
    makers: string, optional
        The marker for the position of the scatter.
    color: tuple, optional
        Color of the marker and circle.

    Returns
    -------
    Annotate scatter in the ax.
    """

    set_ylim_reverse(ax)

    f = scatter_df
    for i in f.index:
        y, x = f.at[i, 'x'], f.at[i, 'y']
        ax.scatter(x, y,
                    s=10,
                    marker=marker,
                    c=[color])


