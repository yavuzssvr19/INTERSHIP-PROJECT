import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
CM = 1 / 2.54
TINY = (5 * CM, 3 * CM)
SMALL = (9 * CM, 5 * CM)
HALF_PAGE = (18 * CM, 7 * CM)
FULL_PAGE = (18 * CM, 15 * CM)

DEEP_BLUE = "#006685"
BLUE = "#3FA5C4"
WHITE = "#FFFFFF"
HALF_BLACK = "#232324"
ORANGE = "#E84653"
RED = "#BF003F"

PURPLE = "#A6587C"
PURPLER = "#591154"
PURPLEST = "#260126"
NIGHT_BLUE = "#394D73"
YELLOW = "#E6B213"

TEAL = "#44cfcf"
SLOW_GREEN = "#a1d4ca"
GRAY = "#b8b8b8"

black_to_gray_to_reds = [GRAY, GRAY, WHITE, ORANGE, RED]
from_white = [DEEP_BLUE, BLUE, WHITE, ORANGE, RED]
white_to_reds = [WHITE, ORANGE, RED]
white_to_blues = [WHITE, BLUE, DEEP_BLUE]
teal_to_red = [TEAL, SLOW_GREEN, WHITE, YELLOW, ORANGE, RED]

black_to_reds = [HALF_BLACK, ORANGE, RED]
black_to_blues = [HALF_BLACK, BLUE, DEEP_BLUE]

from_black = [DEEP_BLUE, BLUE, HALF_BLACK, ORANGE, RED]
purples = [PURPLE, WHITE, NIGHT_BLUE]

discretes = [
    DEEP_BLUE,
    BLUE,
    WHITE,
    HALF_BLACK,
    "#F05A6E",  # Adjusted Orange
    "#A90051",  # Adjusted Red
    "#B3698D",  # Adjusted Purple
    YELLOW,
    "#4D96FF",  # Sky Blue
]

mono_black_gray_red = sns.blend_palette(black_to_gray_to_reds, as_cmap=True)
diverge_from_white = sns.blend_palette(from_white, as_cmap=True)
purples_diverge_from_white = sns.blend_palette(purples, as_cmap=True)

diverge_from_black = sns.blend_palette(from_black, as_cmap=True)

white_red_mono = sns.blend_palette(white_to_reds, as_cmap=True)
white_blue_mono = sns.blend_palette(white_to_blues, as_cmap=True)

black_red_mono = sns.blend_palette(black_to_reds, as_cmap=True)
black_blue_mono = sns.blend_palette(black_to_blues, as_cmap=True)

purple_red = sns.blend_palette([PURPLEST, PURPLE, RED])
teal_red = sns.blend_palette(teal_to_red, as_cmap=True)
monochrome = sns.blend_palette([HALF_BLACK, GRAY], as_cmap=True)
discrete_map = sns.blend_palette(discretes, as_cmap=True)


def set_visual_style():
    sns.set_theme(
        style="ticks",
        font_scale=0.75,
        rc={
            "font.family": "Atkinson Hyperlegible",
            "font.sans-serif": ["Atkinson-Hyperlegible"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.labelpad": 2,
            "axes.linewidth": 0.5,
            "axes.titlepad": 4,
            "lines.linewidth": 1,
            "legend.fontsize": 8,
            "legend.title_fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 2,
            "xtick.major.pad": 1,
            "xtick.major.width": 0.5,
            "ytick.major.size": 2,
            "ytick.major.pad": 1,
            "ytick.major.width": 0.5,
            "xtick.minor.size": 2,
            "xtick.minor.pad": 1,
            "xtick.minor.width": 0.5,
            "ytick.minor.size": 2,
            "ytick.minor.pad": 1,
            "ytick.minor.width": 0.5,
            "text.color": "#232324",
            "patch.edgecolor": "#232324",
            "patch.force_edgecolor": False,
            "hatch.color": "#232324",
            "axes.edgecolor": "#232324",
            "axes.labelcolor": "#232324",
            "xtick.color": "#232324",
            "ytick.color": "#232324",
        },
    )
    
def brain_plotter(
    data: np.ndarray,
    coordinates: np.ndarray,
    axis: plt.Axes,
    view: Tuple[int, int] = (90, 180),
    size: int = 20,
    cmap: any = "viridis",
    scatter_kwargs=Optional[None],
) -> plt.Axes:
    """plots the 3D scatter plot of the brain. It's a simple function that takes the data, the coordinates, and the axis and plots the brain.
    It's a modified version the netneurotools python package but you can give it the axis to plot in. See here:
    https://netneurotools.readthedocs.io/en/latest/

    Args:
        data (np.ndarray): the values that need to be mapped to the nodes. Shape is (N,)
        coordinates (np.ndarray): 3D coordinates fo each node. Shape is (N, 3)
        axis (plt.Axes): Which axis to plot in. This means you have to already have a figure and an axis to plot in.
        view (Tuple[int, int], optional): Which view to look at. Defaults to (90, 180).
        size (int, optional): Size of the nodes. Defaults to 20.
        cmap (any, optional): Color map. Defaults to "viridis" which I don't like but you do you.
        scatter_kwargs (_type_, optional): kwargs for the dots. Defaults to Optional[None].

    Returns:
        plt.Axes: matplotlib axis with the brain plotted.
    """
    scatter_kwargs = scatter_kwargs if scatter_kwargs else {}

    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=data,
        cmap=cmap,
        s=size,
        **scatter_kwargs,
    )
    axis.view_init(*view)
    axis.axis("off")
    scaling = np.array([axis.get_xlim(), axis.get_ylim(), axis.get_zlim()])
    axis.set_box_aspect(tuple(scaling[:, 1] / 1.2 - scaling[:, 0]))
    return axis

