# -----------------------------------------------------------
# hexalattice module creates and prints hexagonal lattices
#
# (C) 2020 Alex Kazakov,
# Released under MIT License
# email alex.kazakov@mail.huji.ac.il
# Full documentation: https://github.com/alexkaz2/hexalattice
# -----------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from typing import List, Union


def create_hex_grid(nx: int = 4,
                    ny: int = 5,
                    min_diam: float = 1.,
                    n: int = 0,
                    align_to_origin: bool = True,
                    face_color: Union[List[float], str] = None,
                    edge_color: Union[List[float], str] = None,
                    plotting_gap: float = 0.,
                    crop_circ: float = 0.,
                    do_plot: bool = False,
                    rotate_deg: float = 0.,
                    keep_x_sym: bool = True,
                    h_ax: plt.Axes = None) -> (np.ndarray, plt.Axes):
    """
    Creates and prints hexagonal lattices.
    :param nx: Number of horizontal hexagons in rectangular grid, [nx * ny]
    :param ny: Number of vertical hexagons in rectangular grid, [nx * ny]
    :param min_diam: Minimal diameter of each hexagon.
    :param n: Alternative way to create rectangular grid. The final grid might have less hexagons
    :param align_to_origin: Shift the grid s.t. the central tile will center at the origin
    :param face_color: Provide RGB triplet, valid abbreviation (e.g. 'k') or RGB+alpha
    :param edge_color: Provide RGB triplet, valid abbreviation (e.g. 'k') or RGB+alpha
    :param plotting_gap: Gap between the edges of adjacent tiles, in fraction of min_diam
    :param crop_circ: Disabled if 0. If >0 a circle of central tiles will be kept, with radius r=crop_circ
    :param do_plot: Add the hexagon to an axes. If h_ax not provided a new figure will be opened.
    :param rotate_deg: Rotate the grid around the center of the central tile, by rotate_deg degrees
    :param keep_x_sym: NOT YET IMPLEMENTED
    :param h_ax: Handle to axes. If provided the grid will be added to it, if not a new figure will be opened.
    :return:
    """

    args_are_ok = check_inputs(nx, ny, min_diam, n, align_to_origin, face_color, edge_color, plotting_gap, crop_circ,
                               do_plot, rotate_deg, keep_x_sym)
    if not args_are_ok:
        print('Aborting hexagonal grid creation...')
        exit()
    coord_x, coord_y = make_grid(nx, ny, min_diam, n, crop_circ, rotate_deg, align_to_origin)

    if do_plot:
        h_ax = plot_single_lattice(coord_x, coord_y, face_color, edge_color, min_diam, plotting_gap, rotate_deg, h_ax)

    return np.hstack([coord_x, coord_y]), h_ax


def check_inputs(nx, ny, min_diam, n, align_to_origin, face_color, edge_color, plotting_gap, crop_circ, do_plot,
                 rotate_deg, keep_x_sym):
    """
    Validate input types, ranges and co-compatibility
    :return: bool - Assertion verdict
    """
    args_are_valid = True
    if (not isinstance(nx, (int, float))) or (not isinstance(ny, (int, float))) or (not isinstance(n, (int, float))) \
            or (nx < 0) or (nx < 0) or (nx < 0):
        print('Argument error in hex_grid: nx, ny and n are expected to be integers')
        args_are_valid = False

    if (not isinstance(min_diam, (float, int))) or (not isinstance(crop_circ, (float, int))) or (min_diam < 0) or \
            (crop_circ < 0):
        print('Argument error in hex_grid: min_diam and crop_circ are expected to be floats')
        args_are_valid = False

    if (not isinstance(align_to_origin, bool)) or (not isinstance(do_plot, bool)):
        print('Argument error in hex_grid: align_to_origin and do_plot are expected to be booleans')
        args_are_valid = False

    VALID_C_ABBR = {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}
    if (isinstance(face_color, str) and (not face_color in VALID_C_ABBR)) or \
       (isinstance(edge_color, str) and (not edge_color in VALID_C_ABBR)):
        print('Argument error in hex_grid: edge_color and face_color are expected to valid color abbrs, e.g. `k`')
        args_are_valid = False

    if (isinstance(face_color, List) and ((len(face_color) not in (3, 4)) or
                                          (True in ((x < 0) or (x > 1) for x in face_color)))) or \
        (isinstance(edge_color, List) and ((len(edge_color) not in (3, 4)) or
                                          (True in ((x < 0) or (x > 1) for x in edge_color)))):
        print('Argument error in hex_grid: edge_color and face_color are expected to be valid RGB color triplets or '
              'color abbreviations, e.g. [0.1 0.3 0.95] or `k`')
        args_are_valid = False

    if (not isinstance(plotting_gap, float)) or (plotting_gap < 0) or (plotting_gap >= 1):
        print('Argument error in hex_grid: plotting_gap is expected to be a float in range [0, 1)')
        args_are_valid = False

    if not isinstance(rotate_deg, (float, int)):
        print('Argument error in hex_grid: float is expected to be float or integer')
        args_are_valid = False

    if (n == 0) and ((nx == 0) or (ny == 0)):
        print('Argument error in hex_grid: Expected either n>0 or both [nx.ny]>0')
        args_are_valid = False

    if (isinstance(min_diam, (float, int)) and isinstance(crop_circ, (float, int))) and \
            (not np.isclose(crop_circ, 0)) and (crop_circ < min_diam):
        print('Argument error in hex_grid: Cropping radius is expected to be bigger than a single hexagon diameter')
        args_are_valid = False

    if not isinstance(keep_x_sym, bool):
        print('Argument error in hex_grid: keep_x_sym is expected to be boolean')
        args_are_valid = False

    return args_are_valid


def plot_single_lattice(coord_x, coord_y, face_color, edge_color, min_diam, plotting_gap, rotate_deg, h_ax=None):
    """
    Adds a single lattice to the axes canvas. Multiple calls can be made to overlay few lattices.
    :return:
    """
    if face_color is None:
        face_color = (1, 1, 1, 0)  # Make the face transparent
    if edge_color is None:
        edge_color = 'k'

    if h_ax is None:
        h_fig = plt.figure(figsize=(5, 5))
        h_ax = h_fig.add_axes([0.05, 0.05, 0.9, 0.9])

    patches = []
    for curr_x, curr_y in zip(coord_x, coord_y):
        polygon = mpatches.RegularPolygon((curr_x, curr_y), numVertices=6,
                                          radius=min_diam / np.sqrt(3) * (1 - plotting_gap),
                                          orientation=np.deg2rad(-rotate_deg))
        patches.append(polygon)
    collection = PatchCollection(patches, edgecolor=edge_color, facecolor=face_color)
    h_ax.add_collection(collection)

    h_ax.set_aspect('equal')
    h_ax.axis([coord_x.min() - 2 * min_diam, coord_x.max() + 2 * min_diam, coord_y.min() - 2 * min_diam,
               coord_y.max() + 2 * min_diam])
    # plt.plot(0, 0, 'r.', markersize=5)   # Add red point at the origin
    return h_ax


def make_grid(nx, ny, min_diam, n, crop_circ, rotate_deg, align_to_origin) -> (np.ndarray, np.ndarray):
    """
    Computes the coordinates of the hexagon centers, given the size rotation and layout specifications
    :return:
    """
    ratio = np.sqrt(3) / 2
    if n > 0:  # n variable overwrites (nx, ny) in case all three were provided
        ny = int(np.sqrt(n / ratio))
        nx = n // ny

    coord_x, coord_y = np.meshgrid(np.arange(nx), np.arange(ny), sparse=False, indexing='xy')
    coord_y = coord_y * ratio
    coord_x = coord_x.astype('float')
    coord_x[1::2, :] += 0.5
    coord_x = coord_x.reshape(-1, 1)
    coord_y = coord_y.reshape(-1, 1)

    coord_x *= min_diam  # Scale to requested size
    coord_y = coord_y.astype('float') * min_diam

    mid_x = (np.ceil(nx / 2) - 1) + 0.5 * (np.ceil(ny/2) % 2 == 0)  # Pick center of some hexagon as origin for rotation or crop...
    mid_y = (np.ceil(ny / 2) - 1) * ratio  # np.median() averages center 2 values for even arrays :\
    mid_x *= min_diam
    mid_y *= min_diam

    # mid_x = (nx // 2 - (nx % 2 == 1)) * min_diam + 0.5 * (ny % 2 == 1)
    # mid_y = (ny // 2 - (ny % 2)) * min_diam * ratio

    if crop_circ > 0:
        rad = ((coord_x - mid_x)**2 + (coord_y - mid_y)**2)**0.5
        coord_x = coord_x[rad.flatten() <= crop_circ, :]
        coord_y = coord_y[rad.flatten() <= crop_circ, :]

    if not np.isclose(rotate_deg, 0):  # Check if rotation is not 0, with tolerance due to float format
        # Clockwise, 2D rotation matrix
        RotMatrix = np.array([[np.cos(np.deg2rad(rotate_deg)), np.sin(np.deg2rad(rotate_deg))],
                              [-np.sin(np.deg2rad(rotate_deg)), np.cos(np.deg2rad(rotate_deg))]])
        rot_locs = np.hstack((coord_x - mid_x, coord_y - mid_y)) @ RotMatrix.T
        # rot_locs = np.hstack((coord_x - mid_x, coord_y - mid_y))
        coord_x, coord_y = np.hsplit(rot_locs + np.array([mid_x, mid_y]), 2)

    if align_to_origin:
        coord_x -= mid_x
        coord_y -= mid_y

    return coord_x, coord_y


def plot_single_lattice_custom_colors(coord_x, coord_y, face_color, edge_color, min_diam, plotting_gap, rotate_deg,
                                      line_width=1., h_ax=None):
    """
    Plot hexagonal lattice where every hexagon is colored by an individual color.
    All inputs are similar to the plot_single_lattice() except:
    :param line_width:
    :param h_ax:
    :param rotate_deg:
    :param plotting_gap:
    :param min_diam:
    :param coord_y:
    :param coord_x:
    :param face_color: numpy array, Nx3 or Nx4 - Color list of length |coord_x| for each hexagon face.
                                                 Each row is a RGB or RGBA values, e.g. [0.3 0.3 0.3 1]
    :param edge_color: numpy array, Nx3 or Nx4 - Color list of length |coord_x| for each hexagon edge.
                                                 Each row is a RGB or RGBA values, e.g. [0.3 0.3 0.3 1]
    :return:
    """

    if h_ax is None:
        h_fig = plt.figure(figsize=(5, 5))
        h_ax = h_fig.add_axes([0.05, 0.05, 0.9, 0.9])

    for i, (curr_x, curr_y) in enumerate(zip(coord_x, coord_y)):
        polygon = mpatches.RegularPolygon((curr_x, curr_y), numVertices=6,
                                          radius=min_diam / np.sqrt(3) * (1 - plotting_gap),
                                          orientation=np.deg2rad(-rotate_deg),
                                          edgecolor=edge_color[i],
                                          facecolor=face_color[i], linewidth=line_width)
        h_ax.add_artist(polygon)

    h_ax.set_aspect('equal')
    h_ax.axis([coord_x.min() - 2 * min_diam, coord_x.max() + 2 * min_diam, coord_y.min() - 2 * min_diam,
               coord_y.max() + 2 * min_diam])
    # plt.plot(0, 0, 'r.', markersize=5)   # Add red point at the origin
    return h_ax


def sample_colors_from_image_by_grid(image_path: str, x_coords, y_coords):
    """
    Sample colors of a set of points from an image.
    :param image_path: str - Path to an image file (.png, .jpg)
    :param x_coords: list - x coordinates of the hexagons. The range doesn't matter since it is rescaled to fit the image
    :param y_coords: list - y -----//------
    :return:
    """
    from matplotlib.image import imread
    img = imread(image_path) / 255
    img = np.flip(img, 0)  # Flip Y axis. Images = matrices where pixel [0, 0] is in the upper left corner

    abs_min = np.min([x_coords.T, y_coords.T])
    abs_max = np.max([x_coords.T, y_coords.T]) + 0.001
    minor_image_dim = min(img.shape[0], img.shape[1])
    p_x = np.floor((x_coords - abs_min) / (abs_max - abs_min) * minor_image_dim)
    p_y = np.floor((y_coords - abs_min) / (abs_max - abs_min) * minor_image_dim)

    colors = img[p_y.astype('int'), p_x.astype('int'), :]
    return colors


def main():

    plt.ion()

    # (1) === Create single hexagonal 5*5 lattice and plot it. Extract the [x,y] locations of the tile centers
    hex_centers, h_ax = create_hex_grid(nx=5, ny=5, do_plot=True)
    tile_centers_x = hex_centers[:, 0]
    tile_centers_y = hex_centers[:, 1]
    # plt.show(block=True)   % The 'show' call should be done explicitly

    # (2) === Create single hexagonal lattice, 5*5, rotated around central tile ====
    hex_centers, _ = create_hex_grid(nx=5,
                                     ny=5,
                                     plotting_gap=0.05,
                                     min_diam=1,
                                     rotate_deg=5,
                                     face_color=[0.9, 0.1, 0.1, 0.05],
                                     do_plot=True)

    # (3) === Plot Moire pattern with two round hexagonal grids ====
    hex_grid1, h_ax = create_hex_grid(nx=50,
                                      ny=50,
                                      rotate_deg=0,
                                      min_diam=1,
                                      crop_circ=20,
                                      do_plot=True)
    create_hex_grid(nx=50,
                    ny=50,
                    min_diam=1,
                    rotate_deg=5,
                    crop_circ=20,
                    do_plot=True,
                    h_ax=h_ax)

    # (4) === Create 5 layers of grids of various sizes ====
    face_c = [0.7, 0.7, 0.7, 0.1]
    _, h_ax = create_hex_grid(nx=5,
                              ny=4,
                              plotting_gap=0.2,
                              face_color=face_c,
                              min_diam=1,
                              do_plot=True)
    create_hex_grid(nx=4,
                    ny=3,
                    min_diam=1,
                    plotting_gap=0.3,
                    face_color=face_c,
                    do_plot=True,
                    h_ax=h_ax)
    create_hex_grid(nx=2,
                    ny=2,
                    plotting_gap=0.4,
                    face_color=face_c,
                    do_plot=True,
                    h_ax=h_ax)
    create_hex_grid(nx=1,
                    ny=1,
                    plotting_gap=0.5,
                    face_color=face_c,
                    do_plot=True,
                    h_ax=h_ax)
    create_hex_grid(nx=10,
                    ny=10,
                    edge_color=[0.9, 0.9, 0.9],
                    do_plot=True,
                    h_ax=h_ax)

    # (5) === Color hexagons with custom colors ===
    image_path = r'example_image.jpg'  # taken from https://en.wikipedia.org/wiki/Apple#/media/File:Red_Apple.jpg
    hex_centers, h_ax = create_hex_grid(nx=50, ny=50, do_plot=False)
    colors = sample_colors_from_image_by_grid(image_path, hex_centers[:, 0], hex_centers[:, 1])

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(plt.imread(image_path))
    plot_single_lattice_custom_colors(hex_centers[:, 0], hex_centers[:, 1],
                                      face_color=colors,
                                      edge_color=colors,
                                      min_diam=1.,
                                      plotting_gap=0,
                                      rotate_deg=0,
                                      line_width=0.3,
                                      h_ax=axs[0, 1])
    plot_single_lattice_custom_colors(hex_centers[:, 0], hex_centers[:, 1],
                                      face_color=colors,
                                      edge_color=colors*0,
                                      min_diam=1.,
                                      plotting_gap=0,
                                      rotate_deg=0,
                                      line_width=1.,
                                      h_ax=axs[1, 0])
    plot_single_lattice_custom_colors(hex_centers[:, 0], hex_centers[:, 1],
                                      face_color=colors,
                                      edge_color=colors*0,
                                      min_diam=1.,
                                      plotting_gap=0,
                                      rotate_deg=0,
                                      line_width=0.1,
                                      h_ax=axs[1, 1])

    plt.show(block=True)


if __name__ == "__main__":
    # Main function includes multiple examples
    main()
