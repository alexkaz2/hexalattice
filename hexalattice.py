import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from typing import List, Dict, Union


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
                    h_ax: plt.Axes = None) -> (Dict, plt.Axes):

    args_are_ok = check_inputs(nx, ny, min_diam, n, align_to_origin, face_color, edge_color, plotting_gap, crop_circ,
                               do_plot, rotate_deg, keep_x_sym)
    if not args_are_ok:
        print('Aborting hexagonal grid creation...')
        exit()
    coord_x, coord_y = make_grid(nx, ny, min_diam, n, crop_circ, rotate_deg, align_to_origin)

    if do_plot:
        h_ax = plot_single_lattice(coord_x, coord_y, face_color, edge_color, min_diam, plotting_gap, rotate_deg, h_ax)

    return {'coord_x': coord_x, 'coord_y': coord_y}, h_ax


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

    if (not isinstance(min_diam, (float, int))) or (not isinstance(crop_circ, (float, int))) or (min_diam < 0) or (crop_circ < 0):
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

    if (not isinstance(keep_x_sym, bool)):
        print('Argument error in hex_grid: keep_x_sym is expected to be boolean')
        args_are_valid = False


    return args_are_valid


def plot_single_lattice(coord_x, coord_y, face_color, edge_color, min_diam, plotting_gap, rotate_deg, h_ax=None):
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
    ratio = np.sqrt(3) / 2
    if n > 0:  # n variable overwrites (nx, ny) in case all three were provided
        ny = int(np.sqrt(n / ratio))
        nx = n // ny

    coord_x, coord_y = np.meshgrid(np.arange(nx), np.arange(ny), sparse=False, indexing='xy')
    coord_y = coord_y * ratio
    coord_x = coord_x.astype(np.float)
    coord_x[1::2, :] += 0.5
    coord_x = coord_x.reshape(-1, 1)
    coord_y = coord_y.reshape(-1, 1)

    coord_x *= min_diam  # Scale to requested size
    coord_y = coord_y.astype(np.float) * min_diam

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


def plot_hex_grid():
    pass


if __name__ == "__main__":

    # === Create single hexagonal lattice, 5*5, rotated around central tile ====
    hex_centers = create_hex_grid(nx=5,
                                  ny=5,
                                  plotting_gap=0.05,
                                  min_diam=1,
                                  rotate_deg=5,
                                  face_color=[0.9, 0.1, 0.1, 0.05],
                                  do_plot=True)
    centers_x = hex_centers[:, 0]
    centers_y = hex_centers[:, 1]
    # plt.show(block=True)   % The 'show' call should be done explicitly

    # === Plot Moire pattern with two round hexagonal grids ====
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

    # === Create 5 layers of grids of various sizes ====
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
    plt.show(block=True)
