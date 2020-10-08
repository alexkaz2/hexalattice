# hexalattice

Generate and plot hexagonal lattices in 2D, with fine control over spacing between hexagons, arbitrary rotation of the grid around central tile, etc.
The module computes and returns the center point for each fo the tiles in the lattice. 

<p align="center">
  <img width="450" src="https://github.com/alexkaz2/hexalattice/blob/master/example_hexagonal_lattices/lattice6.png">
</p>



## Installation

[![PyPI version](https://badge.fury.io/py/hexalattice.svg)](https://pypi.python.org/pypi/hexalattice)
![python version](https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg)

```sh
pip install hexalattice
```



## Usage example

Create and plot 5x5 lattice of hexagons (as in first image):
```sh
import hexalattice
hex_centers, _ = create_hex_grid(nx=5,
                                 ny=5,
                                 do_plot=True)
                                 
plt.show()    # import matplotlib.pyplot as plt
```

Get central points of the hexagons:
```sh
tile_centers_x = hex_centers[:, 0]
tile_centers_y = hex_centers[:, 1]
```

Plot one grid over the other, second with spacing around the hexagons:
```sh
_, h_ax = create_hex_grid(nx=5, 
                          ny=7,
                          do_plot=True,
                          edge_color=(0.85, 0.85, 0.85))
                                    
create_hex_grid(nx=5,
                ny=7,
                do_plot=True,
                edge_color=(0.25,0.25, 0.25),
                h_ax=h_ax,
                plotting_gap=0.3)
plt.show()
```

<p align="center">
  <img width="600" src="https://github.com/alexkaz2/hexalattice/blob/master/example_hexagonal_lattices/lattice5.png">
</p>

Create Moiré pattern from two circularly cropped hexagrids:
```sh
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
```

<p align="center">
  <img width="600" src="https://github.com/alexkaz2/hexalattice/blob/master/example_hexagonal_lattices/lattice2.png">
</p>

_For API and additional examples see [Wiki][wiki]._

<p align="center">
  <img width="450" src="https://github.com/alexkaz2/hexalattice/blob/master/example_hexagonal_lattices/lattice1.png" hspace="10"/>
  <img width="450" src="https://github.com/alexkaz2/hexalattice/blob/master/example_hexagonal_lattices/lattice3.png" hspace="10"/>
</p>

## Release History

* 1.0.0
    * First version

## About & License

Alex Kazakov – [@bio_vs_silico](https://twitter.com/bio_vs_silico) – alex.kazakov@mail.huji.ac.il

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/alexkaz2/hexalattice](https://github.com/alexkaz2/)
