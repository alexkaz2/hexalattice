from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="hexalattice",
    version="1.0.0",
    description="Compute and plot hexagonal grids",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/alexkaz2/hexalattice/wiki",
    author="Alex Kazakov",
    author_email="alex.kazakov@mail.huji.ac.il",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["hexalattice"],
    install_requires=["numpy", "matplotlib"],
)