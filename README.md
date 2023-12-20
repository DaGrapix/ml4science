# ML4Science

This repository shows the implementation of Packed-Ensemble Algorithms for solving Fluid Mechanics problems.
The study provided here is part of the ML4physim challenge hosted by IRT-Systemx (see [Codabench page](https://www.codabench.org/competitions/1534/)).
CFD simulations being very costly, the use of data-based surrogate models can be useful to optimize the shape of airfoils without paying the cost of expensive simulations.

A family of models that is tested here is Packed-ensembles, which are generalizations of Deep-ensembles that allow to keep the number of an Ensemble Method's parameters small.

Two frameworks are proposed:
- A complete and independent framework developped in `ml4science.ipynb` with a custom training function and a cross validation selection implementation.
- An implementation of the Packed-Ensemble model within the LIPS framework in `packed_lips.ipynb`. All the configurations that were tried are developed in the `config.ini` file.

## Installation

### Install the LIPS framework

#### Setup a Virtualenv

```commandline
conda create --name ml4science python=3.9

```

##### Create a virtual environment

##### Enter virtual environment
```commandline
conda activate ml4science
```

#### Install from source
```commandline
git clone https://github.com/IRT-SystemX/LIPS.git
```
Then remove the `numpy` and `scipy` requirement from the `setup.py` file to avoid conflicts.

```commandline
cd LIPS
pip install -U .
cd ..
```

### Install pytorch
Checkout https://pytorch.org/get-started/locally/

### Install the Airfrans library and install the datasets

#### Install the library
```sh
pip install airfrans
```

#### Download the dataset
```sh
import os
import airfrans as af

directory_name='Dataset'
if not os.path.isdir(directory_name):
    af.dataset.download(root = ".", file_name = directory_name, unzip = True, OpenFOAM = False)
```

### Install torch-uncertainty
```sh
pip install torch-uncertainty
```
