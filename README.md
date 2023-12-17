# ML4Science

This repository shows the implementation of Packed-Ensemble Algorithms for solving Fluid Mechanics problems.
The study provided here is part of the ML4physim challenge hosted by IRT-Systemx. (see [Codabench page](https://www.codabench.org/competitions/1534/))
CFD simulations being very costly, the use of data-based surrogate models can 

Two frameworks are proposed:
- A complete and independent framework developped in `ml4science.ipynb` with a custom training function.
- An implementation of the Packed-Ensemble model within the LIPS framework.

## Installation

### Install the LIPS framework


#### Requirements
- Python >= 3.6

#### Setup a Virtualenv (optional)
##### Create a virtual environment

```commandline
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv venv_lips
```
##### Enter virtual environment
```commandline
source venv_lips/bin/activate
```

#### Install from source
```commandline
git clone https://github.com/IRT-SystemX/LIPS.git
cd LIPS
pip3 install -U .
cd ..
```

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
