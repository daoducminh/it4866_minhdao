# Classifying Traffic Light using CNN

## Installation

1. Initialize Python virtual environment (`python3.6`):

- `virtualenv --python=/usr/bin/python3.6 .virtualenvs`

2. Install python modules:

- `.virtualenvs/bin/pip3 install .`

## Preparing dataset

- Download dataset from this url: [Traffic Light Dataset](https://bit.ly/30sw7iy)
- Extracted dataset has structure:
```
data/
----/test_images/
----/udacity/
```

## Running notebook

- Start jupyter notebook: `jupyter lab`
- Run `train_model.ipynb` to train model
- Run `test_model.ipynb` to evaluate model with test data.