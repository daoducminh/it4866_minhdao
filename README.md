# Detecting Traffic Light using Faster R-CNN

## Installation

1. Initialize Python virtual environment (`python3.6`):

- `virtualenv --python=/usr/bin/python3.6 .virtualenvs`

2. Install python modules:

- `.virtualenvs/bin/pip3 install .`

3. Install `pycocotools`

- `.virtualenvs/bin/pip3 install pycocotools`

4. Install `protobuf-compiler`:

- `sudo apt-get install protobuf-compiler -y`

5. Clone `Tensorflow Model Garden` project:

- `git clone https://github.com/tensorflow/models.git`

6. Build `tensorflow-object-detection-api`:

```bash
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```

- Add Libraries to `PYTHONPATH`: Add these lines to `.virtualenvs/bin/activate`
    ```bash
    export PYTHONPATH=$PYTHONPATH:./models/research:./models/research/slim
    ```
    then run: `source .virtualenvs/bin/activate`

- Testing the installation: `.virtualenvs/bin/python3 models/research/object_detection/builders/model_builder_tf1_test.py`

## Preparing dataset

- Download dataset from this url: [Traffic Light Dataset](https://bit.ly/30sw7iy)