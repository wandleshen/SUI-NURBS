# SUI-NURBS: Speed-Up the Intersection between Two NURBS Surfaces

## Introduction

This repository contains the source code for reimplementing the algorithm in Python for speeding up the intersection between two NURBS surfaces. The original algorithm was proposed by [Lin et al.](https://ieeexplore.ieee.org/abstract/document/6616551/) in 2014.

## Installation

Requires Python 3.6+, VS2019+, Cuda 11.3+ and PyTorch 1.10+

Tested in Anaconda3 with Python 3.10 and PyTorch 2.2.1

### One time setup (Windows)

Install the [Cuda toolkit](https://developer.nvidia.com/cuda-toolkit) (required to build the PyTorch extensions). Pick the appropriate version of PyTorch compatible with the installed Cuda toolkit. Below is an example with Cuda 11.8

```bash
conda create -n sui-nurbs python=3.10
conda activate sui-nurbs
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Usage

Currently, we have two demos available for running. You can use the following command to run the demos. And the detail of `args` will be shown below.

```bash
python demo.py [-args args]
python demo_perftest.py [-args args]
```

The `demo_perftest.py` will test the performance of the algorithm and output the `cProfile` stats. The `demo.py` will output the intersection curves between two NURBS surfaces. We only support one NURBS surface intersects w/ its transformed form.

### Arguments

- `-f`/`--filename` (str): The path to the NURBS surface control points in numpy array format. You can generate the file w/ `np.save(filename, array)`, please make sure `array.shape=[a, b, 4]`. Default: None for `demo.py` and `data/plain.npy` for `demo_perftest.py`
- `-m`/`--u_intervals` (int): The number of AABBs' intervals in the u-direction. Default: `1024` for `demo.py` and `32` for `demo_perftest.py`
- `-n`/`--v_intervals` (int): The number of AABBs' intervals in the v-direction. Default: `1024` for `demo.py` and `32` for `demo_perftest.py`
- `-p`/`--u_degree` (int): The degree of the NURBS surface in the u-direction. Default: `3`
- `-q`/`--v_degree` (int): The degree of the NURBS surface in the v-direction. Default: `3`
- `-s`/`--scalar` (float): The scalar for the knotvectors in order to avoid the `nan` problem. Default: `25.0` for `demo.py` and `1.0` for `demo_perftest.py`

### Examples

```bash
python demo.py -f data/cross.npy
```

With the above command, you can get the intersection curves between two NURBS surfaces called `cross.npy` and the result will be shown in a window like the img below.

![](./assets/cross.png)
