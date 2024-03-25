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

Currently, we have two demos available for running. You can use the following command to run it, or modify the `ctrlpts` and `ctrlpts4d` in `demo.py` or `demo_perftest.py` to modify the shape of the surface.

```bash
python demo.py
python demo_perftest.py
```

The `demo.py` will generate the intersection curve between two NURBS surfaces and visualize them. The `demo_perftest.py` will test the performance of the algorithm and output the `cProfile` stats.
