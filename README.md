# `DarkFarseer` Artifact

**This repository is the official Pytorch implementation of "DarkFarseer: Robust Spatio-temporal Kriging under Graph Sparsity and Noise".**

## Instructions

```c++
├── checkpoints  // Model cache
│   ├── AIR36
│   ├── ...
├── experiments  // Experiment file
│   └── train_kriging.py
│   ├── ...
├── networks  // Model file
│   ├── Kriging_model.py  
│   └── layers 
├── preprocessing  // Data preprocessing
│   ├── process_AIR36.py
│   ├── ...
├── scripts  // Executable bash script
│   ├── train_AIR36.sh
│   ├── ...
└── utils  // Utility file
    ├── utils.py
    └── ...
```

## Environment Setup

```bash
$ conda create -n darkfarseer python=3.10.9
$ conda activate darkfarseer
$ pip install -r requirements.txt
```

## Dataset Preparation
Place execute:

```bash
$ unzip datasets.zip
```

## Run Darkfarseer

```bash
$ python -m experiments.train_kriging
```

## Acknowledgements
Our code is built on top of the following codebase:
* [Kaimaoge/IGNNK](https://github.com/Kaimaoge/IGNNK)
* [VAN-QIAN/CIKM23-HIEST](https://github.com/VAN-QIAN/CIKM23-HIEST/tree/main)