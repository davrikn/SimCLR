#!/bin/bash

conda create SimCLR
conda activate SimCLR
python -m pip install -r requirements.txt
python 0_train.py
