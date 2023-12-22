# ONLINE MOUSE BEHAVIOR DETECTION BY HISTORICAL DEPENDENCY AND TYPICAL INSTANCES
This repository provides a PyTorch implementation of the paper "ONLINE MOUSE BEHAVIOR DETECTION BY HISTORICAL DEPENDENCY AND TYPICAL INSTANCES".
![image](https://github.com/Seabook-Y/OMBD/blob/main/pipeline.png)
## Requirements
* python 3.9.7
* pytorch 1.10.2
* Torchvision 0.11.3
## Data
* The processed mouse optic flow features and RGB features should be placed under data/PDMB. As the PDMB dataset is private, we cannot share it.
* Typical instances can be extracted by running 'scripts/generate-KF.py'.
