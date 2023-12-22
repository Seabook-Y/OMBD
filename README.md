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
## Training
* Train the model
```
python main.py
```
## Ethical Proof
All experimental procedures were performed in accordance with the Guidance on the Operation of the Animals (Scientific Procedures) Act, 1986 (UK) and approved by the Queen’s University Belfast Animal.
## Contact
For any discussions, feel free to contact: xy179@leicester.ac.uk
