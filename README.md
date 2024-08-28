# Overview

This repository contains a framework for detecting mitotic figures from hematoxylin and eosin-stained whole-slide images. Details of the framework are introduced in the paper: OMG-Net: A Deep Learning Framework Deploying Segment Anything to Detect Pan-Cancer Mitotic Figures from Haematoxylin and Eosin-Stained Slides https://arxiv.org/abs/2407.12773

![Screenshot 2024-08-28 at 10 58 38](https://github.com/user-attachments/assets/100f5359-e1ba-46be-8d3b-f470b664a530)


# Installation

Install OMG-Net:

```
pip install git+https://github.com/SZY1234567/OMG-Net.git
cd OMG-Net
conda env create -f environment.yml
```

# Data
All images of mitotic figures and their nuclei contours dilated by the Segment Anything Model are available at:
https://figshare.com/projects/OMG-Net_A_Deep_Learning_Framework_Deploying_Segment_Anything_to_Detect_Pan-Cancer_Mitotic_Figures_from_Haematoxylin_and_Eosin-Stained_Slides/218617

The original images and annotations of the open-source datasets can be found via their own repositories:

ICPR: http://ludo17.free.fr/mitos_2012/index.html

TUPAC: https://github.com/DeepMicroscopy/TUPAC16_AlternativeLabels

CCMCT: https://github.com/DeepMicroscopy/MITOS_WSI_CCMCT

CMC: https://github.com/DeepMicroscopy/MITOS_WSI_CMC

MIDOG++: https://github.com/DeepMicroscopy/MIDOGpp 



# Usage
## Quick Start
To run the model on a whole slide image:
```
python Inference/CompleteInference.py /path/to/image.svs
```
We are developing the front end for implementing the model on our website https://www.octopath.ai/
## Training 
Training on only the mitotic figures and mitotic-like figures
```
python Training/Classifier.py Configs/Classifier_config.ini
```
Training on other objects segmented by the Segment Anything Model
```
python Training/Classifier_all_cells.py Configs/Classifier_all_cells_config.ini
```

Please follow the data generation steps to generate the NRRD files required in training. 
Please also change the paths to the data in the configuration files accordingly.

## Data Generation
The training data will be generated from the source format in different datasets by:
```
python Utils/GeneratingNRRD.py
```
Currently, the whole slide images for the cases in the STMF dataset are not publicly available. 

