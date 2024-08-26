# Overview

This repository contains a framework for detecting mitotic figures from hematoxylin and eosin-stained whole-slide images. Details of the framework are introduced in the paper: OMG-Net: A Deep Learning Framework Deploying Segment Anything to Detect Pan-Cancer Mitotic Figures from Haematoxylin and Eosin-Stained Slides https://arxiv.org/abs/2407.12773

# Installation

Install OMG-Net:

```
pip install git+https://github.com/SZY1234567/OMG-Net.git
cd OMG-Net
conda env create -f environment.yml
```

# Data
https://figshare.com/projects/OMG-Net_A_Deep_Learning_Framework_Deploying_Segment_Anything_to_Detect_Pan-Cancer_Mitotic_Figures_from_Haematoxylin_and_Eosin-Stained_Slides/218617

# Usage
## Quick Start
To run the model on a whole slide image:
```
python Inference/CompleteInference.py /path/to/image.svs
```
