
## 1. Project Overview

Project Title: Crop Pest Classification Using CNN-Based Models

Model Type: Resnet

Objective: Classification

Dataset Used: IP102 (Link: https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset)

Expected test evaluation for sanity check: 
8 Class configuration:
GoogleNet: (Accuracy - 66.68, F1-Score: 66.66)
VGG 16: (Accuracy - 70.67, F1-Score: 70.66)

102 Class configuration:
GoogleNet: (Accuracy - 61.07, F1-Score: 60.87)
VGG 16: (Accuracy - 62.80, F1-Score: 62.53)


------------------------------------------------------------

## 2. Repository Structure
project_root/
  class_sampler.py
  dataloader.py
  evaluation.py
  model_googlenet.py
  model_vggnet.py
  requirements.txt
  ckpt/                 (checkpoint goes here)
  dataset/                   (dataset goes here)
  README.md
```

------------------------------------------------------------

## 3. Dataset (Choose ONE of the 3 options)

### OPTION B — CUSTOM SPLITS (test split uploaded to Box)
Box Link to Dataset:
{{paste Box link}}


Where to place the dataset after download:
```
data/
   
```

## 4. Model Checkpoint

Box Link to Best Model Checkpoint:
https://usf.box.com/s/zu9m6gk4kizaoi6z7fy7l5ec1g0nyreq

Give access to:
yusun@usf.edu, kandiyana@usf.edu

Where to place the checkpoint after downloading:
```
ckpt/
   googlenet_ip8.pt
   googlenet_ip102.pt
   vgg16_ip8.pt
   vgg16_ip102.pt

```

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version: 3.10

How to install all dependencies (e.g. requirements.txt):

Using pip:
pip install -r requirements.txt

Using conda (creates env and installs):
```
conda create -n dlproj python=3.10
conda activate dlproj
pip install -r requirements.txt
```

------------------------------------------------------------

## 6. Running the Test Script

Command to run testing:
Model: GoogleNet, Configuration: 8 classes
python -m run_model --model_name googlenet --classes 8 --train_model 0

Model: GoogleNet, Configuration: 102 classes
python -m run_model --model_name googlenet --classes 102 --train_model 0

Model: VGG16, Configuration: 8 classes
python -m run_model --model_name vgg16 --classes 9 --train_model 0

Model: VGG16, Configuration: 102 classes
python -m run_model --model_name vgg16 --classes 102 --train_model 0

------------------------------------------------------------

## 7. Running the Training Script

Command to run training:
```
Model: GoogleNet, Configuration: 8 classes
python -m run_model --model_name googlenet --classes 8 --train_model 1

Model: GoogleNet, Configuration: 102 classes
python -m run_model --model_name googlenet --classes 102 --train_model 1

Model: VGG16, Configuration: 8 classes
python -m run_model --model_name vgg16 --classes 9 --train_model 1

Model: VGG16, Configuration: 102 classes
python -m run_model --model_name vgg16 --classes 102 --train_model 1


------------------------------------------------------------

## 8. Submission Checklist

- [ yes ] Dataset provided using Option A, B, or C and placed correctly.
- [ yes ] Model checkpoint linked and instructions for placement included.
- [ yes ] `requirements.txt` generated and Python version specified.
- [ yes ] Test command works.
- [ yes ] Train command works.

------------------------------------------------------------
