# Practical Deep Raw Image Denoising on Mobile Devices
The denoisers repo is [here](https://github.com/MeridianInnovation/Denoisers/blob/main/README.md). `The repo contains the information about all the models.`

We use .ipynb in colab to perform computation (inference and occasionally training). For the model in pytorch and tensorflow, all the classes, functions and testing are implemented in .py files in this github repo.

## Content
  - [Preparation](#preparation)
  - [Training](#training)
  - [Inference](#inference-and-result)
  - [References](#references)

## Preparation
### Install
In a virtual environment, install all the necessary packages and libraries by running pip install -r requirements.txt at the root directory

### Model
`The difference with the original PMRID is to use three encoders and decoders because of smaller resolution ?`

`Compared with the original PMRID model design, we delete encoder stage 4 because it is the deepest. Then we delete decoder stage 1 because of corresponding skip connection.` 

More information can be found at [here](https://github.com/MeridianInnovation/Denoisers).

### Project Architecture
The project has two versions. One is the model developed in tensorflow (.py files). Another is the model developed in pytorch (*_torch.py files). As of today (2024-10-28), the pytorch model is more well-developed. The general structure is below:

```
├── checkpoints # Directory for saving model checkpoints
├── configs # Configuration files for hyperparameters
├── data # Pre-processed dataset
├── logs # Logs for hyperparameter changes
├── models # Pre-trained and trained models
├── src # Source code
│ ├── data # Data loading and preprocessing scripts
│ ├── model # Model architecture
│ ├── train # Training scripts
│ └── utils # Utility functions
└── tests # test cases
```


### Dataset
We use FLIR dataset and find it [here](https://drive.google.com/file/d/1XFL-vH2puregx8_ApuYVxDrQLzHE9RTQ/view?usp=drive_link). The trainning set has around 110,000 pairs of images (70%). The validation set has around 11,000 pairs of images (7%). The testing set has around 37,000 pairs of images (23%). You can use a reducer script [here](https://github.com/danielliu-meridian/image-processing/blob/main/scripts/image_dataset_reducer.py) to reduce the size of dataset by 2, 4 or 8. You can find a dataset with reducer size 8 [here](https://drive.google.com/file/d/1kWvuOn_u4gQKIUjpKU4fzdPZWWEntJzH/view?usp=sharing).

## Training
There are two ways to train the model. One is locally, another is colab.

- Load pre-processed dataset from [drive](https://drive.google.com/file/d/1kWvuOn_u4gQKIUjpKU4fzdPZWWEntJzH/view)

- Change path variables in Denoiser.ipynb

- Run file ipynb to train model.

## Inference and Result

Clone this repository and run on [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MJnoV_RLyxyodpH9mvuWu7paNOIbbbd9?usp=sharing)

The result in colab was acquired after training 20 epochs.

## References
[1] [Practical Deep Raw Image Denoising on Mobile Devices - Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf).

[2] [Loss Functions for Image Restoration with Neural Networks - Loss](https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf)

[3] [FLIR Kaggle - Dataset](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset)
