# Practical Deep Raw Image Denoising on Mobile Devices
The Tensorflow Reimplementation based the [Practical Deep Raw Image Denoising on Mobile Devices - ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf).

`The difference with the original PMRID is to use three encoders and decoders because of smaller resolution ? . Compared with the original PMRID model design, we delete encoder stage 4 because it is the deepest. Then we delete decoder stage 1 because of corresponding skip connection.` More information can be found at [here](https://github.com/MeridianInnovation/Denoisers).

## Content
  - [Install](#install)
  - [Training](#training)
  - [Inference](#inference)
  - [Result](#result)
  - [References](#references)

## Install
In a virtual environment, install all the necessary packages and libraries by running pip install -r requirements.txt at the root directory

## Training
There are two ways to train the model. One is locally, another is colab.

- Load pre-processed dataset from [drive](https://drive.google.com/file/d/1kWvuOn_u4gQKIUjpKU4fzdPZWWEntJzH/view)

- Change path variables in Denoiser.ipynb

- Run file ipynb to train model.

## Inference

Clone this repository and run on [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MJnoV_RLyxyodpH9mvuWu7paNOIbbbd9?usp=sharing)

## Result

This result acquired after training 20 epochs.

## References
[1] [Practical Deep Raw Image Denoising on Mobile Devices - ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf).

[2] [FLIR Dataset - Kaggle](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset)
