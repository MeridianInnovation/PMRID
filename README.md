# Practical Deep Raw Image Denoising on Mobile Devices
The Tensorflow Reimplementation based the [Practical Deep Raw Image Denoising on Mobile Devices - ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf).

We will modify based on the implementation here.
## Content
  - [Install](#install)
  - [Getting Started](#getting-started)
  - [Running](#running)
    - [Result](#result)
  - [References](#references)

## Install
In a virtual environment, install all the necessary packages and libraries by running pip install -r requirements.txt at the root directory

## Getting Started

- Clone this repository and run on [Colab](https://colab.research.google.com/notebooks/intro.ipynb).

## Running

- Creating Data
  - Clone the repository [CycleISP](https://github.com/swz30/CycleISP) to generate pairs clean and noisy image.

  ```
  python generate_rgb_data.py --input_dir <path to clean images folder> --result_dir <path to output>
  ```

- Change path variables in Deep_Raw_Image_Denoising.ipynb

- Run file ipynb to train model.


### Result

This result acquired after training 20 epochs, input shape: 256x256x3. Number of pairs images is 2476.

## References
[1] [Practical Deep Raw Image Denoising on Mobile Devices - ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf).

[2] [SIDD Dataset](https://www.eecs.yorku.ca/~kamel/sidd/)
