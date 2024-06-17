# Low Light Image Enhancing using EnlightenGAN

## Description
This project implements a low light image enhancement algorithm using EnlightenGAN giving a peak signal to noise ration 16.94 . The implementation and dataset are based on the methods and resources described in the following articles:
- [EnlightenGAN: Deep Light Enhancement without Paired Supervision](https://arxiv.org/pdf/1906.06972v2)
- [Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation](https://arxiv.org/pdf/2112.14022)

EnlightenGAN is a Generative Adversarial Network (GAN) designed to improve the visibility and aesthetics of images captured in low light conditions. This repository provides the code and dataset for training and testing the model.

## Implementation Details
- **Training Data**: The model is trained using images from the `train/low` and `train/high` folders.
  - `train/low`: Contains low light images.
  - `train/high`: Contains corresponding well-lit images.

- **Testing Data**: The model takes input images from the `test/low` folder and generates the enhanced output.
  - `test/low`: Contains low light images for testing.
