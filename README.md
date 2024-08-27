# MR-Brain-Image-Reconstruction

## Overview

The "Brain MRI Reconstruction" project focuses on reconstructing high-quality brain MRI images from corrupted or partial data using deep learning techniques. This repository contains the code and pipeline for training a U-Net model specifically adapted to handle and improve MRI data stored in the .t5 format.

## Data Preprocessing

Data Loading

The project uses the .t5 format for both input and target MRI data. This format requires specialised handling for loading and preprocessing:

- Loading: Data is loaded using custom Python scripts that interface with libraries capable of handling .t5 files, extracting MRI data effectively for further processing.

Preprocessing Steps

The preprocessing routine includes several critical steps to prepare the MRI data for training:

- Normalisation: MRI data from .t5 files is normalised to have a consistent range of values, typically between 0 and 1, which helps in stabilising the neural network training by providing a common scale.
- Rescaling: Images are rescaled to a fixed size to ensure that the input to the U-Net is uniform. This step typically involves adjusting the resolution of the images to fit the model’s expected input size without distorting the important features of the brain structures.

## Model Architecture

U-Net Configuration

The U-Net model used in this project is configured for high-fidelity image reconstruction:

- Encoder: The encoder part reduces the dimensionality of the input image while capturing the essential features. This is achieved through a series of convolutional layers paired with max pooling.
- Decoder: The decoder part reconstructs the detailed MRI image from the encoded representation using transpose convolutions that increase the spatial resolution of the output.
- Skip Connections: Critical for accurate reconstruction, skip connections are used to carry information directly from the encoder to the decoder, helping in restoring fine details lost during down-sampling.

## Training

Optimisation and Loss Functions

- Optimiser: The Adam optimiser is used for its effectiveness in handling sparse gradients and adapting the learning rate, which is crucial for image reconstruction tasks.
- Loss Function: Mean Squared Error (MSE) is commonly used as a loss function in image reconstruction to minimize the difference between the predicted and actual MRI images.

## Training Strategy

- Batch Processing: Images are processed in batches to optimise GPU resources and improve training dynamics.
- Data Augmentation: Techniques such as rotation and scaling are employed to artificially expand the training dataset, which helps the model generalise better to new, unseen data.

## Evaluation and Results

Visualisation

Visual comparisons between corrupted input images, reconstructed images, and ground truth are provided to demonstrate the model’s effectiveness.

![Picture1](https://github.com/user-attachments/assets/252c6421-bad3-4f50-971b-f9648cdefdd7)

## Conclusion

This project showcases the application of U-Net in the reconstruction of brain MRI images from partial or corrupted scans, addressing challenges in medical imaging such as noise reduction and detail enhancement. The models and methods developed here can be adapted to other types of medical image reconstruction tasks.
