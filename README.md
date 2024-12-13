# CNN Image Classification Project

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images of cats and dogs. 

The project is structured into multiple steps, starting from data preprocessing to building, training, and testing the CNN model.

# Introduction

This project uses deep learning techniques to classify images into two categories: cats and dogs.

The model is trained on a labeled dataset and evaluated for accuracy. 

The implementation is done using TensorFlow and Keras libraries.

# Key features of the project:

Preprocessing the dataset to make it suitable for model training.

Designing a custom CNN architecture.

Training and evaluating the model on separate datasets.

Making predictions on new images.

# Project Structure

The project contains the following files and folders:

<img width="380" alt="image" src="https://github.com/user-attachments/assets/8594d815-4e8b-4be4-8e93-acaed35f4ef0" />

CNN for Image Classification.ipynb: Jupyter Notebook with the complete implementation.

dataset/: Contains the dataset organized into training and testing sets.

training_set/: Images for training the model.

test_set/: Images for evaluating the model.

single_prediction/: Folder for testing new images.

README.md: Project documentation (this file).

# Obtain the dataset:

Option 1: Contact me at qss1774@outlook.com to request the dataset.

Option 2: Use your own dataset and place it in the dataset/ directory.

# Steps to Run the Project

## Preprocess the Data:

Normalize pixel values to a range of [0, 1].

Resize images to 64x64 pixels for compatibility with the CNN.

## Build the CNN:

Initialize the CNN.

Add convolutional layers, pooling layers, and a flattening layer.

Add a fully connected layer and an output layer with a sigmoid activation function.

<img width="677" alt="image" src="https://github.com/user-attachments/assets/c688cac2-3e8f-4c6f-a718-7c6c39ca4809" />

## Train the Model:

Compile the model with the Adam optimizer and binary cross-entropy loss function.

Train the model on the training dataset, and validate it on the test dataset.

<img width="1002" alt="image" src="https://github.com/user-attachments/assets/24d53601-1632-433b-a042-5120e6618e36" />

## Make Predictions:

Load and preprocess new images from the single_prediction/ folder.

Use the trained model to predict whether the image is of a cat or a dog.

## Visualize Results:

Plot training and validation accuracy/loss using Matplotlib.

# Results

## Training Accuracy & Validation Accuracy: 

<img width="584" alt="image" src="https://github.com/user-attachments/assets/a74993e2-ff6c-4ba0-8190-0250bc779512" />

<img width="593" alt="image" src="https://github.com/user-attachments/assets/2400ded4-0c41-49e3-af9b-56ea8a87f45d" />

As training proceeds, the training loss decreases significantly, eventually approaching 0.2, as expected.
The verification loss decreased initially, but fluctuated more obviously in the later period, and finally remained at around 0.4.

The current model performs well on the training set, but performs poorly on the validation set, showing a certain tendency to overfitting.

## Prediction Examples:

<img width="537" alt="image" src="https://github.com/user-attachments/assets/96a109ea-a422-42a3-b826-cef95d84922c" />

Image 1: Predicted as Dog.        Image 2: Predicted as Cat.

From the prediction example, the model can correctly predict "dog" and "cat"

# Dataset Access
The dataset used in this project contains labeled images of cats and dogs. Due to size constraints, it is not uploaded directly to this repository.

If you need access to the dataset, please contact me via email at qss1774@outlook.com.

# Improvement suggestions

## Data enhancement:

To alleviate overfitting, we can enhance the dataset (such as flipping, rotating, cropping, etc.).

## Reduce fully connected layer parameters:

We can try to add a global average pooling layer before the fully connected layer to reduce the number of parameters.

## Adjust model depth:

If the amount of data is sufficient, we can try to increase the depth of the convolution layer to extract more features.

## Use transfer learning:

Consider using a pre-trained CNN model (such as VGG, ResNet, etc.) and fine-tune it based on the original model.

# Medium.com

Here is the link about this project I posted on Medium:

https://medium.com/@19976011774qss/building-a-convolutional-neural-network-cnn-for-image-classification-2eab2e0076b5




