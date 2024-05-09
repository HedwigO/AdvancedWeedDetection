# Advanced Weed Detection using CNN under Challenging Weather Conditions  

# Overview
This project focuses on enhancing the robustness of weed detection models by improving the accuracy of Convolutional Neural Networks (CNNs) in challenging weather conditions. Traditional weed detection models, typically trained on images captured under normal weather conditions, often struggle to maintain accuracy when analyzing images taken in adverse weather such as rain, snow, or varying lighting conditions. Our primary objective is to bolster model resilience across diverse weather scenarios by enabling them to learn and adapt to real-world weather patterns.

To achieve this goal, we employ data augmentation techniques to simulate challenging weather conditions. By generating augmented images, we expand the diversity of the training dataset. This augmented dataset, combined with the original images, forms the basis for training CNN models aimed at enhancing robustness.

The benefits of our project are far-reaching. By facilitating more accurate decision-making, we can mitigate the occurrence of False Positives (misclassifying crops as weeds) and False Negatives (failing to detect weeds), both of which have significant consequences for agriculture. Reduced instances of False Positives minimize food wastage resulting from the unnecessary elimination of crops, while fewer False Negatives help prevent extensive crop damage caused by undetected weeds. Ultimately, our project aims to contribute to a more efficient and sustainable agricultural ecosystem.

# Data Source  
The dataset used in this project consists of 1,040 images and a labels.csv indicating whether the image is labled as weed or crop. However, notice that there is a mismatch between the number of images and the number of labels, so we will need to drop those unlabled images.

The images are stored in the train_images directory, and the labels are provided in the labels.csv file. Here is the data from:

- [Weed Dataset]([http://images.cocodataset.org/zips/train2014.zip](https://github.com/wittyicon29/WeedWatch-Weed-Detection-using-CNN/tree/main/Dataset)) provided by `@wittyicon29` on GitHub

# Model Training (architecture)  
The original data set was divided into a training set, validation set, and a test set. There are 2 versions of the training set. One is the original data set and the other is an augmented data set.  

We trained 4 different models:  
3-Layer CNN  
ResNet50  
InceptionV3  
InceptionResNetV2 

The performance of each model was evaluated based on test accuracy and test loss. 

## 3-Layer CNN  
Input Layer  
Convolutional Layer (with activation function)  
Pooling Layer  
Convolution Layer (with activation function)    
Pooling Layer  
Flatten Layer  
Fully Connected Layer  
Output Layer  

## ResNet50  
Input Layer  
Initial Convolutional Layer (7x7)  
Max Pooling Layer (3x3, stride of 2)  
Residual Blocks x16 (Input, 1x1, 3x3 filters, Batch Norm, ReLU activations, Output)  
Average Pooling Layer  
Fully Connected Layers  
Output Layer  

## InceptionV3  
Input Layer  
Stem (Convolutional, Batch Norm, ReLU activations)  
Inception Modules x11 (1x1, 3x3, 5x5 filters, Max Pooling, Concatenation)  
Auxiliary Classifiers (Optional)  
Average Pooling Layer  
Fully Connected Layers  
Output Layer  

## InceptionResNetV2
Input Layer  
Stem (Convolutional, Batch Norm, ReLU activations)  
Inception-ResNet Blocks x10 (1x1, 3x3, 7x1, 1x7 filters, Max Pooling, Concatenation, Residuals, Activation Functions, Output)  
Reduction Blocks  
Average Pooling Layer  
Fully Connected Layers  
Output Layer  

# Hyperparameter Tuning
After training the models, we found that the ResNet50 performed the best with an accuracy of around 96%.  We conducted hyperparameter tuning using 3 different methods for optimizing this model

Grid Search: explores all specified hyperparameter combinations  
Random Search: randomly samples hyperparameter combinations to explore a wider range  
Bayesian Search: adapts to previous evaluation results to identify the next hyperparameter and determine the optimal hyperparameter configuration  


