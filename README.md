# Advanced Weed Detection using CNN under Challenging Weather Conditions  
This project aims to improve model performance (accuracy and loss) especially under special weather conditions using CNN architectures and hyperparameter tuning.  
It is going to compare the performance of several baseline models, select the model with best performance, and try different hyperparameter tuning methods on the selected model to find the optimized combination of hyperparameters.

# Data Source  
The dataset used in this project consists of 1,040 images and a labels.csv indicating whether the image is labled as weed or crop. However, notice that there is a mismatch between the number of images and the number of labels, so we will need to drop those unlabled images.

The images are stored in the train_images directory, and the labels are provided in the labels.csv file. Here is the data from:

- [Weed Dataset]([http://images.cocodataset.org/zips/train2014.zip](https://github.com/wittyicon29/WeedWatch-Weed-Detection-using-CNN/tree/main/Dataset)) provided by `@wittyicon29` on GitHub

# Preprocessing  
All of the images are resized to a fixed size of 224x224 pixels and converted to an array format. The cv2 library is used for reading and preprocessing the images. By standardizing the images, they are formatted suitably to be inputted into the CNN model.

# Models  
We trained 4 different models:  
3-Layer CNN  
ResNet50InceptionV3   
InceptionResNetV2  

# Model Architecture  
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
Output Layers  



