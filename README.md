# Advanced Weed Detection using CNN under Challenging Weather Conditions  

# Overview
This project focuses on enhancing the robustness of weed detection models by improving the accuracy of Convolutional Neural Networks (CNNs) in challenging weather conditions. Traditional weed detection models, typically trained on images captured under normal weather conditions, often struggle to maintain accuracy when analyzing images taken in adverse weather such as rain, snow, or varying lighting conditions. Our primary objective is to bolster model resilience across diverse weather scenarios by enabling them to learn and adapt to real-world weather patterns.

To achieve this goal, we employ data augmentation techniques to simulate challenging weather conditions. By generating augmented images, we expand the diversity of the training dataset. This augmented dataset, combined with the original images, forms the basis for training CNN models aimed at enhancing robustness. We have selected four CNN models for training: 3-Layer CNN, ResNet-50, VGG16, InceptionV3, and InceptionResNetV2. Each model is trained twice: once using the original dataset without augmentation and once with the augmented images. This dual-training approach enables us to directly compare model performance under varying data conditions and assess the impact of data augmentation on accuracy. The model delivering the highest accuracy after training with the augmented dataset will be selected for fine-tuning through hyperparameter optimization.

The benefits of our project are far-reaching. By facilitating more accurate decision-making, we can mitigate the occurrence of False Positives (misclassifying crops as weeds) and False Negatives (failing to detect weeds), both of which have significant consequences for agriculture. Reduced instances of False Positives minimize food wastage resulting from the unnecessary elimination of crops, while fewer False Negatives help prevent extensive crop damage caused by undetected weeds. Ultimately, our project aims to contribute to a more efficient and sustainable agricultural ecosystem.

# Data Source  
The dataset used in this project consists of 1,040 images and a labels.csv indicating whether the image is labled as weed or non-weed. However, there is a mismatch between the number of images and the number of labels, so the unlabeled images were removed to ensure accurate data preprocessing.

The images are stored in the train_images directory, and the labels are provided in the labels.csv file. Many thanks to the data source from: 

[Weed and Non-weed Image Dataset](http://images.cocodataset.org/zips/train2014.zip](https://github.com/wittyicon29/WeedWatch-Weed-Detection-using-CNN/tree/main/Dataset) provided by `@wittyicon29` on GitHub

# Data Preprocessing  
All of the images are resized to a fixed size of 224x224 pixels and converted to an array format. The cv2 library is used for reading and preprocessing the images. By standardizing the images, they are formatted suitably to be inputted into the CNN model.

Also, we remove the unlabeled images by identifying them with `unmatched_images = [img for img in image_filenames if img not in label_filenames]`. After the remove, we check the number of images in the dataset and it is 916, which matches the number of labels.

Another important step in preprocessing is data augmentation. To mimick the weather effects such as rain and snow, we use Albumentations, which is a powerful open-source image augmentation library that provides flexible and efficient framework for data augmentation in computer vision. Here is what we do:

```bash
import albumentations as A

transform = A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1)
transform = A.RandomSnow(brightness_coefficient=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1)
```

Where we can adjust the parameters to generate images with different levels of weather effect. Here are some examples:

![data_agumentation_rain](https://github.com/HedwigO/AdvancedWeedDetection/assets/97476561/9866cb9b-938a-4763-b5d3-602d52ac2646)
![data_agumentation_snow](https://github.com/HedwigO/AdvancedWeedDetection/assets/97476561/bc732224-0154-4587-b66f-d3ac27e29229)

We applied data augumentation to 50% of the original dataset where 25% of them are rain and 25% of them are snow.

# Model Training (Model Architecture)  
The original data set was divided into a training set, validation set, and a test set. There are 2 versions of the training set. One is the original data set and the other is an augmented data set.  

We trained 4 different models:  
- 3-Layer CNN  
- ResNet50  
- InceptionV3  
- InceptionResNetV2 

The performance of each model was evaluated based on test accuracy and test loss. Here are the architectures of the models:

## 3-Layer CNN  
- Input Layer: Takes input of shape `input_shape`.
- Conv2D Layer: 128 filters, 3x3 kernel size, ReLU activation.
- MaxPooling2D Layer: 2x2 pool size.
- Conv2D Layer: 64 filters, 3x3 kernel size, ReLU activation.
- MaxPooling2D Layer: 2x2 pool size.
- Conv2D Layer: 32 filters, 3x3 kernel size, ReLU activation.
- MaxPooling2D Layer: 2x2 pool size.
- Flatten Layer: Flattens the input.
- Dense Layer: 64 units, ReLU activation.
- Dense Layer: `num_classes units`, softmax activation.
<img width="968" alt="3 Layer CNN_architecture" src="https://github.com/HedwigO/AdvancedWeedDetection/assets/97476561/4bfbf06f-070f-42ca-96c8-95cb9ca954c0">

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
<img width="996" alt="image" src="https://github.com/HedwigO/AdvancedWeedDetection/assets/169214308/10466a19-bbe7-488f-a84b-16e6e1ac73be">


## InceptionResNetV2
Input Layer  
Stem (Convolutional, Batch Norm, ReLU activations)  
Inception-ResNet Blocks x10 (1x1, 3x3, 7x1, 1x7 filters, Max Pooling, Concatenation, Residuals, Activation Functions, Output)  
Reduction Blocks  
Average Pooling Layer  
Fully Connected Layers  
Output Layer  
<img width="405" alt="image" src="https://github.com/HedwigO/AdvancedWeedDetection/assets/169214308/8805b809-ceff-4a43-b813-6bcc9c2482e9">

## Baseline and Augmented Models Test Results
<img width="912" alt="image" src="https://github.com/HedwigO/AdvancedWeedDetection/assets/169214308/3008ae7f-9819-4b5c-8211-d4418643599b">

# Hyperparameter Tuning
After training the models, we found that the ResNet50 performed the best with an accuracy of around 96%.  We conducted hyperparameter tuning using 3 different methods for optimizing this model

Grid Search: explores all specified hyperparameter combinations  
Random Search: randomly samples hyperparameter combinations to explore a wider range  
Bayesian Search: adapts to previous evaluation results to identify the next hyperparameter and determine the optimal hyperparameter configuration  


