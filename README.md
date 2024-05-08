# Advanced Weed Detection using CNN under Challenging Weather Conditions

This project aims to improve model performance (accuracy and loss) especially under special weather conditions using CNN architectures and hyperparameter tuning.

It is going to compare the performance of several baseline models, select the model with best performance, and try different hyperparameter tuning methods on the selected model to find the optimized combination of hyperparameters.

## Data Source

The dataset used in this project consists of 1,040 images and a labels.csv indicating whether the image is labled as weed or crop. However, notice that there is a mismatch between the number of images and the number of labels, so we will need to drop those unlabled images.

The images are stored in the train_images directory, and the labels are provided in the labels.csv file. Here is the data from:

- [Weed Dataset]([http://images.cocodataset.org/zips/train2014.zip](https://github.com/wittyicon29/WeedWatch-Weed-Detection-using-CNN/tree/main/Dataset)) provided by `@wittyicon29` on GitHub
