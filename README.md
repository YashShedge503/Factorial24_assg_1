
# Fashion MNIST Image Classifier

This project demonstrates how to classify images of clothing items from the Fashion MNIST dataset using a Random Forest Classifier implemented with Scikit-learn.
(We can use pytorch also, in real world normally we use CNN with pytorch for image classification).

## What this project does

- Loads the Fashion MNIST dataset which consists of 28x28 grayscale images of 10 clothing categories.
- Preprocesses the data by flattening and normalizing the images.
- Trains a Random Forest model on the data.
- Evaluates the model's accuracy and displays a confusion matrix.

## Why Random Forest?

- Random Forest handles high-dimensional feature spaces well.
- It is robust to overfitting.
- Provides interpretable results and does not require extensive preprocessing.

Other models like Logistic Regression are too simplistic for image classification, and while deep learning models would perform better, Random Forest is a strong classical baseline that shows how tabular methods can handle image data when flattened.

## Impact and Problem Solved

This model shows how basic machine learning techniques can automate the process of labeling fashion images. This is useful in e-commerce platforms for product tagging, search, and recommendations (also build model on recommendation system based on reviews).

## Run Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python fashion_mnist_classifier.py
   ```

The script will output accuracy, a detailed classification report, and display a confusion matrix heatmap.
    
