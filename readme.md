# Multiclass Image Classification using CNN and Keras

## Project Overview:
This project aims to develop a Convolutional Neural Network (CNN) based model using Keras to classify images into multiple classes. The dataset used consists of images of various sceneries, and the goal is to predict the category of each image based on its features.

## Key Features:
- **Multiclass Classification**: The model is designed to classify images into 6 categories: glacier, mountain, sea, street, forest, and buildings.
- **CNN Architecture**: Utilizes a CNN model to capture hierarchical features from images.
- **Data Augmentation**: The dataset is enhanced using techniques such as rotation, shifting, and zooming to improve model generalization.
- **PCA Visualization**: Principal Component Analysis (PCA) is used to visualize the feature distribution of the dataset.
- **Model Evaluation**: The model's performance is evaluated based on test accuracy and loss, and predictions are made on unseen data.

## Technical Implementation:
1. **Data Preprocessing**:
   - Images are resized to a uniform size (150x150 pixels).
   - Image normalization and reshaping to ensure consistent input features for the model.
   - Labels are one-hot encoded using `LabelBinarizer`.
   
2. **Model Architecture**:
   - **Convolutional Layers**: Used to extract spatial features from the images.
   - **MaxPooling Layers**: Reduce dimensionality and improve computational efficiency.
   - **Dropout**: Used to avoid overfitting during training.
   - **Batch Normalization**: Ensures stable training by normalizing the input to each layer.
   - **LeakyReLU Activation**: Helps to overcome the “dying ReLU” problem by allowing small negative values.
   - **Fully Connected Layers**: Used to classify the extracted features into predefined categories using softmax activation.

3. **Model Training**:
   - The model is compiled using the Adam optimizer with a learning rate of 0.0005 and categorical cross-entropy loss function.
   - The model is trained for 70 epochs with a batch size of 128.
   
4. **Evaluation**:
   - After training, the model’s accuracy is tested using a separate test dataset.
   - Predictions are made, and accuracy is computed for each class.

## Project Run Through:
1. **Dataset Loading**: The dataset consists of images that are loaded into the model, resized, and normalized.
2. **PCA Visualization**: A PCA plot is generated to analyze the distribution of the classes and their separability.
3. **CNN Model**: A CNN model is built and trained on the dataset with various layers such as Conv2D, MaxPooling2D, and Dense layers.
4. **Model Training**: The model is trained with early stopping to avoid overfitting and reduce the training time.
5. **Test and Validation**: The model is tested on unseen data, and accuracy is evaluated.

## Achievements:
- Achieved a test accuracy of **78%** on unseen data.
- Successfully used CNN for multiclass classification with a moderate number of classes (6).
- Model predictions are evaluated and compared with actual labels for validation.

## How to Use This Project:
1. **Clone the Repository**:
   ```bash
      git clone https://github.com/Arpl14/Multiclass_image_classification_flaskapp-using-CNN-Keras.git
2. **Run app on local PC**:
   To run the flask app on your local computer **follow the steps from the file: Run_app_local_readme.md**  .You can download all the requirements.txt modules and setup the environment using the instructions. The preview above is how the app functions overall. 


## Conclusion:
The model provides a solid foundation for multiclass image classification tasks. While the current accuracy is decent, there is ample room for further improvement using techniques such as transfer learning (using VGG16, InceptionV3 etc), hyperparameter optimization, and regularization. With future enhancements, this project can achieve even better classification results and generalize well to unseen data.

-git clone https://github.com/Arpl14/Multiclass_image_classification_flaskapp-using-CNN-Keras.git
