# Multiclass Image Classification using CNN and Keras


https://github.com/user-attachments/assets/96f0c48a-a60c-4e3b-b9b2-7eb34d9f404d


## Project Overview:
This project aims to develop a Convolutional Neural Network (CNN) based model using Keras to classify images into multiple classes. The dataset used consists of images of various sceneries, and the goal is to predict the category of each image based on its features.

## Key Features:
- **CNN Model**: Uses a Convolutional Neural Network (CNN) for feature extraction and classification.
- **Flask Web Application**: A user-friendly interface to upload images and get predictions.
- **Model Saving**: The trained model is saved for later use, enabling predictions without retraining.
- **Localhost Hosting**: The app is hosted locally, allowing users to test the model interactively.
- **Data Augmentation**: The dataset is augmented with techniques such as rotation, zooming, and shifting to improve the model's generalization.

## Technical Implementation:
1. **Data Preprocessing**:
   - **Image Loading**: The images from the dataset are loaded, resized (to 150x150 pixels), and normalized.
   - **Label Encoding**: The labels are one-hot encoded using the `LabelBinarizer` from `sklearn`.
   - **Data Augmentation**: Techniques like rotation, width/height shift, shear, zoom, and horizontal flip are applied to improve model performance.

2. **CNN Model Architecture**:
   - **Convolutional Layers**: Multiple `Conv2D` layers are used to extract spatial features from the images.
   - **MaxPooling Layers**: `MaxPooling2D` layers are added to reduce spatial dimensions and make the model computationally efficient.
   - **Batch Normalization**: Batch normalization layers are used to normalize the activations of the network, improving training stability.
   - **Dropout**: Dropout layers are added to prevent overfitting by randomly disabling neurons during training.
   - **Fully Connected Layers**: After flattening the output from convolutional layers, fully connected layers are used for classification.
   - **Softmax Output**: The model outputs probabilities for each class using the softmax activation function.

3. **Model Training**:
   - **Optimizer**: The Adam optimizer is used for training the model with a learning rate of 0.0005.
   - **Loss Function**: Categorical Cross-Entropy loss function is used for multiclass classification.
   - **Epochs**: The model is trained for 70 epochs with a batch size of 128.
   - **Early Stopping**: Early stopping is used to prevent overfitting and stop the training process when the validation loss stops improving.

4. **Model Saving**:
   - The trained model is saved using `model.save()` as `intel_image.h5`, making it ready for deployment in the Flask application.

5. **Flask App Deployment**:
   - A Flask web application is developed to allow users to upload images and get predictions.
   - **GET Request**: Displays the home page with a default image when accessed via a GET request.
   - **POST Request**: When a user uploads an image, the image is processed, passed through the model, and the predicted class label is returned.
   - **Image Upload Handling**: The image is saved in the server’s upload directory and preprocessed before prediction.

## Project Run Through:
1. **Data Loading**: The Intel Image Dataset is loaded, preprocessed (resized and normalized), and augmented.
2. **Model Training**: The CNN model is trained on the dataset for 70 epochs. Early stopping is applied to avoid overfitting.
3. **Model Saving**: After training, the model is saved to a file (`intel_image.h5`).
4. **Flask Application**:
   - **Home Route (GET)**: Renders a page with an image upload form.
   - **Prediction Route (POST)**: Processes the uploaded image, uses the model to predict the class, and displays the result.

5. **Testing**: The model is tested on a separate test dataset, and accuracy is calculated.

## Achievements:
- Achieved a **78% test accuracy** on unseen data.
- Successfully built a CNN-based model capable of classifying images into six classes.
- Deployed the trained model on a Flask web application for real-time predictions.
- Implemented data augmentation to enhance model generalization.
- Model predictions are displayed on the web interface after uploading an image.


## How to Use This Project:
1. **Clone the Repository**:
   ```bash
      git clone https://github.com/Arpl14/Multiclass_image_classification_flaskapp-using-CNN-Keras.git
2. **Run app on local PC**:
   To run the flask app on your local computer **follow the steps from the file: Run_app_local_readme.md**  .You can download all the requirements.txt modules and setup the environment using the instructions. The preview above is how the app functions overall. 


## Conclusion:
The model provides a solid foundation for multiclass image classification tasks. While the current accuracy is decent, there is ample room for further improvement using techniques such as transfer learning (using VGG16, InceptionV3 etc), hyperparameter optimization, and regularization. With future enhancements, this project can achieve even better classification results and generalize well to unseen data.
