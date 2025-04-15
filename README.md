# Python Code for Fashion MNIST Image Classification with Convolutional Neural Networks (CNN)

## Overview

This Python script demonstrates how to build and train a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset consists of 70,000 grayscale images of 10 fashion categories, with 60,000 training images and 10,000 test images.

This script performs the following steps:

1.  **Installs TensorFlow:** Ensures the necessary deep learning library is installed.
2.  **Imports Libraries:** Imports essential libraries such as NumPy for numerical operations, Matplotlib for plotting, and TensorFlow/Keras for building and training the neural network.
3.  **Loads the Dataset:** Loads the Fashion MNIST dataset directly from Keras.
4.  **Preprocesses the Data:**
    * Reshapes the image data into the format expected by the CNN (adding a channel dimension for grayscale images).
    * Scales the pixel values to the range of 0 to 1.
    * One-hot encodes the categorical labels.
5.  **Builds the CNN Model:** Creates a sequential CNN model with the following layers:
    * An input layer specifying the shape of the input images (28x28 grayscale).
    * Two convolutional layers (`Conv2D`) with ReLU activation to extract features.
    * Two max-pooling layers (`MaxPooling2D`) to reduce spatial dimensions.
    * A third convolutional layer for further feature extraction.
    * A flattening layer (`Flatten`) to convert the 2D feature maps into a 1D vector.
    * A dense (fully connected) layer with ReLU activation.
    * A dropout layer (`Dropout`) to prevent overfitting.
    * An output dense layer with softmax activation to produce probability scores for each of the 10 classes.
6.  **Compiles the Model:** Configures the model for training by specifying the optimizer (`adam`), loss function (`categorical_crossentropy`), and evaluation metric (`accuracy`).
7.  **Trains the Model:** Trains the CNN model using the training data and evaluates its performance on the test data during training.
8.  **Predicts on Test Images:** Uses the trained model to make predictions on the first two images from the test set.
9.  **Displays Predictions:** Visualizes the two test images along with their predicted and actual labels.

## Prerequisites

* **Python 3.x** installed on your system.
* **pip** (Python package installer) installed.

## Installation

1.  **Install TensorFlow:**
    Open your terminal or command prompt and run the following command:
    ```bash
    pip install tensorflow
    ```
2.  **Install Matplotlib (if not already installed):**
    ```bash
    pip install matplotlib
    ```
3.  **Save the Code:**
    Save the provided Python code as a `.py` file (e.g., `fashion_mnist_cnn.py`).

## Usage

1.  **Navigate to the Directory:**
    Open your terminal or command prompt and navigate to the directory where you saved the `fashion_mnist_cnn.py` file.

2.  **Run the Script:**
    Execute the script using the Python interpreter:
    ```bash
    python fashion_mnist_cnn.py
    ```

    The script will:
    * Download and load the Fashion MNIST dataset.
    * Preprocess the data.
    * Build and train the CNN model.
    * Print training and validation accuracy and loss for each epoch.
    * Make predictions on the first two test images.
    * Display the two test images with their predicted and actual labels in separate Matplotlib windows.

## Code Explanation

```python
!pip install tensorflow - This line is a shell command executed within the Python environment to install the TensorFlow library if it's not already present.

# Import necessary libraries - Import necessary libraries for numerical computation (numpy), plotting (matplotlib.pyplot), dataset loading and model building (tensorflow.keras).
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load dataset - fashion_mnist.load_data() retrieves the training and testing sets of the Fashion MNIST dataset as NumPy arrays.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
- Reshaping: The image data is reshaped from a 2D array (28x28) to a 4D array (number of samples, height, width, number of channels). For grayscale images, the number of channels is 1.
- Normalization: Pixel values are divided by 255 to scale them to the range [0, 1], which helps in faster and more stable training.
- One-Hot Encoding: The categorical labels (integers representing the fashion categories) are converted into a binary matrix format using to_categorical. This is required for the categorical_crossentropy loss function.

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build CNN model
- Sequential: Creates a linear stack of layers.
- Input(shape=(28, 28, 1)): Defines the input layer with the expected shape of each image.
- Conv2D(filters, kernel_size, activation): Convolutional layers extract features from the input images. filters specifies the number of output filters, kernel_size defines the size of the convolutional window, and activation='relu' applies the Rectified Linear Unit activation function.
- MaxPooling2D(pool_size): Max pooling layers reduce the spatial dimensions of the feature maps, making the model more robust to small translations and distortions.   
- Flatten(): Flattens the 2D feature maps into a 1D vector, which can then be fed into fully connected layers.
- Dense(units, activation): Dense (fully connected) layers learn complex relationships between the features. units specifies the number of neurons in the layer. The final dense layer has 10 units (one for each fashion category) and uses the softmax activation function to output probability distributions over the classes.
- Dropout(rate): Dropout layers randomly set a fraction (rate) of input units to 0 during training, which helps to prevent overfitting.

model = Sequential([
    Input(shape=(28, 28, 1)),  # Explicitly define the input shape
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model
- optimizer='adam': Specifies the optimization algorithm to be used for training (Adam is a popular and efficient optimizer).
- loss='categorical_crossentropy': Defines the loss function to be minimized during training (suitable for multi-class classification with one-hot encoded labels).
- metrics=['accuracy']: Specifies the metric to be evaluated during training and testing.

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model - model.fit(x_train, y_train_cat, epochs=5, validation_data=(x_test, y_test_cat)): Trains the model on the training data for a specified number of epochs. The validation_data argument allows monitoring the model's performance on the test set during training.

model.fit(x_train, y_train_cat, epochs=5, validation_data=(x_test, y_test_cat))

# Predict on two test images
- model.predict(x_test[:2]): Uses the trained model to predict the probabilities for each class for the first two images in the test set.
- np.argmax(predictions, axis=1): Converts the probability distributions into class labels by selecting the class with the highest probability.

predictions = model.predict(x_test[:2])
predicted_classes = np.argmax(predictions, axis=1)

# Display predictions - Iterates through the first two test images, displays them using matplotlib.pyplot.imshow, and prints the predicted and actual labels as the title of each plot.

for i in range(2):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {predicted_classes[i]}, Actual: {y_test[i]}")
    plt.show()

Stellamaris Okeh
Stellamarisijeoma0@gmail.com
