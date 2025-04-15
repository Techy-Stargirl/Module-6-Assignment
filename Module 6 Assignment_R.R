#Set a different CRAN mirror (to address potential download issues)
options(repos = c(CRAN = "https://cloud.r-project.org/"))

#Install the keras R package
install.packages("keras")

#Load the keras library
library(keras)

#Load the reticulate library (for Python integration)
library(reticulate)

#Specify the path to your Anaconda Python executable
use_python("C:/users/AMLIT/anaconda3/python.exe")


#Install the necessary Python backend (TensorFlow) and other dependencies
install_keras(tensorflow = "2.19")

# Try importing keras.datasets and loading the data
tryCatch({
  keras_datasets <- import("keras.datasets")
  fashion_mnist <- keras_datasets$fashion_mnist$load_data()
  print("Fashion MNIST dataset loaded successfully via Python import!")
}, error = function(e) {
  print(paste("Error loading Fashion MNIST via Python import:", e$message))
})

# Load dataset
fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# Preprocess
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1)) / 255
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1)) / 255
y_train_cat <- to_categorical(y_train, 10)
y_test_cat <- to_categorical(y_test, 10)

# Define model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit
history <- model %>% fit(x_train, y_train_cat, epochs = 5, validation_data = list(x_test, y_test_cat))

# Predict on 2 images
preds <- model %>% predict(x_test[1:2,,,])
predicted_classes <- apply(preds, 1, which.max) - 1

# Show predictions
cat(paste("Predicted:", predicted_classes[1], ", Actual:", y_test[1]), "\n")
cat(paste("Predicted:", predicted_classes[2], ", Actual:", y_test[2]), "\n")


