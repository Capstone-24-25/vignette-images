library(tidyverse)
library(tensorflow)
library(keras3)

tensorflow::set_random_seed(197)

# the data has already been partitioned
batch_size <- 32
img_height <- 256
img_width <- 256

train_tumor <- image_dataset_from_directory(
  directory = 'data/Training',
  image_size = c(img_height, img_width),
  batch_size = batch_size,
  seed = 197
)

test_tumor <- image_dataset_from_directory(
  directory = 'data/Testing',
  image_size = c(img_height, img_width),
  batch_size = batch_size,
  seed = 197
)




library(keras)

residual_block <- function(input, filters, strides = c(1, 1)) {
  # Shortcut connection (input bypass)
  shortcut <- input
  
  # If input and output shapes are different, apply projection
  if (input$shape[[3]] != filters || !all(strides == c(1, 1))) {
    shortcut <- layer_conv_2d(
      filters = filters,
      kernel_size = c(1, 1),
      strides = strides,
      padding = "same",
      activation = NULL
    )(input)
    shortcut <- layer_batch_normalization()(shortcut)
  }
  
  # First convolution
  x <- layer_conv_2d(
    filters = filters,
    kernel_size = c(3, 3),
    strides = strides,
    padding = "same",
    activation = NULL
  )(input)
  x <- layer_batch_normalization()(x)
  x <- layer_activation_relu()(x)
  
  # Second convolution
  x <- layer_conv_2d(
    filters = filters,
    kernel_size = c(3, 3),
    strides = c(1, 1),
    padding = "same",
    activation = NULL
  )(x)
  x <- layer_batch_normalization()(x)
  
  # Add shortcut and output
  x <- layer_add(list(shortcut, x))
  x <- layer_activation_relu()(x)
  
  return(x)
}

# Define the ResNet Model
input_layer <- layer_input(shape = c(256, 256, 3))

# Initial Convolution and Pooling
x <- layer_conv_2d(filters = 64, kernel_size = c(7, 7), strides = c(2, 2), padding = "same")(input_layer)
x <- layer_batch_normalization()(x)
x <- layer_activation_relu()(x)
x <- layer_max_pooling_2d(pool_size = c(3, 3), strides = c(2, 2), padding = "same")(x)

# Residual Blocks
x <- residual_block(x, filters = 64)
x <- residual_block(x, filters = 64)
x <- residual_block(x, filters = 128, strides = c(2, 2))  # Downsample
x <- residual_block(x, filters = 128)
x <- residual_block(x, filters = 256, strides = c(2, 2))  # Downsample
x <- residual_block(x, filters = 256)


# Fully Connected Layers
x <- layer_global_average_pooling_2d()(x)
output_layer <- layer_dense(units = 4, activation = "softmax")(x)

# Compile the Model
resnet_model <- keras_model(inputs = input_layer, outputs = output_layer)

# Print Model Summary
summary(resnet_model)

# Compile the model
resnet_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",  # For integer labels
  metrics = c("accuracy")
)

# Train the model
history <- resnet_model %>% fit(
  train_tumor,  # Replace with your training dataset
  validation_data = test_tumor,  # Replace with your validation dataset
  epochs = 10
)

evaluate(resnet_model, test_tumor)