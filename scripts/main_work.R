# packages
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

# Single Layer Model

model_single_layer <- keras_model_sequential(input_shape = c(img_height, img_width, 3)) %>%
  layer_rescaling(1./255) %>%
  layer_flatten() %>%
  layer_dense(4) %>%
  layer_activation(activation = 'softmax')

summary(model_single_layer)

model_single_layer %>% compile(
  optimizer = 'adam',
  loss = 'crossentropy',
  metrics = 'accuracy'
)

history_single_layer <- model_single_layer %>%
  fit(train_tumor, epochs = 20)

evaluate(model_single_layer, test_tumor)

# Data Augmentation

data_augmentation <- keras_model_sequential(input_shape = c(256, 256, 3)) %>%
  layer_random_brightness(factor = 0.1) %>%
  layer_random_contrast(factor = 0.15)

par(mfrow = c(3,3), mar = c(1,1,1,1))
for (i in 1:9){
  images[1,,,, drop = FALSE] %>%
    data_augmentation() %>%
    display_image()
}

model_aug <- keras_model_sequential(input_shape = c(img_height, img_width, 3)) %>%
  layer_random_brightness(factor = 0.1) %>%
  layer_random_contrast(factor = 0.15) %>%
  layer_rescaling(1./255) %>%
  layer_flatten() %>%
  layer_dense(4) %>%
  layer_activation(activation = 'softmax')

summary(model_aug)

model_aug %>% compile(
  optimizer = 'adam',
  loss = 'crossentropy',
  metrics = 'accuracy'
)

history_aug <- model_aug %>%
  fit(train_tumor, epochs = 20)

evaluate(model_aug, test_tumor)

# Multi Layer Model

model_vgg16 <- keras_model_sequential(input_shape = c(256, 256, 3)) %>%
  # Preprocessing layers
  layer_random_brightness(factor = 0.1) %>%
  layer_random_contrast(factor = 0.15) %>%
  layer_rescaling(1./255) %>%
  
  # Convolutional layers
  layer_conv_2d(filters = 8, kernel_size = c(3,3), activation = 'relu', padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2)) %>%
  
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu', padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2)) %>%
  
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2)) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu', padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2)) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu', padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2,2), strides = c(2,2)) %>%
  
  # Fully connected layers
  layer_flatten() %>%
  layer_dense(units = 4096, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model_vgg16)

model_vgg16 %>% compile(loss = 'crossentropy',
                        optimizer = 'adam',
                        metrics = 'accuracy')

history_vgg16 <- model_vgg16 %>%
  fit(train_tumor, epochs = 10)

evaluate(model_vgg16, test_tumor)

