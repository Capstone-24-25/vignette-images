library(keras)
library(tensorflow)

batch_size <- 32
img_height <- 256
img_width <- 256
#increases resolution, helps model generalize

# Define class names
class_names <- c('notumor', 'pituitary', 'meningioma', 'glioma')

# Normalizes pixel values to [0,1]
train_gen <- image_data_generator(rescale = 1/255, validation_split = 0.2)  # Include validation split
test_gen <- image_data_generator(rescale = 1/255)

#validation splits helps with over fitting, noticed the model before the split had a high accuracy 
# but low validation accuracy which told us the model was overfitting.

# Train dataset (80% of the training data)
train_dataset <- flow_images_from_directory( #loads and processes images from directory
  directory = "data/Training",
  generator = train_gen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  class_mode = "categorical",
  subset = "training",  # Specify training subset
  seed = 1111
)

# Validation dataset (20% of the training data)
validation_dataset <- flow_images_from_directory(
  directory = "data/Training",
  generator = train_gen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  class_mode = "categorical",
  subset = "validation",  # Specify validation subset, loads 20% for validation
  seed = 1111
)

# Test dataset, no splits
test_dataset <- flow_images_from_directory(
  directory = "data/Testing",
  generator = test_gen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  class_mode = "categorical"
)

# Define the CNN model, multi-class, vgg16 architecture - good for feature extraction
# Dropout helps with overfitting, dropping out 50% of the output units randomly from
# applied layer during training process
cnn_model <- keras_model_sequential() %>%
  # Block 1
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(img_height, img_width, 3)) %>%
  layer_batch_normalization() %>%  # Batch normalization
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Block 2
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_batch_normalization() %>%  # Batch normalization
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Block 3
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_batch_normalization() %>%  # Batch normalization
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Block 4
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_batch_normalization() %>%  # Batch normalization
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Global Average Pooling instead of Flatten
  layer_global_average_pooling_2d() %>%
  
  # Dense Layers
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(0.5) %>%
  layer_dense(units = length(class_names), activation = 'softmax')  # Output layer
# layer_conv_2d extracts spatial features, filters - # of feature detectors, 
# kernel_size - size of the filter matrix, relu - non-linearity
# layer_max_pooling_2d - reduces spatial dimensions to prevent overfitting
# layer_flatten - flattens 2D feature maps in 1D vector for input into dense layers
# layer_dense - dense layers for classification
# layer_dropout - randomly drops 50% of nodes during training to prevent overfitting
# units = # of classes, softmax - generates probabilites for each class



# Compile the model
cnn_model %>% compile(
  optimizer = optimizer_adam(), #adaptive optimization algorithim
  loss = "categorical_crossentropy", #Loss function for multi-class classification
  metrics = c("accuracy")
)

# Train the model
history <- cnn_model %>% fit(
  train_dataset,
  validation_data = validation_dataset,
  epochs = 10,
  steps_per_epoch = train_dataset$samples %/% batch_size, ## of batches per epoch to process
  validation_steps = validation_dataset$samples %/% batch_size
)

# Save the model
cnn_model %>% save_model_hdf5("brain_tumor_model.h5")

# Evaluate the model on the test dataset
evaluation <- cnn_model %>% evaluate(
  test_dataset,
  steps = test_dataset$samples %/% batch_size
)
cat("Test accuracy:", evaluation["accuracy"], "\n")