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
test_gen <- image_data_generator(rescale = 1/255, validation_split = 0.2)

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

# Load the pre-trained ResNet50 model
base_model <- application_resnet50(
  weights = "imagenet",        # Use pre-trained weights on ImageNet
  include_top = FALSE,         # Exclude the fully connected layers
  input_shape = c(img_height, img_width, 3)  # Match your input shape
)

# Freeze the base model to retain pre-trained features
freeze_weights(base_model)

# Add custom layers for classification
resnet_model <- keras_model_sequential() %>%
  base_model %>%
  layer_global_average_pooling_2d() %>%   # Reduce dimensionality
  layer_dense(units = 128, activation = 'relu') %>%  # Add a dense layer
  layer_dropout(rate = 0.5) %>%           # Dropout for regularization
  layer_dense(units = length(class_names), activation = 'softmax')  # Output layer for classification

unfreeze_weights(base_model, from = "conv5_block1_out")
# Compile the model
resnet_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-4),  # Lower learning rate for fine-tuning
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# Train the model
history <- resnet_model %>% fit(
  train_dataset,
  validation_data = validation_dataset,
  epochs = 10,
  steps_per_epoch = train_dataset$samples %/% batch_size,
  validation_steps = validation_dataset$samples %/% batch_size
)

# Evaluate on the test dataset
evaluation <- resnet_model %>% evaluate(
  test_dataset,
  steps = test_dataset$samples %/% batch_size
)
cat("Test accuracy:", evaluation["accuracy"], "\n")
