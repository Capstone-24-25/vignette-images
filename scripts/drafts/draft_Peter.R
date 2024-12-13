library(imager)
library(magick)
library(keras)
library(tensorflow)

### Data preparation

load_images_from_folder <- function(folder_path, label) {
  # Get all image file paths
  image_files <- list.files(folder_path, pattern = "\\.jpg$", full.names = TRUE)
  
  # Load and preprocess each image
  images <- lapply(image_files, function(img_path) {
    img <- load.image(img_path)               # Load the image
    img_resized <- resize(img, 128, 128)      # Resize to 128x128
    img_array <- as.array(img_resized) / 255  # Normalize pixel values
    list(image = img_array, label = label)    # Return image and label
  })
  
  return(images)
}

train_dir <- "data/Training/"
test_dir <- "data/Testing/"

train_tumor <- load_images_from_folder(paste0(train_dir, "tumor/"), 1)
train_notumor <- load_images_from_folder(paste0(train_dir, "notumor/"), 0)

# Combine training data
train_data <- c(train_tumor, train_notumor)

# Load testing data
test_tumor <- load_images_from_folder(paste0(test_dir, "tumor/"), 1)
test_notumor <- load_images_from_folder(paste0(test_dir, "notumor/"), 0)

# Combine testing data
test_data <- c(test_tumor, test_notumor)

prepare_data <- function(data) {
  images <- array(unlist(lapply(data, function(x) x$image)), 
                  dim = c(length(data), 128, 128, 1))  # For grayscale
  labels <- as.array(unlist(lapply(data, function(x) x$label)))
  return(list(images = images, labels = labels))
}

# Prepare training and testing data
train <- prepare_data(train_data)
test <- prepare_data(test_data)



### Model Construction

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", 
                input_shape = c(128, 128, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Train the model
model %>% fit(
  x = train$images, 
  y = train$labels, 
  epochs = 10, 
  batch_size = 32, 
  validation_split = 0.2
)



### Model Evaluation
model %>% evaluate(test$images, test$labels)

predictions <- model %>% predict(test$images)
predicted_labels <- ifelse(predictions > 0.5, 1, 0)
