library(caret)  # For confusion matrix visualization
library(tensorflow)

load('data/Testing')

set.seed(2222)
# Step 1: Predict probabilities for the test dataset
predictions <- cnn_model %>% predict(test_dataset, steps = test_dataset$samples %/% batch_size)

# Step 2: Convert probabilities to predicted class indices
predicted_classes <- apply(predictions, 1, which.max)

# Step 3: Extract true labels
# Re-run the test dataset to obtain true labels
true_labels <- unlist(lapply(1:(test_dataset$samples %/% batch_size), function(i) {
  batch <- generator_next(test_dataset)
  apply(batch[[2]], 1, which.max)  # Extract true class indices from one-hot encoding
}))

# Step 4: Create confusion matrix
confusion <- confusionMatrix(
  factor(predicted_classes, levels = 1:length(class_names), labels = class_names),
  factor(true_labels, levels = 1:length(class_names), labels = class_names)
)

# Print confusion matrix
print(confusion)
