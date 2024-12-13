install.packages("EBImage")  # For advanced image processing

library(EBImage)
library(fs)

# Function to preprocess and resize images
preprocess_image <- function(img_path, img_size = 256) {
  # Load the image
  img <- readImage(img_path)
  
  # Convert to grayscale
  gray_img <- channel(img, "gray")
  
  # Apply Gaussian blur
  blurred_img <- gblur(gray_img, sigma = 3)
  
  # Threshold the image
  thresh <- blurred_img > mean(blurred_img)
  
  # Erode and dilate to remove noise
  thresh_clean <- dilate(erode(thresh, makeBrush(5, shape = "disc")), makeBrush(5, shape = "disc"))
  
  # Identify the bounding box of the largest object (contour)
  labels <- bwlabel(thresh_clean)
  features <- computeFeatures.shape(labels)
  
  # Check if any features are detected
  if (nrow(features) == 0) {
    warning("No features detected. Returning the original image resized.")
    return(resize(img, img_size, img_size))
  }
  
  largest_object <- which.max(features[, "s.area"])
  bbox <- features[largest_object, c("s.cx", "s.cy", "s.width", "s.height")]
  
  # Define cropping coordinates
  x_center <- bbox["s.cx"]
  y_center <- bbox["s.cy"]
  width <- bbox["s.width"]
  height <- bbox["s.height"]
  xmin <- max(0, x_center - width / 2)
  ymin <- max(0, y_center - height / 2)
  xmax <- min(dim(img)[2], x_center + width / 2)
  ymax <- min(dim(img)[1], y_center + height / 2)
  
  # Crop and resize
  cropped_img <- crop(img, xmin, ymin, xmax, ymax)
  resized_img <- resize(cropped_img, img_size, img_size)
  
  return(resized_img)
}

# Batch processing function
resize_images_in_directory <- function(input_dir, output_dir, img_size = 256) {
  # Ensure output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # List all main subdirectories (e.g., Testing and Training)
  main_subdirs <- dir_ls(input_dir, type = "directory")
  
  for (main_subdir in main_subdirs) {
    main_subdir_name <- path_file(main_subdir)  # Get the name (e.g., Testing, Training)
    main_output_path <- file.path(output_dir, main_subdir_name)
    
    # Ensure output main subdirectory exists
    if (!dir.exists(main_output_path)) {
      dir.create(main_output_path, recursive = TRUE)
    }
    
    # List all subfolders (e.g., glioma, meningioma, etc.)
    subdirs <- dir_ls(main_subdir, type = "directory")
    
    for (subdir in subdirs) {
      subdir_name <- path_file(subdir)
      subdir_output_path <- file.path(main_output_path, subdir_name)
      
      # Create output subdirectory
      if (!dir.exists(subdir_output_path)) {
        dir.create(subdir_output_path, recursive = TRUE)
      }
      
      # List all image files in the subdirectory
      image_files <- dir_ls(subdir, glob = "*.jpg")  # Adjust for your image format
      
      # Process each image
      for (image_path in image_files) {
        # Use preprocess_image to process and resize the image
        resized_img <- preprocess_image(image_path, img_size)
        
        # Save the resized image
        output_path <- file.path(subdir_output_path, path_file(image_path))
        writeImage(resized_img, output_path)
      }
    }
  }
}



#Example when Using
input_directory <- "path/to/your/data"   # Replace with data directory path
output_directory <- "path/to/cleaned"   # Replace with desired output directory

resize_images_in_directory(input_directory, output_directory)

