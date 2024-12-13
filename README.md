# vignette-images

Vignette on implementing a convolutional neural network using a brain tumor image data set for the purpose of multi-class image classification; created as a class project for PSTAT197A in Fall 2024.

# Contributors

David Pan, James San, Peter Xiong, Amy Ji

# Abstract

Image Classification is a type of supervised machine learning, model is trained using labeled data and known outputs, whose goal is to be able to recognize and classify images into their respective class based on features gathered from said images. For our vignette we focus on multi-class classification using a convolutional neural network, a model type designed for spatial data such as images, so using a model to classify images into one of multiple categories.

Our example data involves the case of brain tumors, there are many different types of brain tumors with over 120 being categorized by location and cell formation. Thus, diagnosing brain tumors is often challenging and requires extensive imaging tests and scans which often leads to delays in diagnosis. Using machine learning can combat this issue, specifically image classification to classify MRI scans of brain tumors into diagnosis categories.

Overall, our model achieved

# Repository Contents

In our directory can be found multiple folders, the data folder contains the data contents which are found in the subdirectories, Training and Testing. The scripts folder contains the code contents of the model, data visualizations, and testing and training which can be found in the drafts subdirectory.

- `data` contains

  - `Training` contains four types of brain tumor images we use for training
  
  - `Testing` contains images we use for testing
  
- `Scripts` contains

  - `drafts` contains drafts we create
  
  - `main_work` contains the code of our project
  
  - `preprocessing` contains the code of preprocessing part
  
  - `resnet` contains the code of resnet model
  
- `vignette.qmd` a markdown file contain our main work

- `vignette.html` the html rendered by the qmd file

- `README.md` the introduction of the project and references


# References

- Image Classification

  - [What is Image Classification?](https://www.geeksforgeeks.org/what-is-image-classification/)
  
- CNN architecture

  - [Basic CNN Architecture: Explaining 5 Layers of Convolutional Neural Network](https://www.upgrad.com/blog/basic-cnn-architecture/)

  - [VGG-16 | CNN model](https://www.geeksforgeeks.org/vgg-16-cnn-model/)
  
- Dataset

  - [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
  
