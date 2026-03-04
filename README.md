# DeepLearning Project 2

This project is about designing, training, and analyzing a convolutional neural network (CNN) for image classification. The work is completed in [Project2.ipynb](c:\Users\sia\Desktop\DL\src\DeepLearning_Project2\Project2.ipynb), where the full implementation, plots, evaluation results, and written answers are included.

## What This Project Covers

The notebook focuses on the core ideas behind CNNs and compares them with a standard deep neural network (DNN). It includes:

- conceptual CNN questions such as weight sharing, stride, and pooling,
- implementation of a CNN with two convolution layers and pooling,
- manual computation of layer output shapes,
- model training and evaluation,
- confusion matrix and per-class accuracy analysis,
- feature map visualization from the first convolution layer,
- comparison between a CNN and a DNN on the same dataset,
- short written reflections on the results.

## Dataset Used

The project uses the handwritten digits dataset from `sklearn.datasets.load_digits`.

Why this dataset was used:

- it is available locally,
- it avoids external downloads,
- it is suitable for quick CNN experiments,
- it contains 10 classes (`0` to `9`) of grayscale digit images.

Each image is `8 x 8`, which makes the task small enough to train quickly while still supporting CNN analysis.

## Models Implemented

### CNN

The CNN includes:

- `Conv2d(1, 16, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(2, 2)`
- `Conv2d(16, 32, kernel_size=3, padding=1)`
- `ReLU`
- `MaxPool2d(2, 2)`
- fully connected classifier with dropout

Manual shape flow:

- input: `1 x 8 x 8`
- after conv1: `16 x 8 x 8`
- after pool1: `16 x 4 x 4`
- after conv2: `32 x 4 x 4`
- after pool2: `32 x 2 x 2`
- flattened: `128`

### DNN Baseline

For comparison, a fully connected neural network was also trained on the same train/test split. This provides a baseline to compare CNN behavior against a non-convolutional architecture.

## What We Did

The notebook was completed from start to finish:

1. Answered all conceptual Markdown questions.
2. Implemented dataset loading, preprocessing, normalization, and light augmentation.
3. Built the CNN model and documented the output shapes.
4. Wrote the full training and evaluation loop in PyTorch.
5. Plotted loss and accuracy curves.
6. Computed the confusion matrix and per-class accuracy.
7. Identified the hardest class and explained why it was harder.
8. Visualized feature maps from the first convolution layer.
9. Trained a DNN on the same dataset for comparison.
10. Added the comparison table and final reflection.

## Results Summary

From the validated run in the `mlhub` environment:

- CNN accuracy: `94.72%`
- DNN accuracy: `96.11%`
- CNN parameters: `13,706`
- DNN parameters: `17,226`

The hardest class for the CNN was digit `8`, with a per-class accuracy of `80.00%`. This makes sense because digit `8` can look similar to several other digits when represented in a very small `8 x 8` image.

An interesting outcome of this project is that the DNN slightly outperformed the CNN. That is reasonable here because the dataset is tiny, simple, and already centered, so the DNN can still perform very well. On larger and more varied image datasets, CNNs usually become more advantageous.

## How To Run

Use the `mlhub` Jupyter kernel for the notebook.

Open and run:

- [Project2.ipynb](c:\Users\sia\Desktop\DL\src\DeepLearning_Project2\Project2.ipynb)

The notebook contains all code, plots, and written explanations needed for the project submission.

## Project Goal

The main goal of this project is to understand how CNNs work in practice, not just how to code them. It combines implementation, mathematical reasoning, model evaluation, and interpretation of learned features.
