# DeepLearning Project 2

This project focuses on designing, training, and analyzing a convolutional neural network for image classification on CIFAR-10. The full work is contained in [Project2.ipynb](/c:/Users/sia/Desktop/DL/src/DeepLearning_Project2/Project2.ipynb), including the implementation, plots, evaluation results, comparison with a DNN baseline, and written answers for each project section.

## Project Scope

The notebook covers:

- conceptual CNN questions such as weight sharing, stride, and pooling,
- implementation of a CNN for CIFAR-10 classification,
- manual output-shape calculations through the convolution and pooling stack,
- model training and evaluation in PyTorch,
- loss and accuracy curves for both CNN and DNN training,
- confusion matrices and per-class accuracy analysis,
- feature map visualization from the first CNN layer,
- CNN versus DNN comparison on the same train/test split,
- final reflection on which architectural choice mattered most.

## Dataset

The project uses the local **CIFAR-10** dataset stored in the [`data`](/c:/Users/sia/Desktop/DL/src/DeepLearning_Project2/data) folder.

- training data comes from `data_batch_1` to `data_batch_5`
- test data comes from `test_batch`
- each image has shape `3 x 32 x 32`
- there are 10 classes:
  `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

Preprocessing in the notebook includes:

- scaling pixel values to `[0, 1]`,
- computing channel-wise mean and standard deviation from the training split,
- normalization using those training statistics,
- random crop and random horizontal flip for training augmentation.

## Models

### CNN

The CNN uses:

- `Conv2d(3, 32, kernel_size=3, padding=1)`
- batch normalization and `ReLU`
- `MaxPool2d(2, 2)`
- `Conv2d(32, 64, kernel_size=3, padding=1)`
- batch normalization and `ReLU`
- `MaxPool2d(2, 2)`
- `Conv2d(64, 128, kernel_size=3, padding=1)`
- batch normalization and `ReLU`
- `MaxPool2d(2, 2)`
- fully connected classifier:
  `Linear(2048, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 10)`

Manual CNN shape flow:

- input: `3 x 32 x 32`
- after conv1: `32 x 32 x 32`
- after pool1: `32 x 16 x 16`
- after conv2: `64 x 16 x 16`
- after pool2: `64 x 8 x 8`
- after conv3: `128 x 8 x 8`
- after pool3: `128 x 4 x 4`
- flatten: `2048`
- output: `10`

CNN trainable parameters: `620,810`

### DNN Baseline

The DNN baseline flattens the image immediately and uses fully connected layers:

- `Linear(3 * 32 * 32, 512)`
- `ReLU`
- `Dropout(0.3)`
- `Linear(512, 256)`
- `ReLU`
- `Dropout(0.3)`
- `Linear(256, 10)`

DNN trainable parameters: `1,707,274`

## Training Setup

Both models are trained in the notebook with:

- PyTorch
- Adam optimizer
- cross-entropy loss
- `5` epochs
- CPU-friendly settings so the notebook remains runnable in the provided environment

The shared training helper tracks:

- training loss per epoch,
- test loss per epoch,
- training accuracy per epoch,
- test accuracy per epoch,
- total training time.

## Notebook Outputs

The notebook produces:

- CNN training loss and accuracy curves,
- CNN confusion matrix,
- CNN per-class accuracy and hardest class,
- DNN training loss and accuracy curves,
- DNN confusion matrix,
- DNN per-class accuracy and hardest class,
- first-layer CNN feature map visualizations,
- comparison table for CNN versus DNN.

## Results Summary

From the run documented in the notebook:

| Model | Accuracy | Parameters | Training Time |
|---|---:|---:|---:|
| CNN | 74.09% | 620,810 | 663.37 s |
| DNN | 40.87% | 1,707,274 | 236.63 s |

Additional reported observations:

- hardest CNN class: `cat` with per-class accuracy `50.10%`
- the CNN clearly outperformed the DNN despite using fewer parameters

The main reason is that CIFAR-10 is an image dataset with strong spatial structure. The CNN keeps that structure through convolution and pooling, while the DNN flattens the image at the start and loses the locality bias that helps on natural-image tasks.

## How To Run

Use the `mlhub` Jupyter kernel and open:

- [Project2.ipynb](/c:/Users/sia/Desktop/DL/src/DeepLearning_Project2/Project2.ipynb)

Run the notebook from top to bottom to reproduce:

- data loading and preprocessing,
- CNN and DNN training,
- plotted training curves,
- confusion matrices,
- feature map visualizations,
- final comparison outputs.

## Project Goal

The goal of this project is to understand CNNs beyond implementation alone by combining model design, manual reasoning about tensor shapes, training behavior analysis, feature interpretation, and comparison against a non-convolutional baseline.
