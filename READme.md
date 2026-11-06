# Fashion-MNIST Image Classification: FNN vs. CNN

## 1. Project Overview

This project explores image classification using neural networks and PyTorch.

The main objective was to build, train, and evaluate two different neural network architectures:

1. **Fully Connected Neural Network (FNN)** — A baseline model that processes images as flattened pixel values.
2. **Convolutional Neural Network (CNN)** — A model designed specifically for image data that learns spatial patterns using convolutional filters.

The goal was to understand how these architectures work, compare their performance, and investigate why CNNs are particularly effective for computer vision tasks.

## 2. Dataset

The project uses the **Fashion-MNIST dataset**, which contains 70,000 grayscale images of clothing items.

Each image has a resolution of 28 × 28 pixels.

### Dataset Distribution

| Dataset    | Number of Images |
| ---------- | ---------------: |
| Training   |           54,000 |
| Validation |            6,000 |
| Testing    |           10,000 |
| **Total**  |       **70,000** |

The original training dataset of 60,000 images was divided into training and validation sets. The test set remained separate for final evaluation.

### Classification Classes

The dataset contains 10 clothing categories:

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

### Data Preprocessing

- Loaded the dataset using `torchvision.datasets.FashionMNIST`.
- Converted images into PyTorch tensors using `transforms.ToTensor()`.
- Normalized pixel values from the original 0–255 range to 0–1.
- Divided the training data into training and validation sets.
- Used PyTorch `DataLoader` to organize images into batches of 64.

## 3. Model 1: Fully Connected Neural Network (FNN)

The first model served as our baseline.

Since fully connected layers expect one-dimensional input features, each 28 × 28 image was flattened into a vector containing 784 pixel values.

### Architecture

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

### Architecture Breakdown

| Layer              | Purpose                              |
| ------------------ | ------------------------------------ |
| Flatten            | Converts the image into 784 features |
| Linear (784 → 128) | First hidden layer                   |
| ReLU               | Introduces nonlinearity              |
| Linear (128 → 64)  | Second hidden layer                  |
| ReLU               | Introduces nonlinearity              |
| Linear (64 → 10)   | Produces scores for 10 classes       |

### Training

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 64
- Training epochs: 11

The model was trained using forward propagation, loss calculation, backpropagation, and gradient-based weight updates.

## 4. Model 2: Convolutional Neural Network (CNN)

The second model was designed specifically for image classification.

Unlike the FNN, which flattens the image immediately, the CNN preserves its spatial structure and uses convolutional filters to learn visual patterns.

### Architecture

```python
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### Architecture Breakdown

| Layer               | Purpose                             |
| ------------------- | ----------------------------------- |
| Conv2d (1 → 32)     | Learns 32 initial visual filters    |
| ReLU                | Introduces nonlinearity             |
| MaxPool2d           | Reduces spatial dimensions          |
| Conv2d (32 → 64)    | Learns more complex visual patterns |
| ReLU                | Introduces nonlinearity             |
| MaxPool2d           | Further reduces spatial dimensions  |
| Flatten             | Converts feature maps into a vector |
| Linear (1600 → 128) | Hidden classification layer         |
| ReLU                | Introduces nonlinearity             |
| Linear (128 → 10)   | Produces scores for 10 classes      |

### Training

The CNN used the same training and validation datasets as the FNN.

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 64
- Training epochs: 10

Both models used the same training procedure, allowing their performance to be compared fairly.

## 5. Model Evaluation

Both models were evaluated using the same unseen test dataset containing 10,000 images.

### Performance Comparison

| Metric                   |    FNN |    CNN |
| ------------------------ | -----: | -----: |
| Best Validation Accuracy | 88.83% | 91.43% |
| Test Accuracy            | 88.32% | 90.76% |
| Test Loss                | 0.3293 | 0.2703 |
| Correct Predictions      |  8,832 |  9,076 |
| Incorrect Predictions    |  1,168 |    924 |

### Key Findings

The CNN achieved:

- 2.44 percentage points higher test accuracy.
- 244 more correct predictions.
- Lower test loss than the FNN.

This demonstrates the advantage of preserving spatial information and learning visual patterns through convolutional layers.

## 6. Confusion Matrix Analysis

Confusion matrices were generated for both models to examine their classification performance across individual clothing categories.

### FNN Observations

- Performed particularly well on Bags, Trousers, Sneakers, and Sandals.
- Struggled most with Shirts, correctly classifying 735 out of 1,000.
- Frequently confused Shirts with T-shirts, Pullovers, and Coats.
- Showed confusion between visually similar clothing categories.

### CNN Observations

- Correctly classified 984 Bags, 985 Sandals, and 976 Trousers.
- Improved classification of several categories compared with the FNN.
- Still struggled with Shirts, correctly classifying 705 out of 1,000.
- Continued to confuse Shirts with T-shirts and other similar clothing categories.

### Important Observation

Although the CNN achieved better overall accuracy, it did not outperform the FNN on every individual class.

For example, Shirt classification decreased from 735 correct predictions with the FNN to 705 with the CNN.

This highlights why evaluating a model using accuracy alone is insufficient. Confusion matrices help identify specific weaknesses that overall accuracy can hide.

## 7. Technologies Used

- **Python** — Programming language.
- **PyTorch** — Neural network construction and training.
- **Torchvision** — Dataset loading and image transformations.
- **NumPy** — Numerical operations.
- **Matplotlib** — Image visualization and performance plots.
- **Scikit-learn** — Confusion matrix evaluation.
- **Jupyter Notebook** — Interactive development and experimentation.

## 8. Key Concepts Learned

Through this project, I explored and implemented:

- Tensor dimensions, shapes, and data types.
- Image preprocessing and normalization.
- Training, validation, and testing datasets.
- DataLoader and batch processing.
- Neural network architecture design.
- Fully connected layers and activation functions.
- Forward propagation.
- Loss functions and CrossEntropyLoss.
- Backpropagation and gradient descent.
- Optimizers and weight updates.
- Convolutional layers and filters.
- Feature maps and spatial feature extraction.
- Max pooling and dimensionality reduction.
- Model evaluation and confusion matrix interpretation.
- Comparing different neural network architectures.

## 9. Conclusion

This project demonstrated the difference between fully connected and convolutional neural networks when working with image data.

The FNN provided a strong baseline, achieving 88.32% test accuracy. However, the CNN achieved 90.76%, demonstrating the benefit of convolutional operations for learning spatial patterns in images.

An important takeaway was that better overall accuracy does not necessarily mean better performance across every class. Examining confusion matrices revealed that certain visually similar clothing categories remained difficult to distinguish.

Most importantly, this project provided practical experience in designing, training, evaluating, and comparing neural networks using PyTorch.
