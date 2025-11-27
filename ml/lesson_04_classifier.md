# Lesson 4: Building and Training a Simple Classifier

**Duration:** 5-6 hours

**Prerequisites:** Lessons 1-3

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Build a convolutional neural network (CNN) from scratch
2. Train a classifier on MNIST digit dataset
3. Achieve >95% accuracy through proper training techniques
4. Use TensorBoard for visualization and debugging
5. Implement data augmentation and normalization
6. Understand evaluation metrics (accuracy, precision, recall, F1)
7. Debug common training issues

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

### Convolutional Neural Networks (CNNs)

CNNs are specialized for processing **gridlike data** such as images.
Instead of looking at all pixels at once with a fully connected layer,
they use small filters that scan across the image and look for local
patterns.

The core idea of a **convolution** is to take a small **filter (or
kernel)**, for example a 33 grid of weights, and slide it over the
image. At each position, you multiply the filter weights by the pixel
values underneath, sum them up, and write the result into an output
image. Because the **same filter is reused everywhere**, the network can
detect the same pattern (like a vertical edge) no matter where it
appears in the input. By stacking several convolutional layers, the
network first learns simple edges, then combinations of edges, and
eventually more complex shapes.

The main building blocks of a CNN work together to turn raw pixels into
useful features. **Convolutional layers** are responsible for learning
spatial hierarchies: early layers tend to respond to lowlevel patterns
like edges and textures, middle layers respond to shapes or simple
parts, and deeper layers respond to highlevel object parts (for MNIST,
things like loops and strokes that define digits).

Between convolutions, we often insert **pooling layers** that reduce the
spatial resolution of the feature maps. For example, a 22 maxpooling
layer replaces each 22 block with its maximum value. This makes the
representation smaller and more robust to small shifts in the input: if
an edge moves by one pixel, it still activates roughly the same pooled
unit. Average pooling does something similar but takes the mean instead
of the maximum, producing a smoother representation.

After several rounds of convolution and pooling, we typically **flatten**
the feature maps and feed them into one or more **fully connected
layers**. These final layers treat the extracted features as a vector and
perform the last steps of reasoning needed for **classification**: for
MNIST, mapping from highlevel digit features to ten output scores (one
for each digit 07).

CNNs matter for RL because many environments provide **visual
observations**. In Atari, the state is literally the game screen; in
Rocket League you may use camera images or rendered views of the field.
CNNs act as **feature extractors** that convert raw game frames into
compact, informative state vectors. These vectors then feed into your RL
algorithm, allowing the agent to reason about **spatial structure** such
as object positions, layouts, and motion.

### Classification Metrics

When you train a classifier, you need ways to judge how good its
predictions are. The most familiar metric is **accuracy**, defined as
`(number of correct predictions) / (total predictions)`. Accuracy is easy
to understand, but it can be misleading when the classes are imbalanced
(for example, 99% of examples are class 0): a model that always predicts
the majority class can have high accuracy but be useless.

A **confusion matrix** gives a more detailed picture. It is a table that
shows, for each true class, how many examples were predicted as each
possible class. For digit classification you get a 1010 table: row 3,
column 8 tells you how many 3 digits were incorrectly labeled as
8. Looking at this matrix can reveal systematic mistakes (for
example, confusing 3 and 5).

Two closely related metrics are **precision** and **recall**. Precision
answers: Of all the examples the model predicted as positive, how many
were actually positive? Recall answers: Of all the truly positive
examples, how many did the model find? In other words, precision cares
about **false positives**, while recall cares about **false negatives**.
The **F1 score** is the harmonic mean of precision and recall; it is
high only if both are reasonably high, so it is a good single number to
summarize performance when you care about both kinds of error.

### Batch Normalization

During training, the distribution of activations in a layer can change as
earlier layers update their weights. This phenomenon (sometimes called
*internal covariate shift*) can slow down or destabilize learning.
**Batch normalization** tackles this by normalizing each layers
activations within a minibatch so that they have roughly zero mean and
unit variance, and then learning a separate scale and shift for each
feature so the layer can still represent any distribution it needs.

Normalizing activations in this way often leads to **faster training**
and allows you to use **higher learning rates** without divergence. It
also tends to **regularize** the model slightly, because the statistics
are computed from small batches and add a bit of noise to the activations.

### Data Augmentation

Instead of collecting more real images, we can synthetically
**augment** the dataset by applying random transformations so the model
sees slightly different versions of each example on every epoch. For
image data this might mean random **rotations**, **translations** (small
shifts), **scaling**, or flips.

By creating many such variations of the training images, you encourage
the model to focus on the underlying digit shape rather than the exact
pixel arrangement. This makes the classifier more **robust** to natural
variations (for example, a digit drawn slightly offcenter) and helps
**prevent overfitting**, because the effective size and diversity of the
training set is increased without collecting new labeled data.

## 💻 Practical Implementation

### Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
```

### 1. Load and Explore MNIST Dataset

```python
# Download and load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Visualize some examples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png')
print("MNIST samples saved")

# Check data distribution
labels = [label for _, label in train_dataset]
unique, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(10, 4))
plt.bar(unique, counts)
plt.xlabel('Digit')
plt.ylabel('Count')
plt.title('MNIST Training Set Distribution')
plt.savefig('mnist_distribution.png')
print("Distribution plot saved")
```

### 2. Build CNN Architecture

```python
class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 14x14 -> 14x14

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14

        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

model = SimpleCNN().to(device)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### 3. Improved CNN with Batch Normalization

```python
class ImprovedCNN(nn.Module):
    """CNN with batch normalization and better architecture."""
    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # FC layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 28 -> 14

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 14 -> 7

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 7 -> 3

        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

model = ImprovedCNN().to(device)
print(f"\nImproved CNN parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 4. Training Loop with Metrics

```python
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Setup training
model = ImprovedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# TensorBoard
writer = SummaryWriter('runs/mnist_experiment')

# Training loop
num_epochs = 20
best_acc = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # TensorBoard logging
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_mnist_model.pth')
        print(f"✓ Saved new best model (acc: {best_acc:.2f}%)")

writer.close()
print(f"\n✅ Training complete! Best accuracy: {best_acc:.2f}%")
print("Run 'tensorboard --logdir=runs' to view training curves")
```

### 5. Detailed Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load best model
model.load_state_dict(torch.load('best_mnist_model.pth'))
model.eval()

# Collect all predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved")

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                          target_names=[str(i) for i in range(10)]))

# Per-class accuracy
class_correct = [0] * 10
class_total = [0] * 10

for pred, label in zip(all_preds, all_labels):
    class_correct[label] += (pred == label)
    class_total[label] += 1

print("\nPer-class Accuracy:")
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"Digit {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
```

### 6. Visualize Predictions

```python
def visualize_predictions(model, dataset, device, num_images=10):
    """Visualize model predictions."""
    model.eval()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    indices = np.random.choice(len(dataset), num_images, replace=False)

    with torch.no_grad():
        for idx, ax in zip(indices, axes.flat):
            image, true_label = dataset[idx]
            image_input = image.unsqueeze(0).to(device)

            output = model(image_input)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            # Display
            ax.imshow(image.squeeze(), cmap='gray')
            color = 'green' if predicted.item() == true_label else 'red'
            ax.set_title(f'True: {true_label}, Pred: {predicted.item()}\n'
                        f'Confidence: {confidence.item():.2%}',
                        color=color)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    print("Predictions visualization saved")

visualize_predictions(model, test_dataset, device)
```

### 7. Activation Visualization

```python
def visualize_conv_filters(model):
    """Visualize first layer conv filters."""
    # Get first conv layer weights
    first_conv = model.conv1
    weights = first_conv.weight.data.cpu()

    # weights shape: (out_channels, in_channels, kernel_h, kernel_w)
    # For MNIST: (32, 1, 3, 3)

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            ax.imshow(weights[i, 0], cmap='viridis')
            ax.set_title(f'Filter {i}')
        ax.axis('off')

    plt.suptitle('First Convolutional Layer Filters')
    plt.tight_layout()
    plt.savefig('conv_filters.png')
    print("Conv filters visualization saved")

visualize_conv_filters(model)

def visualize_feature_maps(model, image, device):
    """Visualize intermediate feature maps."""
    model.eval()
    image = image.unsqueeze(0).to(device)

    # Hook to capture activations
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))
    model.conv3.register_forward_hook(hook_fn('conv3'))

    # Forward pass
    with torch.no_grad():
        model(image)

    # Visualize
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))

    for layer_idx, (layer_name, activation) in enumerate(activations.items()):
        activation = activation.cpu().squeeze()

        for i in range(8):
            ax = axes[layer_idx, i]
            if i < activation.shape[0]:
                ax.imshow(activation[i], cmap='viridis')
                ax.set_title(f'{layer_name}[{i}]')
            ax.axis('off')

    plt.suptitle('Feature Maps Visualization')
    plt.tight_layout()
    plt.savefig('feature_maps.png')
    print("Feature maps visualization saved")

# Test with a random image
test_image, _ = test_dataset[0]
visualize_feature_maps(model, test_image, device)
```

### 8. Data Augmentation

```python
# Define augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create augmented dataset
augmented_dataset = datasets.MNIST('./data', train=True, transform=train_transform)
augmented_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)

# Visualize augmented samples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    img, label = augmented_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.suptitle('Augmented MNIST Samples')
plt.tight_layout()
plt.savefig('augmented_samples.png')
print("Augmented samples saved")

# Train with augmentation
print("\nTraining with data augmentation...")
augmented_model = ImprovedCNN().to(device)
optimizer_aug = optim.Adam(augmented_model.parameters(), lr=0.001)

for epoch in range(10):
    train_loss, train_acc = train_epoch(augmented_model, augmented_loader,
                                       criterion, optimizer_aug, device)
    val_loss, val_acc = validate(augmented_model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
```

## 📚 Key References

### Papers
- LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition" (LeNet)
- Krizhevsky et al. (2012) - "ImageNet Classification with Deep CNNs" (AlexNet)
- Ioffe & Szegedy (2015) - "Batch Normalization" - [arXiv](https://arxiv.org/abs/1502.03167)

### Tutorials
- [PyTorch MNIST Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [Understanding CNNs](https://poloclub.github.io/cnn-explainer/)

### Tools
- [TensorBoard Documentation](https://pytorch.org/docs/stable/tensorboard.html)
- [Netron - Model Visualizer](https://netron.app/)

## 🏋️ Exercises

### Exercise 1: Architecture Comparison (Medium)

Compare three architectures on MNIST:
1. Simple MLP (no convolutions)
2. SimpleCNN
3. ImprovedCNN

Report: accuracy, training time, parameter count. Which is best?

### Exercise 2: Transfer Learning (Medium-Hard)

Load a pretrained ResNet18 and fine-tune it for MNIST:

```python
from torchvision.models import resnet18

# Your code: adapt ResNet18 for MNIST
# Hint: Modify first conv layer (3 channels → 1 channel)
# Hint: Modify final FC layer (1000 classes → 10 classes)
```

### Exercise 3: Misclassification Analysis (Medium)

Find the 20 most confidently wrong predictions. Visualize them. Are there patterns? Which digits are confused most?

### Exercise 4: Adversarial Examples (Hard)

Implement FGSM (Fast Gradient Sign Method) to create adversarial examples:

```python
def fgsm_attack(model, image, label, epsilon):
    """
    Create adversarial example using FGSM.

    Args:
        model: trained model
        image: input image
        label: true label
        epsilon: perturbation magnitude

    Returns:
        adversarial image
    """
    # Your code here
    pass
```

### Exercise 5: Custom Dataset (Hard)

Create your own digit dataset:
1. Draw 100 digits (10 of each)
2. Preprocess to 28x28 grayscale
3. Test your trained model on them
4. Analyze errors - why do they occur?

## 🔧 Troubleshooting Tips

### Common Issues

1. **Low accuracy (<90%)**
   - Check data normalization
   - Verify model architecture
   - Try different learning rates
   - Train for more epochs

2. **Overfitting**
   - Add dropout
   - Use data augmentation
   - Reduce model complexity
   - Add L2 regularization

3. **Slow training**
   - Increase batch size (if memory allows)
   - Use GPU
   - Reduce model size
   - Use fewer epochs

4. **TensorBoard not showing**
   ```bash
   # Make sure to run:
   tensorboard --logdir=runs
   # Then open http://localhost:6006
   ```

5. **Out of memory**
   - Reduce batch size
   - Reduce model size
   - Clear CUDA cache: `torch.cuda.empty_cache()`

## ✅ Self-Check

Before moving to Lesson 5, you should be able to:

- [ ] Build a CNN from scratch
- [ ] Train a classifier achieving >95% accuracy
- [ ] Use TensorBoard for visualization
- [ ] Implement data augmentation
- [ ] Evaluate with confusion matrix and classification report
- [ ] Visualize learned features
- [ ] Debug training issues
- [ ] Understand batch normalization

## 🚀 Next Steps

Congratulations! You've mastered supervised learning with neural networks. Now you're ready to dive into reinforcement learning!

**Next:** [Lesson 5: Reinforcement Learning Theory](lesson_05_rl_theory.md)
- Markov Decision Processes (MDPs)
- Value functions and Bellman equations
- The RL problem formulation
- Policy vs value-based methods

This is where the real RL journey begins!

---

**Estimated completion time:** 5-6 hours (including exercises)

**Next lesson:** [Lesson 5: RL Theory (MDPs, Bellman) →](lesson_05_rl_theory.md)
