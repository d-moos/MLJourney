# Lesson 3: Supervised Learning Fundamentals

**Duration:** 4-5 hours

**Prerequisites:** Lessons 1-2 (NumPy and PyTorch)

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Understand the supervised learning paradigm and how it differs from RL
2. Know the key components of a training pipeline
3. Master different loss functions and when to use them
4. Understand optimization algorithms (SGD, Momentum, Adam)
5. Recognize and address overfitting and underfitting
6. Properly split data into train/validation/test sets
7. See how these concepts apply to RL (value function approximation)

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

### The Supervised Learning Paradigm

**Core idea:** Learn a function f: X → Y from labeled examples (x, y)

```
Input (x) → Model (f_θ) → Prediction (ŷ) → Compare to Truth (y) → Update θ
```

**Key differences from RL:**

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|-------------------|----------------------|
| **Feedback** | Direct labels (y) | Delayed rewards (r) |
| **Data** | i.i.d. samples | Sequential, correlated states |
| **Goal** | Minimize prediction error | Maximize cumulative reward |
| **Exploration** | Not needed | Critical for learning |

**Why learn supervised learning for RL?**
- Value functions V(s) and Q(s,a) are learned via supervised regression
- Policy networks can use supervised loss (behavior cloning)
- Many RL algorithms reduce to supervised learning sub-problems

### Components of Supervised Learning

#### 1. The Model f_θ

A parameterized function (e.g., neural network) that makes predictions:

```python
ŷ = f_θ(x)
```

**Parameters θ:** Weights and biases that we optimize

#### 2. The Loss Function L

Measures how wrong predictions are:

```python
L(ŷ, y) = loss(prediction, true_value)
```

**Common loss functions:**

**Regression (continuous outputs):**
- **Mean Squared Error (MSE):** L = (ŷ - y)²
  - Sensitive to outliers
  - Used for: Value function approximation in RL

- **Mean Absolute Error (MAE):** L = |ŷ - y|
  - More robust to outliers
  - Used for: Robust value estimation

- **Huber Loss:** Combines MSE (small errors) + MAE (large errors)
  - Used for: DQN (more stable than MSE)

**Classification (discrete outputs):**
- **Cross-Entropy:** L = -Σ y_i log(ŷ_i)
  - For probability distributions
  - Used for: Policy networks in RL

#### 3. The Optimizer

Algorithm for updating parameters to minimize loss:

```python
θ_new = θ_old - learning_rate * ∇_θ L
```

**Common optimizers:**
- **SGD:** θ = θ - α∇L
- **Momentum:** Adds velocity term for faster convergence
- **Adam:** Adaptive learning rates per parameter (most popular)

#### 4. Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch.x)
        loss = loss_function(predictions, batch.y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Overfitting and Underfitting

**Underfitting:** Model too simple, high training error
- Solution: Bigger model, more features, train longer

**Overfitting:** Model memorizes training data, poor generalization
- Solutions:
  - More training data
  - Regularization (L2, dropout)
  - Early stopping
  - Data augmentation

**Bias-Variance Tradeoff:**
- **High bias:** Underfitting
- **High variance:** Overfitting
- **Sweet spot:** Good generalization

### Train/Validation/Test Split

- **Training set (70-80%):** Used to update model parameters
- **Validation set (10-15%):** Used to tune hyperparameters, early stopping
- **Test set (10-15%):** Final evaluation, never used during training

**Critical:** Test set must remain unseen until final evaluation!

### Connection to Reinforcement Learning

Many RL algorithms use supervised learning internally:

**Value-based methods (DQN):**
```python
# Supervised regression problem:
# Input: state s
# Target: r + γ max_a' Q(s', a')
# Prediction: Q(s, a)
loss = MSE(Q(s,a), r + γ * max_a' Q(s',a'))
```

**Policy-based methods (Behavior Cloning):**
```python
# Supervised classification problem:
# Input: state s
# Target: expert action a*
# Prediction: π(a|s)
loss = CrossEntropy(π(·|s), a*)
```

**Actor-Critic methods:**
- Actor: Policy network (supervised by policy gradient)
- Critic: Value network (supervised by TD error)

## 💻 Practical Implementation

### Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(42)
np.random.seed(42)
```

### 1. Loss Functions

```python
# Regression losses
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.2, 2.1, 2.8])

# Mean Squared Error
mse_loss = nn.MSELoss()
mse = mse_loss(y_pred, y_true)
print(f"MSE Loss: {mse.item():.4f}")

# Mean Absolute Error
mae_loss = nn.L1Loss()
mae = mae_loss(y_pred, y_true)
print(f"MAE Loss: {mae.item():.4f}")

# Huber Loss (smooth L1)
huber_loss = nn.SmoothL1Loss()
huber = huber_loss(y_pred, y_true)
print(f"Huber Loss: {huber.item():.4f}")

# Classification losses
logits = torch.tensor([[2.0, 1.0, 0.1],   # Batch of 2
                       [0.5, 2.5, 1.0]])
targets = torch.tensor([0, 1])  # Class indices

# Cross-Entropy (combines LogSoftmax + NLLLoss)
ce_loss = nn.CrossEntropyLoss()
ce = ce_loss(logits, targets)
print(f"\nCross-Entropy Loss: {ce.item():.4f}")

# Binary Cross-Entropy (for binary classification)
probs = torch.tensor([0.9, 0.2, 0.8])  # Predicted probabilities
targets = torch.tensor([1.0, 0.0, 1.0])  # True labels

bce_loss = nn.BCELoss()
bce = bce_loss(probs, targets)
print(f"Binary Cross-Entropy: {bce.item():.4f}")

# Visualize loss functions
errors = torch.linspace(-3, 3, 100)

mse_values = errors ** 2
mae_values = torch.abs(errors)
huber_values = torch.where(torch.abs(errors) < 1,
                           0.5 * errors ** 2,
                           torch.abs(errors) - 0.5)

plt.figure(figsize=(10, 4))
plt.plot(errors.numpy(), mse_values.numpy(), label='MSE')
plt.plot(errors.numpy(), mae_values.numpy(), label='MAE')
plt.plot(errors.numpy(), huber_values.numpy(), label='Huber')
plt.xlabel('Error (ŷ - y)')
plt.ylabel('Loss')
plt.title('Comparison of Loss Functions')
plt.legend()
plt.grid(True)
plt.savefig('loss_functions.png')
print("\nLoss functions plot saved")
```

### 2. Optimizers Comparison

```python
# Create simple problem: learn y = 3x + 2
X = torch.randn(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.1

def train_with_optimizer(optimizer_class, lr, name, epochs=100):
    """Train a model with specified optimizer."""
    model = nn.Linear(1, 1)
    optimizer = optimizer_class(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses, model

# Compare optimizers
optimizers = [
    (optim.SGD, 0.01, 'SGD'),
    (optim.SGD, 0.01, 'SGD + Momentum', {'momentum': 0.9}),
    (optim.Adam, 0.01, 'Adam'),
]

plt.figure(figsize=(12, 4))

for i, (opt_class, lr, name, *kwargs) in enumerate(optimizers):
    extra_kwargs = kwargs[0] if kwargs else {}

    if extra_kwargs:
        losses, model = train_with_optimizer(
            lambda params, lr: opt_class(params, lr, **extra_kwargs),
            lr, name
        )
    else:
        losses, model = train_with_optimizer(opt_class, lr, name)

    plt.plot(losses, label=name)

    # Print final learned parameters
    w = model.weight.item()
    b = model.bias.item()
    print(f"{name:20s} - Learned: y = {w:.2f}x + {b:.2f}")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.savefig('optimizer_comparison.png')
print("\nOptimizer comparison plot saved")
```

### 3. Complete Training Pipeline

```python
# Create synthetic dataset
class SyntheticDataset(Dataset):
    """Simple dataset: y = sin(x) + noise"""
    def __init__(self, num_samples=1000):
        self.X = torch.randn(num_samples, 1) * 5
        self.y = torch.sin(self.X) + torch.randn(num_samples, 1) * 0.1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset
dataset = SyntheticDataset(1000)

# Split into train/val/test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model
class SimpleRegressor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion):
    """Validate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)

# Train with early stopping
num_epochs = 200
patience = 10
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Load best model and evaluate on test set
model.load_state_dict(torch.load('best_model.pth'))
test_loss = validate(model, test_loader, criterion)
print(f"\nFinal Test Loss: {test_loss:.4f}")

# Plot training curves
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_curves.png')
print("Training curves saved")
```

### 4. Overfitting Demonstration

```python
# Create small dataset (easier to overfit)
small_dataset = SyntheticDataset(100)
train_small, val_small = random_split(small_dataset, [80, 20])

train_loader_small = DataLoader(train_small, batch_size=16, shuffle=True)
val_loader_small = DataLoader(val_small, batch_size=16, shuffle=False)

# Train models of different complexities
def train_model_complexity(hidden_dim, name):
    """Train model and return losses."""
    model = SimpleRegressor(hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(300):
        train_loss = train_epoch(model, train_loader_small, criterion, optimizer)
        val_loss = validate(model, val_loader_small, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses

# Compare different model sizes
complexities = [
    (8, 'Small (8)'),
    (64, 'Medium (64)'),
    (256, 'Large (256)'),
]

plt.figure(figsize=(15, 4))

for i, (hidden_dim, name) in enumerate(complexities):
    print(f"Training {name}...")
    train_losses, val_losses = train_model_complexity(hidden_dim, name)

    plt.subplot(1, 3, i+1)
    plt.plot(train_losses, label='Train', alpha=0.7)
    plt.plot(val_losses, label='Validation', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{name} Model')
    plt.legend()
    plt.grid(True)

    # Check for overfitting
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    gap = final_val - final_train

    if gap > 0.1:
        plt.text(150, max(val_losses) * 0.8, 'OVERFITTING',
                color='red', fontsize=12, ha='center')

plt.tight_layout()
plt.savefig('overfitting_demo.png')
print("Overfitting demonstration saved")
```

### 5. Regularization Techniques

```python
# L2 Regularization (Weight Decay)
model_l2 = SimpleRegressor(hidden_dim=256)
optimizer_l2 = optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=1e-4)

# Dropout
class RegularizedRegressor(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model_dropout = RegularizedRegressor(hidden_dim=256, dropout=0.3)

print("Regularization techniques:")
print("1. L2 regularization (weight_decay in optimizer)")
print("2. Dropout (randomly zeros neurons during training)")
print("3. Early stopping (stop when validation loss increases)")
```

### 6. Batch Size Effects

```python
def train_with_batch_size(batch_size, epochs=50):
    """Train model with specific batch size."""
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = SimpleRegressor(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        loss = train_epoch(model, loader, criterion, optimizer)
        losses.append(loss)

    return losses

# Compare batch sizes
batch_sizes = [8, 32, 128]
plt.figure(figsize=(10, 4))

for bs in batch_sizes:
    print(f"Training with batch size {bs}...")
    losses = train_with_batch_size(bs)
    plt.plot(losses, label=f'Batch size {bs}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Effect of Batch Size on Training')
plt.legend()
plt.grid(True)
plt.savefig('batch_size_effects.png')
print("Batch size comparison saved")
```

### 7. Learning Rate Scheduling

```python
model = SimpleRegressor()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

train_losses = []
learning_rates = []

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, LR={learning_rates[-1]:.6f}")

# Plot learning rate schedule
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.savefig('lr_scheduling.png')
print("Learning rate scheduling plot saved")
```

## 📚 Key References

### Books
- **Ian Goodfellow et al.** - "Deep Learning" - [Free online](https://www.deeplearningbook.org/)
  - Chapter 5: Machine Learning Basics
  - Chapter 6: Deep Feedforward Networks
  - Chapter 8: Optimization

### Online Courses
- **Andrew Ng** - "Machine Learning Course" - [Coursera](https://www.coursera.org/learn/machine-learning)
- **Fast.ai** - "Practical Deep Learning" - [Free](https://course.fast.ai/)

### Papers
- Ruder, S. (2016) - "An overview of gradient descent optimization algorithms" - [arXiv](https://arxiv.org/abs/1609.04747)
- Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization" - [arXiv](https://arxiv.org/abs/1412.6980)

### Tutorials
- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [CS231n: Training Neural Networks](http://cs231n.github.io/neural-networks-3/)

## 🏋️ Exercises

### Exercise 1: Loss Function Analysis (Easy)

Plot and compare how MSE, MAE, and Huber loss behave for different error magnitudes. Answer:
1. Which loss is most sensitive to outliers?
2. When would you use Huber loss over MSE?
3. How does this relate to RL (DQN uses Huber loss)?

### Exercise 2: Implement Momentum SGD (Medium)

Implement SGD with momentum from scratch (without using PyTorch's optimizer):

```python
class MomentumSGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        """
        Implement momentum SGD.

        Update rule:
        v_t = momentum * v_{t-1} + lr * gradient
        param = param - v_t
        """
        # Your code here
        pass

    def zero_grad(self):
        # Your code here
        pass

    def step(self):
        # Your code here
        pass

# Test it on a simple problem and compare to torch.optim.SGD
```

### Exercise 3: Regularization Comparison (Medium-Hard)

Train models with:
1. No regularization
2. L2 regularization (weight_decay=1e-4)
3. Dropout (p=0.3)
4. Both L2 + Dropout

Use a small dataset (100 samples) and large model (hidden_dim=256). Plot training and validation curves for all four. Which prevents overfitting best?

```python
# Your code here
# Create comparison plots showing all four approaches
```

### Exercise 4: Learning Rate Finder (Medium-Hard)

Implement the "LR Range Test" to find optimal learning rate:

1. Start with very small LR (1e-7)
2. Gradually increase LR exponentially each batch
3. Plot loss vs learning rate
4. Optimal LR is usually where loss decreases fastest

```python
def find_learning_rate(model, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
    """
    Implement learning rate range test.

    Returns:
        lrs: list of learning rates tried
        losses: list of losses at each LR
    """
    # Your code here
    pass

# Use it to find best LR for your model
```

### Exercise 5: Bellman Error as Supervised Learning (Conceptual + Coding)

**Part A:** Explain how Q-learning can be viewed as supervised learning. What is the "input", "target", and "prediction"?

**Part B:** Implement a simple tabular Q-learning update as a supervised regression problem:

```python
# Given:
# - Current Q-value estimate: Q(s, a) = 5.0
# - Observed reward: r = 10.0
# - Next state max Q-value: max_a' Q(s', a') = 8.0
# - Discount factor: γ = 0.9

# Treat this as supervised learning:
# Input: (s, a)
# Target: r + γ * max_a' Q(s', a')
# Prediction: Q(s, a)
# Loss: MSE

# Compute the gradient and update using gradient descent (lr=0.1)
# Show the step-by-step calculation

# Your code here
```

## 🔧 Troubleshooting Tips

### Common Issues

1. **Loss not decreasing**
   - Learning rate too high or too low
   - Wrong loss function
   - Bug in model forward pass
   - Data not normalized

2. **Loss becomes NaN**
   - Learning rate too high (exploding gradients)
   - Division by zero
   - Log of negative number
   - Solution: Lower LR, add gradient clipping

3. **Validation loss much higher than training loss**
   - Overfitting
   - Solution: Regularization, more data, simpler model

4. **Training loss much higher than validation loss**
   - Using dropout/batch norm (disabled in eval mode)
   - This is actually normal!

5. **Model predicts same value for all inputs**
   - Dead neurons (ReLU)
   - Learning rate too high
   - Poor initialization

### Best Practices

```python
# 1. Always normalize inputs
X_mean = X_train.mean(dim=0, keepdim=True)
X_std = X_train.std(dim=0, keepdim=True)
X_normalized = (X - X_mean) / (X_std + 1e-8)

# 2. Use proper weight initialization (PyTorch does this automatically)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)

# 3. Monitor multiple metrics
metrics = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
}

# 4. Use gradient clipping for stability
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 5. Save checkpoints regularly
if epoch % 10 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_epoch_{epoch}.pth')
```

## ✅ Self-Check

Before moving to Lesson 4, you should be able to:

- [ ] Explain the supervised learning paradigm
- [ ] Choose appropriate loss functions for regression/classification
- [ ] Understand different optimizers and their tradeoffs
- [ ] Implement a complete training pipeline with train/val/test splits
- [ ] Recognize and address overfitting
- [ ] Use regularization techniques (L2, dropout, early stopping)
- [ ] Tune hyperparameters (LR, batch size, architecture)
- [ ] Connect supervised learning concepts to RL

## 🚀 Next Steps

You're now ready for [Lesson 4: Building and Training a Simple Classifier](lesson_04_classifier.md), where you'll:
- Build a CNN for MNIST digit classification
- Implement data augmentation
- Use TensorBoard for visualization
- Achieve >95% accuracy
- Learn debugging techniques

This will solidify your supervised learning skills before diving into reinforcement learning!

---

**Estimated completion time:** 4-5 hours (including exercises)

**Next lesson:** [Lesson 4: Building a Simple Classifier →](lesson_04_classifier.md)
