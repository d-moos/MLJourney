# Lesson 2: Introduction to PyTorch

**Duration:** 4-6 hours

**Prerequisites:** Lesson 1 (NumPy fundamentals)

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Understand PyTorch tensors and their relationship to NumPy arrays
2. Perform tensor operations and utilize GPU acceleration
3. Grasp automatic differentiation with autograd
4. Build simple neural networks using `torch.nn`
5. Understand the computation graph and backpropagation
6. Know when to use PyTorch vs NumPy in your RL implementations

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

### What is PyTorch?

PyTorch is a deep learning framework that provides:

1. **Tensors:** Like NumPy arrays but with GPU support and autodiff
2. **Autograd:** Automatic differentiation for backpropagation
3. **Neural Network Building Blocks:** Pre-built layers, optimizers, loss functions
4. **Dynamic Computation Graphs:** Build and modify networks on-the-fly

### Why PyTorch for RL?

- **Flexibility:** Easy to implement custom RL algorithms
- **Speed:** GPU acceleration for large neural networks
- **Gradient Computation:** Automatic backprop for policy/value networks
- **Community:** Extensive RL libraries (Stable-Baselines3, TorchRL, etc.)

### Key Concepts

#### 1. Tensors vs NumPy Arrays

```python
import numpy as np
import torch

# NumPy array
np_array = np.array([1, 2, 3])

# PyTorch tensor
torch_tensor = torch.tensor([1, 2, 3])

# Easy conversion
tensor_from_numpy = torch.from_numpy(np_array)
numpy_from_tensor = torch_tensor.numpy()
```

**Key differences:**
- Tensors can run on GPU
- Tensors track gradients for backpropagation
- Tensors integrate seamlessly with PyTorch's neural network modules

#### 2. Automatic Differentiation (Autograd)

PyTorch automatically computes gradients using the chain rule:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # Compute dy/dx

print(x.grad)  # Gradient: dy/dx = 2x = 4
```

**Why this matters for RL:**
- Policy gradient methods need ∇_θ log π(a|s)
- Value function approximation needs ∇_w V(s)
- Actor-critic methods need both!

#### 3. Computation Graphs

PyTorch builds a directed acyclic graph (DAG) of operations:

```
Input (x) → Operation (x²) → Output (y)
   ↑                            ↓
   └────────── Gradient ─────────┘
```

Each operation knows how to compute its local gradient, and PyTorch chains them together.

#### 4. Neural Network Modules

```python
import torch.nn as nn

# Define a simple network
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
```

This structure will be the foundation for Q-networks, policy networks, and value networks in RL.

## 💻 Practical Implementation

### Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Check PyTorch version and CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

### 1. Creating Tensors

```python
# From Python lists
tensor1 = torch.tensor([1, 2, 3, 4, 5])
print(f"1D Tensor: {tensor1}")
print(f"Shape: {tensor1.shape}, Dtype: {tensor1.dtype}")

# 2D tensor (matrix)
tensor2 = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
print(f"\n2D Tensor:\n{tensor2}")

# Common creation methods
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
eye = torch.eye(4)
arange = torch.arange(0, 10, 2)
linspace = torch.linspace(0, 1, 5)

# Random tensors
rand_uniform = torch.rand(3, 3)          # Uniform [0, 1)
rand_normal = torch.randn(3, 3)          # Standard normal
rand_int = torch.randint(0, 10, (3, 3))  # Random integers

# Specify dtype and device
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
if torch.cuda.is_available():
    cuda_tensor = torch.tensor([1, 2, 3], device='cuda')

print(f"\nFloat tensor: {float_tensor}, dtype: {float_tensor.dtype}")

# From NumPy (shares memory!)
np_array = np.array([1, 2, 3, 4, 5])
tensor_from_np = torch.from_numpy(np_array)
print(f"\nFrom NumPy: {tensor_from_np}")

# Copy vs share memory
np_array[0] = 99
print(f"After modifying NumPy array: {tensor_from_np}")  # Also changed!

# To copy instead
tensor_copy = torch.tensor(np_array)  # Creates a copy
```

### 2. Tensor Operations

```python
# Element-wise operations
a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
b = torch.tensor([10, 20, 30, 40], dtype=torch.float32)

print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a ** 2 = {a ** 2}")

# Mathematical functions
angles = torch.tensor([0, torch.pi/2, torch.pi])
print(f"\nsin(angles) = {torch.sin(angles)}")
print(f"exp([1,2,3]) = {torch.exp(torch.tensor([1.0, 2.0, 3.0]))}")

# Aggregation functions
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]], dtype=torch.float32)

print(f"\nSum: {torch.sum(tensor)}")
print(f"Sum along axis 0: {torch.sum(tensor, dim=0)}")
print(f"Sum along axis 1: {torch.sum(tensor, dim=1)}")
print(f"Mean: {torch.mean(tensor)}")
print(f"Max: {torch.max(tensor)}")
print(f"Argmax: {torch.argmax(tensor)}")

# Matrix operations
A = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6],
                  [7, 8]], dtype=torch.float32)

# Matrix multiplication
C = torch.mm(A, B)      # Method 1
D = A @ B                # Method 2 (preferred)

print(f"\nMatrix multiplication:\n{C}")

# Transpose
print(f"\nTranspose of A:\n{A.T}")

# RL Example: Batch Q-value computation
batch_size, state_dim, num_actions = 32, 4, 2
states = torch.randn(batch_size, state_dim)
weights = torch.randn(state_dim, num_actions)
q_values = states @ weights

print(f"\nQ-values shape: {q_values.shape}")
print(f"Q-values for first state: {q_values[0]}")
```

### 3. Automatic Differentiation

```python
# Basic gradient computation
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

print(f"x = {x}")
print(f"y = x² = {y}")

# Compute gradient
y.backward()
print(f"dy/dx = {x.grad}")  # Should be 2x = 4

# More complex example
x = torch.tensor([3.0], requires_grad=True)
y = 2 * x ** 3 + 3 * x ** 2 + 5

y.backward()
print(f"\nFor y = 2x³ + 3x² + 5 at x=3:")
print(f"dy/dx = {x.grad}")  # Should be 6x² + 6x = 72

# Gradient accumulation (important!)
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()
print(f"\nFirst backward: {x.grad}")

y = x ** 2
y.backward()  # Adds to existing gradient!
print(f"Second backward (accumulated): {x.grad}")

# Clear gradients
x.grad.zero_()
y = x ** 2
y.backward()
print(f"After zero_(): {x.grad}")

# Vector gradients
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = torch.sum(y)  # Need scalar for backward()

z.backward()
print(f"\nVector gradient dz/dx = {x.grad}")  # Should be 2x for each element

# RL Example: Simple policy gradient
# Suppose we have log probabilities and returns
log_probs = torch.tensor([0.5, -0.2, 0.1], requires_grad=True)
returns = torch.tensor([10.0, -5.0, 20.0])

# Policy gradient loss: -sum(log_prob * return)
loss = -torch.sum(log_probs * returns)
loss.backward()

print(f"\nPolicy gradient:")
print(f"∇ loss = {log_probs.grad}")
```

### 4. Building Neural Networks

```python
# Method 1: Using nn.Sequential (simple)
simple_net = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

print("Simple Network:")
print(simple_net)

# Test it
dummy_input = torch.randn(1, 10)
output = simple_net(dummy_input)
print(f"\nOutput shape: {output.shape}")

# Method 2: Using nn.Module (flexible, preferred for RL)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# Create and test
state_dim = 4
hidden_dim = 64
value_net = ValueNetwork(state_dim, hidden_dim)

print("\n\nValue Network:")
print(value_net)

# Forward pass
state = torch.randn(1, state_dim)
value = value_net(state)
print(f"\nEstimated value: {value.item():.3f}")

# Check parameters
total_params = sum(p.numel() for p in value_net.parameters())
print(f"Total parameters: {total_params}")

# Access specific parameters
print(f"\nFirst layer weights shape: {value_net.fc1.weight.shape}")
print(f"First layer bias shape: {value_net.fc1.bias.shape}")
```

### 5. Q-Network Example (for RL)

```python
class QNetwork(nn.Module):
    """
    Deep Q-Network for estimating action values.
    Input: state
    Output: Q-value for each action
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # No activation on output
        return q_values

# Create Q-network for CartPole
state_dim = 4      # CartPole state: position, velocity, angle, angular velocity
action_dim = 2     # Actions: left or right
q_net = QNetwork(state_dim, action_dim)

print("Q-Network:")
print(q_net)

# Simulate selecting action
state = torch.randn(1, state_dim)
q_values = q_net(state)
action = torch.argmax(q_values, dim=1)

print(f"\nState: {state}")
print(f"Q-values: {q_values}")
print(f"Selected action: {action.item()}")
```

### 6. Training Loop Basics

```python
# Simple training example: learn y = 2x + 3 from data
torch.manual_seed(42)

# Generate synthetic data
x_train = torch.randn(100, 1)
y_train = 2 * x_train + 3 + torch.randn(100, 1) * 0.1  # Add noise

# Define model
model = nn.Sequential(
    nn.Linear(1, 1)
)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # Forward pass
    predictions = model(x_train)
    loss = criterion(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights

    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Check learned parameters
learned_weight = model[0].weight.item()
learned_bias = model[0].bias.item()
print(f"\nLearned: y = {learned_weight:.2f}x + {learned_bias:.2f}")
print(f"True:    y = 2.00x + 3.00")

# Plot training loss
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5, label='Data')
x_test = torch.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = model(x_test).detach().numpy()
plt.plot(x_test.numpy(), y_pred, 'r-', label='Learned function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitted Line')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('pytorch_training.png')
print("\nPlot saved as pytorch_training.png")
```

### 7. GPU Acceleration

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move tensors to GPU
if torch.cuda.is_available():
    # Create tensor on GPU
    tensor_gpu = torch.randn(3, 3, device='cuda')
    print(f"\nTensor on GPU: {tensor_gpu.device}")

    # Move existing tensor to GPU
    tensor_cpu = torch.randn(3, 3)
    tensor_gpu = tensor_cpu.to('cuda')

    # Move model to GPU
    model = QNetwork(4, 2).to(device)
    print(f"Model device: {next(model.parameters()).device}")

    # In training loop, move data to same device as model
    state = torch.randn(1, 4).to(device)
    q_values = model(state)

    # Move back to CPU for NumPy conversion
    q_values_cpu = q_values.cpu()
else:
    print("GPU not available, using CPU")

# Performance comparison
def benchmark_computation(device, size=10000):
    import time

    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    start = time.time()
    C = A @ B
    if device == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU to finish
    end = time.time()

    return end - start

if torch.cuda.is_available():
    cpu_time = benchmark_computation('cpu', 1000)
    gpu_time = benchmark_computation('cuda', 1000)

    print(f"\nMatrix multiplication (1000x1000):")
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")
```

### 8. Saving and Loading Models

```python
# Train a simple model
model = QNetwork(4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Save model
torch.save(model.state_dict(), 'q_network.pth')
print("Model saved to q_network.pth")

# Save full checkpoint (model + optimizer state)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 100,
}
torch.save(checkpoint, 'checkpoint.pth')
print("Checkpoint saved to checkpoint.pth")

# Load model
loaded_model = QNetwork(4, 2)
loaded_model.load_state_dict(torch.load('q_network.pth'))
loaded_model.eval()  # Set to evaluation mode
print("\nModel loaded successfully")

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
print(f"Resumed from epoch {epoch}")
```

## 📚 Key References

### Official Documentation
- [PyTorch Official Tutorial](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

### Tutorials
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [PyTorch for Deep Learning (freeCodeCamp)](https://www.youtube.com/watch?v=V_xro1bcAuA)

### Books
- Eli Stevens et al. - "Deep Learning with PyTorch" - [Manning](https://www.manning.com/books/deep-learning-with-pytorch)

### RL-Specific
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Spinning Up: PyTorch Introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

## 🏋️ Exercises

### Exercise 1: Tensor Manipulation (Easy)

Create a function that takes a batch of states and normalizes them:

```python
def normalize_states(states):
    """
    Normalize states to have mean=0 and std=1 for each feature.

    Args:
        states: torch.Tensor of shape (batch_size, state_dim)

    Returns:
        normalized_states: torch.Tensor of same shape
    """
    # Your code here
    pass

# Test
states = torch.randn(100, 4) * 10 + 5  # Mean≈5, std≈10
normalized = normalize_states(states)

# Verify
print(f"Original - Mean: {states.mean(dim=0)}, Std: {states.std(dim=0)}")
print(f"Normalized - Mean: {normalized.mean(dim=0)}, Std: {normalized.std(dim=0)}")
```

### Exercise 2: Gradient Computation (Medium)

Given the Bellman error function used in Q-learning:

L = (Q(s,a) - (r + γ * max_a' Q(s',a')))²

Manually compute the gradient ∂L/∂Q(s,a) and verify it matches PyTorch's autograd.

```python
# Given values
Q_sa = torch.tensor([5.0], requires_grad=True)
r = torch.tensor([10.0])
gamma = 0.9
Q_next_max = torch.tensor([8.0])

# Compute loss
target = r + gamma * Q_next_max
loss = (Q_sa - target) ** 2

# Your analytical gradient here
analytical_gradient = None  # Calculate this

# PyTorch gradient
loss.backward()
pytorch_gradient = Q_sa.grad

print(f"Analytical gradient: {analytical_gradient}")
print(f"PyTorch gradient: {pytorch_gradient.item()}")
print(f"Match: {abs(analytical_gradient - pytorch_gradient.item()) < 1e-6}")
```

### Exercise 3: Custom Neural Network (Medium)

Build a "Dueling DQN" architecture with:
- Shared feature layers
- Separate value stream V(s)
- Separate advantage stream A(s,a)
- Combined output: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        # Your code here
        pass

    def forward(self, state):
        # Your code here
        pass

# Test
net = DuelingQNetwork(4, 2)
state = torch.randn(1, 4)
q_values = net(state)
print(f"Q-values: {q_values}")
```

### Exercise 4: Training Loop (Medium-Hard)

Implement a complete training loop for learning the XOR function:

Inputs: [[0,0], [0,1], [1,0], [1,1]]
Outputs: [0, 1, 1, 0]

Requirements:
- Use a network with at least one hidden layer
- Train until loss < 0.01 or 10000 epochs
- Plot the training loss
- Visualize the decision boundary

```python
# Your code here
# Hints:
# - XOR is not linearly separable, need hidden layer
# - Use sigmoid activation on output
# - Use BCELoss (binary cross-entropy)
# - Learning rate around 0.1
```

### Exercise 5: GPU Performance (Conceptual + Coding)

**Part A (Conceptual):**
Explain when you should and shouldn't use GPU acceleration in RL training.

**Part B (Coding):**
Write a function that automatically handles device placement:

```python
def get_device(prefer_gpu=True):
    """Returns appropriate device and prints info."""
    # Your code here
    pass

def to_device(data, device):
    """
    Recursively move data to device.
    Handles tensors, lists, tuples, and dicts.
    """
    # Your code here
    pass

# Test
device = get_device()
data = {
    'states': torch.randn(10, 4),
    'actions': torch.randint(0, 2, (10,)),
    'rewards': torch.randn(10),
}
data_on_device = to_device(data, device)
```

## 🔧 Troubleshooting Tips

### Common Issues

1. **RuntimeError: Expected all tensors to be on the same device**
   - Mixing CPU and GPU tensors
   - Solution: Use `.to(device)` consistently

2. **RuntimeError: Trying to backward through the graph a second time**
   - Reusing computation graph
   - Solution: Set `retain_graph=True` or recompute forward pass

3. **Gradients not updating**
   - Forgot `optimizer.zero_grad()`
   - Solution: Always clear gradients before backward pass

4. **Out of memory on GPU**
   - Batch size too large
   - Solution: Reduce batch size or use gradient accumulation

5. **Gradient is None**
   - Tensor not created with `requires_grad=True`
   - Operations that don't track gradients (e.g., `tensor.numpy()`)
   - Solution: Check `requires_grad`, use `.detach()` intentionally

### Best Practices

```python
# 1. Always set model to train/eval mode
model.train()    # Enable dropout, batch norm training mode
model.eval()     # Disable dropout, batch norm eval mode

# 2. Use torch.no_grad() for inference
with torch.no_grad():
    output = model(input)

# 3. Detach tensors when needed
loss = compute_loss(output.detach(), target)

# 4. Clear gradients before backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 5. Move data and model to same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

## ✅ Self-Check

Before moving to Lesson 3, you should be able to:

- [ ] Create and manipulate PyTorch tensors
- [ ] Understand how autograd computes gradients
- [ ] Build custom neural networks using nn.Module
- [ ] Implement a basic training loop
- [ ] Use GPU acceleration when available
- [ ] Save and load model checkpoints
- [ ] Debug common PyTorch errors
- [ ] Explain when to use `.detach()` and `torch.no_grad()`

## 🚀 Next Steps

Now that you understand PyTorch fundamentals, you're ready for [Lesson 3: Supervised Learning Fundamentals](lesson_03_supervised_learning.md), where you'll learn:
- The supervised learning paradigm
- Loss functions and their properties
- Optimization algorithms (SGD, Adam, etc.)
- Overfitting, underfitting, and regularization
- Training, validation, and test sets

**Optional practice:**
- Implement linear regression from scratch in PyTorch
- Build a simple autoencoder
- Experiment with different activation functions
- Profile GPU vs CPU performance on your machine

---

**Estimated completion time:** 4-6 hours (including exercises)

**Next lesson:** [Lesson 3: Supervised Learning Fundamentals →](lesson_03_supervised_learning.md)
