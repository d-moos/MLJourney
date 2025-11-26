# Lesson 1: Python Fundamentals and Scientific Computing with NumPy

**Duration:** 4-5 hours

**Prerequisites:** Basic programming concepts (variables, loops, functions)

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Understand and use NumPy arrays for efficient numerical computation
2. Perform vectorized operations instead of Python loops
3. Master array indexing, slicing, and broadcasting
4. Understand why vectorization matters for ML/RL
5. Manipulate multi-dimensional arrays representing states and actions

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

### Why NumPy for Machine Learning?

Python lists are flexible but slow for numerical computation. NumPy provides:

- **Fast operations:** C-level performance for array operations
- **Vectorization:** Apply operations to entire arrays without loops
- **Memory efficiency:** Contiguous memory storage
- **Broadcasting:** Automatic expansion of dimensions for compatible operations

### Core Concepts

#### 1. Arrays vs Lists

```python
# Python list - each element is a separate object
python_list = [1, 2, 3, 4, 5]

# NumPy array - single contiguous block of memory
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])
```

**Key difference:** NumPy arrays have a fixed type and size, making them much faster.

#### 2. Vectorization

Vectorization means writing operations on **whole arrays at once** instead of
explicit Python `for` loops. NumPy then performs the loop internally in fast
compiled code (C), so the same math runs much faster and uses your CPU better.

Instead of looping over elements:

```python
# Slow: Python loop
result = []
for x in range(1000):
    result.append(x ** 2)

# Fast: NumPy vectorized operation
x = np.arange(1000)
result = x ** 2  # 10-100x faster!
```

#### 3. Broadcasting

Broadcasting is NumPy’s rule for automatically **stretching smaller arrays**
so that element‑wise operations like `+` or `*` still make sense when shapes
are different but compatible.

NumPy automatically expands arrays to compatible shapes:

```python
# Adding scalar to array
arr = np.array([1, 2, 3])
result = arr + 10  # [11, 12, 13]

# Row + column = matrix
row = np.array([[1, 2, 3]])        # shape: (1, 3)
col = np.array([[10], [20], [30]]) # shape: (3, 1)
matrix = row + col                  # shape: (3, 3)
```

#### 4. Why This Matters for RL

In RL, you'll frequently:
- Process batches of states (2D/3D arrays)
- Apply actions across multiple environments simultaneously
- Compute rewards for many transitions at once
- Update Q-values or policy parameters using vectorized math

**Example:** Computing Q-values for a batch of states

```python
# Without NumPy: slow, one at a time
q_values = []
for state in states:
    q = compute_q_value(state)
    q_values.append(q)

# With NumPy: fast, vectorized
# states shape: (batch_size, state_dim)
# weights shape: (state_dim, num_actions)
q_values = states @ weights  # Matrix multiplication, all at once!
```

## 💻 Practical Implementation

### Setup

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Check NumPy version
print(f"NumPy version: {np.__version__}")
```

### 1. Creating Arrays

```python
# From Python lists
arr1 = np.array([1, 2, 3, 4, 5])
print(f"1D array: {arr1}")
print(f"Shape: {arr1.shape}, Dtype: {arr1.dtype}")

# 2D array (matrix)
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(f"\n2D array:\n{arr2}")
print(f"Shape: {arr2.shape}")

# Useful creation functions
zeros = np.zeros((3, 4))        # 3x4 matrix of zeros
ones = np.ones((2, 3))          # 2x3 matrix of ones
identity = np.eye(4)             # 4x4 identity matrix
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # 5 evenly spaced values from 0 to 1

# Random arrays (crucial for RL!)
random_uniform = np.random.rand(3, 3)           # Uniform [0, 1)
random_normal = np.random.randn(3, 3)           # Standard normal
random_int = np.random.randint(0, 10, size=(3, 3))  # Random integers

print(f"\nRandom normal array:\n{random_normal}")
```

### 2. Indexing and Slicing

```python
# 1D indexing
arr = np.array([10, 20, 30, 40, 50])
print(f"First element: {arr[0]}")
print(f"Last element: {arr[-1]}")
print(f"Slice [1:4]: {arr[1:4]}")  # Elements 1, 2, 3

# 2D indexing
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"\nElement at (1,2): {matrix[1, 2]}")  # Row 1, Column 2 = 6
print(f"First row: {matrix[0, :]}")
print(f"Second column: {matrix[:, 1]}")

# Boolean indexing (very useful for filtering!)
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3
print(f"\nMask: {mask}")
print(f"Elements > 3: {arr[mask]}")  # [4, 5, 6]

# Practical RL example: Select positive rewards
rewards = np.array([-1, 0, 5, -2, 10, 3])
positive_rewards = rewards[rewards > 0]
print(f"Positive rewards: {positive_rewards}")
```

### 3. Array Operations

```python
# Element-wise operations
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"a + b = {a + b}")      # [11, 22, 33, 44]
print(f"a * b = {a * b}")      # [10, 40, 90, 160] - element-wise!
print(f"a ** 2 = {a ** 2}")    # [1, 4, 9, 16]

# Universal functions (ufuncs)
angles = np.array([0, np.pi/2, np.pi])
print(f"\nsin(angles) = {np.sin(angles)}")
print(f"exp([1,2,3]) = {np.exp([1, 2, 3])}")

# Aggregation functions
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(f"\nSum of all elements: {np.sum(arr)}")
print(f"Sum along axis 0 (columns): {np.sum(arr, axis=0)}")  # [5, 7, 9]
print(f"Sum along axis 1 (rows): {np.sum(arr, axis=1)}")     # [6, 15]
print(f"Mean: {np.mean(arr)}")
print(f"Max: {np.max(arr)}")
print(f"Argmax (index of max): {np.argmax(arr)}")

# Statistical functions (important for RL!)
data = np.random.randn(1000)
print(f"\nMean: {np.mean(data):.3f}")
print(f"Std dev: {np.std(data):.3f}")
print(f"Min: {np.min(data):.3f}, Max: {np.max(data):.3f}")
```

### 4. Broadcasting Examples

```python
# Example 1: Normalize an array
arr = np.array([10, 20, 30, 40, 50])
mean = np.mean(arr)
std = np.std(arr)
normalized = (arr - mean) / std  # Broadcasting!
print(f"Normalized: {normalized}")

# Example 2: Apply discount factor to rewards
rewards = np.array([1, 2, 3, 4, 5])
gamma = 0.9
discounts = gamma ** np.arange(len(rewards))  # [1, 0.9, 0.81, ...]
discounted_rewards = rewards * discounts
print(f"\nDiscounted rewards: {discounted_rewards}")

# Example 3: Batch state processing
# Imagine 100 states, each with 4 features
states = np.random.randn(100, 4)
# Normalize each feature (column)
state_mean = np.mean(states, axis=0, keepdims=True)  # Shape: (1, 4)
state_std = np.std(states, axis=0, keepdims=True)
normalized_states = (states - state_mean) / state_std  # Broadcasting!
print(f"\nBatch normalization - Original shape: {states.shape}")
print(f"Mean shape: {state_mean.shape}")
print(f"Normalized shape: {normalized_states.shape}")
```

### 5. Matrix Operations

```python
# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot_product}")

# Matrix multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Three equivalent ways to multiply matrices
C1 = np.dot(A, B)
C2 = A @ B          # Python 3.5+ operator
C3 = np.matmul(A, B)

print(f"\nMatrix multiplication:\n{C1}")

# Transpose
print(f"\nTranspose of A:\n{A.T}")

# RL Example: Computing Q-values from states
# States: (batch_size, state_dim) = (32, 4)
# Weights: (state_dim, num_actions) = (4, 2)
# Q-values: (batch_size, num_actions) = (32, 2)

batch_size, state_dim, num_actions = 32, 4, 2
states = np.random.randn(batch_size, state_dim)
weights = np.random.randn(state_dim, num_actions)
q_values = states @ weights

print(f"\nRL Q-value computation:")
print(f"States shape: {states.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Q-values shape: {q_values.shape}")
print(f"Q-values for first state: {q_values[0]}")
```

### 6. Reshaping and Stacking

```python
# Reshape
arr = np.arange(12)
print(f"Original: {arr}")
print(f"Reshaped (3,4):\n{arr.reshape(3, 4)}")
print(f"Reshaped (2,6):\n{arr.reshape(2, 6)}")

# Flatten
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"\nFlattened: {matrix.flatten()}")

# Stack arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

v_stack = np.vstack([a, b])  # Vertical stack
h_stack = np.hstack([a, b])  # Horizontal stack

print(f"\nVertical stack:\n{v_stack}")
print(f"Horizontal stack: {h_stack}")

# RL Example: Concatenate state and action
state = np.array([1.0, 2.0, 3.0])
action = np.array([0.5, -0.3])
state_action = np.concatenate([state, action])
print(f"\nState-action concatenation: {state_action}")
```

### 7. Performance Comparison

```python
import time

# Compare loop vs vectorized performance
size = 1000000

# Python loop
start = time.time()
result_loop = []
for i in range(size):
    result_loop.append(i ** 2)
loop_time = time.time() - start

# NumPy vectorized
start = time.time()
arr = np.arange(size)
result_vec = arr ** 2
vec_time = time.time() - start

print(f"Loop time: {loop_time:.4f}s")
print(f"Vectorized time: {vec_time:.4f}s")
print(f"Speedup: {loop_time / vec_time:.1f}x faster!")
```

### 8. Visualization with Matplotlib

```python
import matplotlib.pyplot as plt

# Plot a simple function
x = np.linspace(-5, 5, 100)
y = x ** 2

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Quadratic Function')
plt.xlabel('x')
plt.ylabel('y = x²')
plt.grid(True)

# Plot multiple functions
plt.subplot(1, 2, 2)
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.title('Trigonometric Functions')
plt.xlabel('x')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('numpy_plots.png')
print("Plot saved as numpy_plots.png")
```

## 📚 Key References

### Official Documentation
- [NumPy Official Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

### Tutorials
- [Python NumPy Tutorial (Stanford CS231n)](http://cs231n.github.io/python-numpy-tutorial/)
- [NumPy Illustrated: The Visual Guide to NumPy](https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d)

### Books
- Jake VanderPlas - "Python Data Science Handbook" (Chapter 2) - [Free online](https://jakevdp.github.io/PythonDataScienceHandbook/)

### Video Resources
- [NumPy Crash Course - Keith Galli](https://www.youtube.com/watch?v=QUT1VHiLmmI)

## 🏋️ Exercises

### Exercise 1: Array Manipulation (Easy)

Create a 5x5 matrix where:
- The diagonal elements are all 1
- The elements above the diagonal are 2
- The elements below the diagonal are 0

```python
# Your code here
# Hint: Use np.eye(), np.triu(), np.tril()
```

### Exercise 2: Vectorized Computation (Medium)

You have an agent taking 1000 steps in an environment. The rewards at each step are randomly sampled from a normal distribution (mean=1.0, std=2.0).

Calculate the **discounted cumulative return** for each starting position using:
- Discount factor γ = 0.99
- Formula: R_t = r_t + γ*r_(t+1) + γ²*r_(t+2) + ...

Do this WITHOUT using any Python loops. Compare your vectorized solution's speed to a loop-based solution.

```python
import numpy as np

# Generate rewards
np.random.seed(42)
rewards = np.random.normal(1.0, 2.0, 1000)
gamma = 0.99

# Your vectorized solution here
# Hint: Think about creating a matrix of discount factors
```

### Exercise 3: State Normalization (Medium)

Given a batch of 200 environment states, where each state has 10 features:

1. Generate random states from normal distribution (mean=5, std=10)
2. Normalize each feature to have mean=0 and std=1
3. Clip any values outside [-3, 3] range
4. Verify your normalization worked

```python
# Your code here
# Hints: np.mean(), np.std(), np.clip()
```

### Exercise 4: Q-Table Initialization (Medium-Hard)

Create a Q-table for a GridWorld environment:
- Grid size: 10x10 (100 states)
- Actions: Up, Down, Left, Right (4 actions)
- Initialize Q-values optimistically (all values = 1.0)
- Set terminal state (position 99) Q-values to 0

Then implement a function that:
1. Takes a state index and returns the action with highest Q-value
2. Handles ties by randomly choosing among the best actions

```python
# Your code here
def get_best_action(q_table, state):
    """
    Returns the action with highest Q-value for given state.
    Breaks ties randomly.

    Args:
        q_table: numpy array of shape (num_states, num_actions)
        state: int, state index

    Returns:
        int: action index
    """
    pass

# Test your function
```

### Exercise 5: Performance Analysis (Conceptual + Coding)

**Part A (Conceptual):** Explain in your own words why NumPy operations are faster than Python loops. What does "vectorization" mean?

**Part B (Coding):** You need to apply the ReLU activation function (max(0, x)) to a large array.

Implement three versions:
1. Pure Python loop
2. Python list comprehension
3. NumPy vectorized

Measure and compare their execution times on an array of 1 million elements.

```python
import time
import numpy as np

def relu_loop(arr):
    """Pure Python loop version"""
    pass

def relu_comprehension(arr):
    """List comprehension version"""
    pass

def relu_numpy(arr):
    """NumPy vectorized version"""
    pass

# Benchmark code
arr = np.random.randn(1000000)
# Time each implementation
```

## 🔧 Troubleshooting Tips

### Common Issues

1. **"ValueError: could not broadcast..."**
   - Arrays have incompatible shapes
   - Solution: Check shapes with `.shape`, use `keepdims=True`, or reshape

2. **Slow performance despite using NumPy**
   - Still using Python loops instead of vectorized operations
   - Solution: Look for `for` loops and replace with array operations

3. **Memory errors with large arrays**
   - Creating arrays that don't fit in RAM
   - Solution: Use smaller batch sizes or data types like `float32` instead of `float64`

4. **Unexpected results with integer division**
   ```python
   np.array([1, 2]) / 2  # Returns floats: [0.5, 1.0]
   np.array([1, 2]) // 2 # Returns ints: [0, 1]
   ```

5. **Modifying array copies vs views**
   ```python
   a = np.array([1, 2, 3])
   b = a        # b is a view - modifying b changes a!
   c = a.copy() # c is a copy - independent
   ```

## ✅ Self-Check

Before moving to Lesson 2, you should be able to:

- [ ] Create NumPy arrays using various methods
- [ ] Index and slice arrays (including boolean indexing)
- [ ] Perform vectorized operations without loops
- [ ] Understand broadcasting rules
- [ ] Compute matrix operations (dot products, multiplication)
- [ ] Reshape and stack arrays
- [ ] Explain why NumPy is faster than Python lists
- [ ] Apply aggregation functions along specific axes

## 🚀 Next Steps

Now that you understand NumPy fundamentals, you're ready for [Lesson 2: Introduction to PyTorch](lesson_02_pytorch_intro.md), where you'll learn:
- PyTorch tensors (NumPy arrays on steroids!)
- Automatic differentiation with autograd
- Building neural networks
- GPU acceleration

**Optional practice:** Before moving on, try implementing some simple algorithms with NumPy:
- Matrix multiplication from scratch (without using @)
- K-means clustering
- Linear regression (closed-form solution)
- Softmax function

---

**Estimated completion time:** 4-5 hours (including exercises)

**Next lesson:** [Lesson 2: Introduction to PyTorch →](lesson_02_pytorch_intro.md)
