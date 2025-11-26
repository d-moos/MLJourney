# Lesson 7: Deep Q-Networks (DQN)

**Duration:** 6-8 hours

**Prerequisites:** Lessons 1-6

## 🎯 Learning Objectives

1. Understand function approximation for Q-learning
2. Implement experience replay buffers
3. Use target networks for stability
4. Build a DQN agent that solves CartPole
5. Understand the deadly triad and solutions
6. Debug DQN training issues
7. Explore DQN improvements (Double DQN, Dueling DQN)

## 📖 Theory

### Why Function Approximation?

Tabular Q-learning stores Q(s,a) for every state-action pair.

**Problems with large state spaces:**
- Atari games: 210×160×3 ≈ 10^{67000} possible states
- Cannot store in memory
- Never see same state twice → no learning

**Solution:** Use a neural network Q_θ(s,a) to approximate Q*(s,a)

### The Deadly Triad

Three things that cause instability when combined:
1. **Function approximation** (neural networks)
2. **Bootstrapping** (TD learning)
3. **Off-policy learning** (Q-learning)

DQN uses all three! Solutions needed:

### Experience Replay

**Problem:** Consecutive samples are correlated
**Solution:** Store transitions in replay buffer, sample randomly

```python
buffer = []
...
buffer.append((s, a, r, s', done))
batch = random_sample(buffer, batch_size)
```

**Benefits:**
- Breaks correlation
- Reuse experiences (sample efficiency)
- Smooths out learning

### Target Networks

**Problem:** Target changes every update (moving target)
**Solution:** Separate target network, update periodically

```python
# Main network (updated every step)
Q_θ(s,a)

# Target network (updated every C steps)
Q_θ'(s',a') ← Q_θ(s',a')
```

**Update rule:**
```
y = r + γ max_a' Q_θ'(s',a')  // Use target network
loss = (Q_θ(s,a) - y)²
```

### DQN Algorithm

```
Initialize Q_θ, target Q_θ', replay buffer D
For episode = 1..M:
    Reset environment, get state s
    For t = 1..T:
        Select action: a = ε-greedy(Q_θ(s,·))
        Execute a, observe r, s'
        Store (s,a,r,s') in D
        Sample mini-batch from D
        Compute targets: y = r + γ max_a' Q_θ'(s',a')
        Update Q_θ: θ ← θ - α∇(Q_θ(s,a) - y)²
        Every C steps: θ' ← θ
        s ← s'
```

### Improvements

**Double DQN:**
- Reduces overestimation bias
- Select action with Q_θ, evaluate with Q_θ'

**Dueling DQN:**
- Split network: V(s) + A(s,a)
- Better for states where actions don't matter

**Prioritized Experience Replay:**
- Sample important transitions more often
- Weight by TD error

## 💻 Practical Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

# Train on CartPole
env = gym.make('CartPole-v1')
agent = DQNAgent(state_dim=4, action_dim=2)

episodes = 500
rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.push(state, action, reward, next_state, done)
        loss = agent.update()

        total_reward += reward
        state = next_state

        if done:
            break

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    rewards.append(total_reward)

    if ep % 50 == 0:
        avg = np.mean(rewards[-50:])
        print(f"Episode {ep}, Avg Reward: {avg:.2f}, Epsilon: {agent.epsilon:.3f}")
```

## 📚 Key References

### Papers
- **Mnih et al. (2015)** - "Human-level control through deep RL" - [Nature](https://www.nature.com/articles/nature14236)
- **van Hasselt et al. (2015)** - "Deep Reinforcement Learning with Double Q-learning" - [arXiv](https://arxiv.org/abs/1509.06461)
- **Wang et al. (2016)** - "Dueling Network Architectures" - [arXiv](https://arxiv.org/abs/1511.06581)
- **Schaul et al. (2015)** - "Prioritized Experience Replay" - [arXiv](https://arxiv.org/abs/1511.05952)

### Tutorials & Blogs
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) - Official PyTorch implementation
- [Spinning Up: DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) - OpenAI's guide with key equations
- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) - Andrej Karpathy's classic blog post
- [Lilian Weng's RL Overview](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) - Comprehensive DQN explanation

### Video Lectures
- [David Silver RL Course - Lecture 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps) - Theory foundation
- [DeepMind x UCL RL Lecture Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) - Full course

### Code Examples
- [CleanRL DQN Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py) - Single-file, well-commented
- [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) - Production-ready library

## 🏋️ Exercises

1. **Implement Double DQN** - Reduce overestimation bias
2. **Add Dueling architecture** - Separate value and advantage streams
3. **Solve LunarLander-v2** - Achieve average reward >200
4. **Implement Prioritized Experience Replay** - Sample important transitions more
5. **Train on Atari Pong** - Use frame stacking and CNN from Lesson 10

## 🔧 Troubleshooting Tips

### Common Issues

**1. Training is unstable / rewards don't improve**
- **Check:** Is epsilon decaying too fast? Agent needs exploration.
- **Solution:** Slow down epsilon decay or increase `epsilon_min`
- **Check:** Is the target network updating too frequently?
- **Solution:** Increase `target_update` frequency (e.g., every 1000 steps)

**2. Loss explodes or becomes NaN**
- **Cause:** Q-values growing unbounded
- **Solution:** Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)`
- **Solution:** Reduce learning rate (try 1e-4 instead of 1e-3)
- **Solution:** Check reward scaling - normalize or clip rewards

**3. Agent learns suboptimal policy and gets stuck**
- **Cause:** Insufficient exploration or replay buffer too small
- **Solution:** Increase buffer size (e.g., 100k instead of 10k)
- **Solution:** Train longer before reducing epsilon
- **Solution:** Use epsilon-greedy with minimum epsilon (e.g., 0.01-0.05)

**4. Slow training**
- **Solution:** Use GPU if available: `device = torch.device("cuda")`
- **Solution:** Increase batch size (64 → 128 or 256)
- **Solution:** Reduce network size if environment is simple

**5. "Deadly triad" instability**
- **Symptoms:** Sudden performance collapse, oscillating Q-values
- **Solution:** All three DQN tricks are essential:
  - Experience replay (break correlation)
  - Target network (stabilize targets)
  - Gradient clipping (prevent explosions)

### Debugging Checklist

```python
# Add these to your training loop for debugging
print(f"Epsilon: {agent.epsilon:.3f}")
print(f"Buffer size: {len(agent.buffer)}")
print(f"Avg Q-value: {q_values.mean().item():.2f}")
print(f"Loss: {loss:.4f}")

# Plot Q-value distribution
import matplotlib.pyplot as plt
plt.hist(q_values.detach().cpu().numpy().flatten(), bins=50)
plt.title('Q-value Distribution')
plt.show()
```

## ✅ Self-Check

Before moving to Lesson 8, you should be able to:

- [ ] Explain why tabular Q-learning fails for large state spaces
- [ ] Describe the "deadly triad" and why it causes instability
- [ ] Implement experience replay from scratch
- [ ] Explain how target networks stabilize training
- [ ] Train a DQN agent that solves CartPole (avg reward >195)
- [ ] Debug common DQN training issues (exploding loss, no learning)
- [ ] Understand the difference between DQN, Double DQN, and Dueling DQN
- [ ] Know when to use DQN vs policy gradient methods

## 🚀 Next Steps

Now that you understand value-based deep RL, you're ready for [Lesson 8: Policy Gradient Methods and PPO](lesson_08_policy_gradients.md), where you'll learn:
- Direct policy optimization (no Q-function needed!)
- REINFORCE and actor-critic methods
- Proximal Policy Optimization (PPO)
- Handling continuous action spaces

**Optional challenge:** Before moving on, try to:
- Achieve >200 reward on LunarLander-v2 with DQN
- Implement Double DQN and compare performance
- Visualize learned Q-values for different states

---

**Duration:** 6-8 hours | **Next:** [Lesson 8 →](lesson_08_policy_gradients.md)
