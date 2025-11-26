# Lesson 6: Tabular Q-Learning Implementation

**Duration:** 4-6 hours

**Prerequisites:** Lesson 5 (RL Theory)

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Implement Q-learning algorithm from scratch
2. Understand temporal difference (TD) learning
3. Master ε-greedy exploration
4. Train agents on GridWorld and FrozenLake environments
5. Debug common Q-learning issues
6. Visualize Q-value convergence
7. Understand the difference between SARSA and Q-learning

## 📖 Theory

### Q-Learning Algorithm

Q-learning learns the optimal Q-function directly without needing a model of the environment.

**Update rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
                        \_____________________/
                                TD error
```

**Components:**
- α: learning rate (step size)
- γ: discount factor
- TD target: r + γ max_{a'} Q(s',a')
- TD error: how wrong our current Q-value is

**Key properties:**
- **Off-policy:** Learns optimal Q* while following ε-greedy policy
- **Model-free:** Doesn't need to know transition probabilities
- **Guaranteed convergence:** Under certain conditions (tabular case, proper α decay)

### Temporal Difference (TD) Learning

TD learning updates value estimates based on other estimates (bootstrapping):

**Monte Carlo:** Wait until end of episode
```
V(s) ← V(s) + α[G_t - V(s)]  // G_t = actual return
```

**TD(0):** Update immediately after one step
```
V(s) ← V(s) + α[r + γV(s') - V(s)]  // Bootstrap from V(s')
```

**Advantages of TD:**
- Learn from incomplete episodes
- Lower variance (but higher bias)
- Works in continuing (non-episodic) tasks

### ε-Greedy Exploration

Balance exploration and exploitation:

```python
if random() < ε:
    action = random_action()  # Explore
else:
    action = argmax_a Q(s,a)  # Exploit
```

**ε decay:** Start with high ε, decrease over time
```python
ε = ε_min + (ε_max - ε_min) * exp(-decay_rate * episode)
```

### SARSA vs Q-Learning

**SARSA (on-policy):**
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
                              ^
                    actual action taken
```

**Q-Learning (off-policy):**
```
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
                              ^^^
                    best possible action
```

**Difference:** SARSA learns about policy it's following, Q-learning learns about optimal policy.

## 💻 Practical Implementation

### Setup

```python
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm

np.random.seed(42)
```

### 1. Q-Learning Agent

```python
class QLearningAgent:
    """Tabular Q-learning agent."""

    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions))

    def get_action(self, state, training=True):
        """Select action using ε-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """Q-learning update."""
        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        # TD error
        td_error = td_target - self.Q[state, action]

        # Q-table update
        self.Q[state, action] += self.lr * td_error

        return td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Test on simple GridWorld
print("Q-Learning Agent created")
```

### 2. Train on FrozenLake

```python
# Create FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

print(f"FrozenLake: {n_states} states, {n_actions} actions")
print("\nEnvironment description:")
print("S: Start, F: Frozen, H: Hole, G: Goal")
print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")

# Create agent
agent = QLearningAgent(n_states, n_actions,
                       learning_rate=0.1,
                       gamma=0.99,
                       epsilon=1.0,
                       epsilon_decay=0.9995)

# Training
n_episodes = 10000
episode_rewards = []
episode_lengths = []
td_errors = []

for episode in tqdm(range(n_episodes), desc="Training"):
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    episode_td_errors = []

    for step in range(100):
        # Select and take action
        action = agent.get_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update Q-table
        td_error = agent.update(state, action, reward, next_state, done)
        episode_td_errors.append(abs(td_error))

        total_reward += reward
        steps += 1
        state = next_state

        if done:
            break

    # Decay epsilon
    agent.decay_epsilon()

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    td_errors.append(np.mean(episode_td_errors))

print(f"\nTraining complete!")
print(f"Final epsilon: {agent.epsilon:.4f}")

# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# Moving average helper
def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Rewards
axes[0].plot(moving_average(episode_rewards))
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Average Reward (100 ep)')
axes[0].set_title('Training Rewards')
axes[0].grid(True)

# Episode lengths
axes[1].plot(moving_average(episode_lengths))
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Average Steps (100 ep)')
axes[1].set_title('Episode Length')
axes[1].grid(True)

# TD errors
axes[2].plot(moving_average(td_errors))
axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Average |TD Error|')
axes[2].set_title('TD Error (Convergence Indicator)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('qlearning_training.png')
print("Training curves saved")
```

### 3. Evaluate Trained Agent

```python
def evaluate_agent(agent, env, n_episodes=100):
    """Evaluate agent performance."""
    rewards = []
    successes = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(100):
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        if total_reward > 0:
            successes += 1

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': successes / n_episodes
    }

results = evaluate_agent(agent, env, n_episodes=1000)
print(f"\nEvaluation Results (1000 episodes):")
print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
print(f"Success Rate: {results['success_rate']:.1%}")
```

### 4. Visualize Q-Values

```python
def visualize_q_table(agent, env_name='FrozenLake-v1'):
    """Visualize Q-table as heatmap."""
    Q = agent.Q

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']

    for action in range(4):
        ax = axes[action // 2, action % 2]

        # Reshape Q-values for this action (4x4 grid for FrozenLake)
        q_values = Q[:, action].reshape(4, 4)

        im = ax.imshow(q_values, cmap='coolwarm')
        ax.set_title(f'Q-values for {action_names[action]}')

        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax.text(j, i, f'{q_values[i, j]:.2f}',
                             ha="center", va="center", color="black")

        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('q_table_visualization.png')
    print("Q-table visualization saved")

visualize_q_table(agent)

def visualize_policy(agent):
    """Visualize learned policy."""
    policy = np.argmax(agent.Q, axis=1).reshape(4, 4)
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            action = policy[i, j]
            ax.text(j, i, arrows[action], ha='center', va='center',
                   fontsize=24)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_title('Learned Policy (arrows show actions)')
    plt.savefig('learned_policy.png')
    print("Policy visualization saved")

visualize_policy(agent)
```

### 5. SARSA Implementation

```python
class SARSAAgent(QLearningAgent):
    """SARSA agent (on-policy TD control)."""

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update - uses actual next action."""
        if done:
            td_target = reward
        else:
            # Use Q(s', a') instead of max Q(s', a)
            td_target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

        return td_error

# Train SARSA
sarsa_agent = SARSAAgent(n_states, n_actions)

sarsa_rewards = []
for episode in tqdm(range(5000), desc="Training SARSA"):
    state, _ = env.reset()
    action = sarsa_agent.get_action(state, training=True)
    total_reward = 0

    for step in range(100):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = sarsa_agent.get_action(next_state, training=True)

        # SARSA update
        sarsa_agent.update(state, action, reward, next_state, next_action, done)

        total_reward += reward
        state = next_state
        action = next_action

        if done:
            break

    sarsa_agent.decay_epsilon()
    sarsa_rewards.append(total_reward)

# Compare Q-learning vs SARSA
plt.figure(figsize=(10, 4))
plt.plot(moving_average(episode_rewards, 100), label='Q-Learning', alpha=0.7)
plt.plot(moving_average(sarsa_rewards, 100), label='SARSA', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Q-Learning vs SARSA')
plt.legend()
plt.grid(True)
plt.savefig('qlearning_vs_sarsa.png')
print("Comparison saved")
```

### 6. Hyperparameter Sensitivity

```python
def train_with_params(lr, gamma, epsilon_decay, n_episodes=5000):
    """Train agent with specific hyperparameters."""
    agent = QLearningAgent(n_states, n_actions,
                          learning_rate=lr,
                          gamma=gamma,
                          epsilon_decay=epsilon_decay)
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(100):
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)

    return moving_average(rewards, 100)

# Test different learning rates
learning_rates = [0.01, 0.1, 0.5]

plt.figure(figsize=(10, 4))
for lr in learning_rates:
    rewards = train_with_params(lr, gamma=0.99, epsilon_decay=0.9995)
    plt.plot(rewards, label=f'lr={lr}')

plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Learning Rate Sensitivity')
plt.legend()
plt.grid(True)
plt.savefig('lr_sensitivity.png')
print("Learning rate sensitivity analysis done")
```

### 7. Custom GridWorld with Q-Learning

```python
# Reuse GridWorld from Lesson 5
class GridWorldEnv:
    """GridWorld as Gym-like environment."""

    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4

        self.goal_state = (4, 4)
        self.pits = [(2, 2), (3, 1)]
        self.start_state = (0, 0)

        self.state = self.start_state

    def state_to_index(self, state):
        return state[0] * self.size + state[1]

    def index_to_state(self, index):
        return (index // self.size, index % self.size)

    def reset(self):
        self.state = self.start_state
        return self.state_to_index(self.state)

    def step(self, action):
        row, col = self.state

        # Take action
        if action == 0:    # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)

        self.state = (row, col)
        state_idx = self.state_to_index(self.state)

        # Determine reward and done
        if self.state == self.goal_state:
            reward = 10.0
            done = True
        elif self.state in self.pits:
            reward = -10.0
            done = True
        else:
            reward = -0.1
            done = False

        return state_idx, reward, done

# Train on GridWorld
grid_env = GridWorldEnv(size=5)
grid_agent = QLearningAgent(grid_env.n_states, grid_env.n_actions,
                           learning_rate=0.1, gamma=0.95)

grid_rewards = []
for episode in tqdm(range(2000), desc="GridWorld Training"):
    state = grid_env.reset()
    total_reward = 0

    for step in range(100):
        action = grid_agent.get_action(state, training=True)
        next_state, reward, done = grid_env.step(action)

        grid_agent.update(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if done:
            break

    grid_agent.decay_epsilon()
    grid_rewards.append(total_reward)

plt.figure(figsize=(10, 4))
plt.plot(moving_average(grid_rewards, 50))
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Q-Learning on Custom GridWorld')
plt.grid(True)
plt.savefig('gridworld_training.png')
```

## 📚 Key References

### Textbooks
- **Sutton & Barto** - "RL: An Introduction" - Chapter 6 (Temporal-Difference Learning)
- Read sections 6.1-6.5 carefully

### Papers
- **Watkins & Dayan (1992)** - "Q-Learning" - [Link](https://link.springer.com/article/10.1007/BF00992698)

### Tutorials
- [Spinning Up: Q-Learning](https://spinningup.openai.com/en/latest/algorithms/dqn.html)
- [OpenAI Gym Tutorial](https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/)

## 🏋️ Exercises

### Exercise 1: Learning Rate Decay (Easy)

Implement learning rate decay: α_t = α_0 / (1 + decay_rate * t)

Compare fixed vs decaying learning rate. Which converges better?

### Exercise 2: Double Q-Learning (Medium)

Implement Double Q-Learning to reduce overestimation bias:
- Maintain two Q-tables: Q1 and Q2
- Alternate updates between them
- For action selection: a* = argmax Q1(s,a)
- For evaluation: Q2(s, a*)

### Exercise 3: Cliff Walking (Medium-Hard)

Solve the Cliff Walking environment:
```python
env = gym.make('CliffWalking-v0')
```

Compare Q-learning vs SARSA. Which takes the safer path?

### Exercise 4: N-step Q-Learning (Hard)

Implement n-step Q-learning:
- Store last n transitions
- Compute n-step return
- Update Q-values

Compare n=1 (standard), n=3, n=5, n=10

### Exercise 5: Optimistic Initialization (Medium)

Initialize Q-table with high values (e.g., all 10.0) instead of zeros.

Does this encourage exploration? How does it affect learning?

## 🔧 Troubleshooting Tips

### Common Issues

1. **Agent not learning**
   - Learning rate too low/high
   - Epsilon not decaying
   - Insufficient exploration
   - Terminal states not handled correctly

2. **Unstable learning**
   - Learning rate too high
   - Discount factor too close to 1
   - Environment has loops

3. **No improvement**
   - Stuck in local optimum (increase exploration)
   - Reward structure unclear
   - Need more episodes

4. **Slow convergence**
   - Learning rate too conservative
   - Epsilon decay too slow
   - Q-table initialization

### Debugging Checklist

```python
# 1. Check Q-table updates
print(f"Q-table before: {agent.Q[state, action]}")
agent.update(state, action, reward, next_state, done)
print(f"Q-table after: {agent.Q[state, action]}")

# 2. Monitor TD errors
if abs(td_error) > 10:
    print(f"Large TD error: {td_error}")

# 3. Check epsilon decay
print(f"Episode {ep}: epsilon = {agent.epsilon:.4f}")

# 4. Verify action distribution
actions_taken = [agent.get_action(0) for _ in range(1000)]
print(f"Action distribution: {np.bincount(actions_taken)}")
```

## ✅ Self-Check

Before moving to Lesson 7, you should be able to:

- [ ] Implement Q-learning from scratch
- [ ] Understand TD error and bootstrapping
- [ ] Use ε-greedy exploration effectively
- [ ] Tune learning rate, discount factor, epsilon
- [ ] Explain on-policy vs off-policy
- [ ] Visualize and interpret Q-tables
- [ ] Debug convergence issues

## 🚀 Next Steps

You've mastered tabular Q-learning! But it doesn't scale to large state spaces...

**Next:** [Lesson 7: Deep Q-Networks (DQN)](lesson_07_dqn.md)
- Replace Q-table with neural network
- Experience replay
- Target networks
- Solve CartPole and Atari games

Time to go deep!

---

**Estimated completion time:** 4-6 hours

**Next lesson:** [Lesson 7: Deep Q-Networks →](lesson_07_dqn.md)
