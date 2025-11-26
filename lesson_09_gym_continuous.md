# Lesson 9: OpenAI Gym and Continuous Control

**Duration:** 4-5 hours

**Prerequisites:** Lessons 1-8

## 🎯 Learning Objectives

1. Master Gymnasium environment API
2. Handle continuous action spaces
3. Implement Gaussian policies for continuous control
4. Solve classic control tasks (Pendulum, LunarLander)
5. Understand action normalization and scaling
6. Create custom Gym environments
7. Use wrappers for preprocessing

## 📖 Theory

### Gymnasium API

```python
env = gym.make('env_name')
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
```

**Key concepts:**
- **terminated:** Episode ended naturally (goal/failure)
- **truncated:** Episode ended due to time limit
- **done = terminated or truncated**

### Continuous Action Spaces

**Discrete:** a ∈ {0, 1, 2, ...}
**Continuous:** a ∈ ℝ^n (e.g., joint torques, steering angles)

**Gaussian Policy:**
```
π_θ(a|s) = N(μ_θ(s), σ_θ(s))
```

Sample action: a ~ N(μ, σ)

### Action Normalization

Many environments expect actions in [-1, 1]:

```python
action_scaled = action_low + (action + 1) * (action_high - action_low) / 2
```

## 💻 Practical Implementation

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std.clamp(-20, 2))  # Stability
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

# Train on Pendulum
env = gym.make('Pendulum-v1')
print(f"State space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Custom environment wrapper
class NormalizedEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Normalize actions to environment bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Create custom environment
class SimpleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.state = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        reward = -np.sum(self.state ** 2)  # Get to origin
        self.state += action * 0.1
        self.state = np.clip(self.state, -1, 1)
        terminated = np.sum(self.state ** 2) < 0.01
        return self.state, reward, terminated, False, {}
```

## 📚 Key References

### Official Documentation
- [Gymnasium Documentation](https://gymnasium.farama.org/) - Complete API reference
- [Creating Custom Environments](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) - Step-by-step guide
- [Gymnasium Spaces](https://gymnasium.farama.org/api/spaces/) - Action and observation space types

### Tutorials & Blogs
- [Stable-Baselines3 Guide](https://stable-baselines3.readthedocs.io/) - Using SB3 with Gym
- [Hugging Face Deep RL Course - Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/introduction) - Gym basics
- [Continuous Control with Deep RL (DDPG paper)](https://arxiv.org/abs/1509.02971) - Lillicrap et al.
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) - State-of-the-art continuous control

### Code Examples
- [Stable-Baselines3 Examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html) - Continuous control demos
- [CleanRL Continuous Control](https://github.com/vwxyzjn/cleanrl#continuous-actions) - PPO, SAC, TD3 implementations
- [Gymnasium Wrappers](https://gymnasium.farama.org/api/wrappers/) - Preprocessing and normalization

### Video Resources
- [Gymnasium Tutorial Series](https://www.youtube.com/results?search_query=gymnasium+reinforcement+learning+tutorial) - Community tutorials
- [Stable-Baselines3 Tutorials](https://www.youtube.com/results?search_query=stable+baselines3+tutorial) - Practical examples

## 🏋️ Exercises

1. **Solve Pendulum-v1 with PPO** - Achieve reward >-200 (closer to 0 is better)
2. **Create custom 2D navigation environment** - Agent navigates to goal, avoids obstacles
3. **Implement action squashing (tanh)** - Map unbounded actions to [-1, 1]
4. **Solve BipedalWalker-v3** - Challenging continuous control task
5. **Build multi-agent environment** - Multiple agents in shared space

## 🔧 Troubleshooting Tips

### Common Issues

**1. Actions out of bounds**
- **Symptom:** `ValueError: actions are out of bounds`
- **Solution:** Clip actions: `action = np.clip(action, env.action_space.low, env.action_space.high)`
- **Solution:** Use tanh squashing in policy network output
- **Check:** Are you scaling actions correctly from network output to env range?

**2. Gaussian policy has zero/negative std**
- **Symptom:** NaN losses or sampling errors
- **Solution:** Clamp log_std: `log_std = log_std.clamp(-20, 2)`
- **Solution:** Use softplus for std: `std = F.softplus(raw_std) + 1e-5`
- **Alternative:** Use fixed std initially, then learn it

**3. Poor performance on continuous tasks**
- **Check:** Are you normalizing observations? Use `VecNormalize` wrapper
- **Solution:** Tune entropy coefficient for exploration
- **Solution:** Try different algorithms: PPO → SAC → TD3
- **Check:** Is action space properly scaled? Some envs expect [-1,1], others different ranges

**4. Custom environment not working**
- **Check:** Did you call `super().reset(seed=seed)` in reset method?
- **Check:** Are observation/action spaces defined correctly?
- **Solution:** Test with `env.check_env()` from stable-baselines3
- **Check:** Are you returning correct tuple format? `(obs, info)` for reset, `(obs, reward, terminated, truncated, info)` for step

**5. Slow training**
- **Solution:** Use vectorized environments: `SubprocVecEnv` or `DummyVecEnv`
- **Solution:** Reduce network size for simple tasks
- **Check:** Are you using GPU? `.to(device)` for networks and tensors

### Debugging Checklist

```python
# Verify environment
import gymnasium as gym
env = gym.make('Pendulum-v1')
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Action low: {env.action_space.low}")
print(f"Action high: {env.action_space.high}")

# Test random policy
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Obs: {obs}, Reward: {reward:.2f}")
    if terminated or truncated:
        obs, info = env.reset()

# Check Gaussian policy output
mean, std = policy(torch.FloatTensor(obs))
print(f"Mean: {mean}, Std: {std}")
assert (std > 0).all(), "Std must be positive!"
```

## ✅ Self-Check

Before moving to Lesson 10, you should be able to:

- [ ] Explain the Gymnasium API (`reset`, `step`, `terminated`, `truncated`)
- [ ] Understand continuous vs discrete action spaces
- [ ] Implement Gaussian policies for continuous control
- [ ] Handle action normalization and scaling
- [ ] Create custom Gymnasium environments from scratch
- [ ] Use environment wrappers for preprocessing
- [ ] Solve Pendulum-v1 with PPO (reward > -200)
- [ ] Debug common continuous control issues

## 🚀 Next Steps

Now that you understand continuous control, you're ready for [Lesson 10: Game State Representation and Feature Engineering](lesson_10_game_states.md), where you'll learn:
- Designing state representations for games
- Processing visual observations with CNNs
- Feature engineering for complex environments
- Reward shaping techniques

**Optional challenge:** Before moving on, try to:
- Implement SAC (Soft Actor-Critic) for continuous control
- Solve MuJoCo Hopper or HalfCheetah environments
- Create a custom continuous control environment (e.g., drone navigation)
- Compare PPO, SAC, and TD3 on the same task

**Connection to Rocket League:** Rocket League uses continuous actions (throttle, steering, pitch, yaw, roll). The Gaussian policy skills you learned here are essential for Lesson 13!

---

**Duration:** 4-5 hours | **Next:** [Lesson 10 →](lesson_10_game_states.md)
