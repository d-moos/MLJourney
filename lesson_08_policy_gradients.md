# Lesson 8: Policy Gradient Methods and PPO

**Duration:** 6-8 hours

**Prerequisites:** Lessons 1-7

## 🎯 Learning Objectives

1. Understand policy gradient theorem
2. Implement REINFORCE algorithm
3. Learn actor-critic methods (A2C)
4. Master Proximal Policy Optimization (PPO)
5. Handle continuous action spaces
6. Understand advantage estimation
7. Debug policy gradient training

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

### Policy Gradient Methods

In value-based methods (like Q-learning / DQN) we first learn a value function and then
*derive* a policy from it (e.g., greedy w.r.t. Q). **Policy gradient methods flip this
around:** we **parameterize the policy directly** and optimize it.

- We write the policy as \(\pi_\theta(a \mid s)\): a neural network with parameters
  \(\theta\) that outputs a probability distribution over actions.
- Our goal is to adjust \(\theta\) so that the policy chooses actions that yield high
  long-term reward.

Formally we want to maximize the **expected return**

```text
J(θ) = 𝔼_π [G_t] = 𝔼_π [ Σ_t γ^t r_t ]
```

The **policy gradient theorem** tells us that a valid gradient of this objective is

```text
∇_θ J(θ) = 𝔼_π [ ∇_θ log π_θ(a | s) · Q^π(s, a) ]
```

Intuition:

- `log π_θ(a | s)` tells us *how much more* or *less* likely the policy makes action `a`.
- If an action leads to **higher-than-average return** in that state, we want to
  **increase** its probability.
- If it leads to poor returns, we want to **decrease** its probability.

So the update

```text
θ ← θ + α · ∇_θ log π_θ(a | s) · ("how good" this action was)
```

is literally "nudge the policy towards actions that worked well".

### REINFORCE Algorithm (Monte Carlo Policy Gradient)

REINFORCE is the simplest policy gradient method. It uses **complete episodes** to
estimate how good actions were.

Algorithm sketch:

```text
1. Run the policy π_θ in the environment to generate an episode
   (s₀, a₀, r₁, s₁, a₁, r₂, ..., s_T)
2. For each time step t in the episode:
       G_t = r_{t+1} + γ r_{t+2} + ... + γ^{T-t-1} r_T   # return from t
       θ ← θ + α · ∇_θ log π_θ(a_t | s_t) · G_t
```

A useful mental model:

- If the episode return from step `t` onward is **high**, we increase the probability
  of the actions that were taken.
- If the return is **low**, we decrease their probability.

Pros:

- Conceptually simple
- Works with **discrete or continuous** actions

Cons:

- High variance: each update depends on a full episode return
- Cannot update until the episode finishes

This motivates adding a **baseline** to reduce variance.

### Actor–Critic Methods

REINFORCE uses the raw return `G_t` as the "how good" signal, which is noisy.
Actor–critic methods introduce a separate **critic** to estimate value functions and
produce lower-variance signals.

- **Actor:** policy network \(\pi_\theta(a \mid s)\)
- **Critic:** value network \(V_\phi(s)\) estimating expected return from state `s`

We use TD learning for the critic and use its **TD error** as a better signal for the
actor:

```text
# Critic update (TD(0))
δ = r + γ V_φ(s') − V_φ(s)      # TD error
φ ← φ − α_v ∇_φ (δ)²

# Actor update
θ ← θ + α_π ∇_θ log π_θ(a | s) · δ
```

If the outcome is **better than expected** (δ > 0) we increase the probability of `a`;
if it is **worse than expected** (δ < 0) we decrease it.

### Advantage Function

Sometimes even using the critic's value estimates directly is not ideal. We often use
the **advantage function**:

```text
A(s, a) = Q(s, a) − V(s)
```

Interpretation:

- `V(s)` is the average value of being in state `s`
- `Q(s, a)` is the value when we specifically take action `a`
- `A(s, a)` measures **how much better or worse** action `a` is relative to the
  average action in that state.

Advantages of using `A(s, a)` in the policy gradient:

- Centered around zero 	→ updates are more balanced
- Lower variance than using raw returns
- If all actions are equally good, advantages are ~0 and the policy barely changes

In practice we approximate `A(s, a)` using combinations of TD errors and returns (e.g.,
Generalized Advantage Estimation, which you can explore as an extension).

### PPO (Proximal Policy Optimization)

Vanilla actor–critic still suffers from a key problem: if we take **too large a gradient
step**, the policy may change so much that performance collapses (we move into a bad
region of parameter space).

PPO is a **trust-region-inspired** method that tries to keep each policy update
"close" to the old policy, while still improving performance.

- Let `π_θ_old` be the policy before an update and `π_θ` the new policy.
- Define the **probability ratio**

  ```text
  r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
  ```

- If `r_t(θ) > 1`, the new policy makes `a_t` more likely; if `< 1`, less likely.

The PPO **clipped objective** is

```text
L^CLIP(θ) = 𝔼_t [ min( r_t(θ) · A_t,
                       clip(r_t(θ), 1 − ε, 1 + ε) · A_t ) ]
```

Effect:

- When the policy change is **small** (ratio near 1), the objective behaves like a
  standard policy gradient.
- When the change would make `r_t` go outside `[1−ε, 1+ε]`, we **clip** it so that the
  improvement is limited. This prevents enormous destructive updates.

Empirically, PPO is:

- Much more **stable** than vanilla policy gradient
- Easier to implement than TRPO (no second‑order optimization)
- A very strong default for many continuous control and game tasks

## 💻 Practical Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.policy_head(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value_head(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr)

        self.buffer = []

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def store_transition(self, state, action, reward, log_prob):
        self.buffer.append((state, action, reward, log_prob))

    def update(self):
        # Unpack buffer
        states = torch.FloatTensor([t[0] for t in self.buffer]).to(self.device)
        actions = torch.LongTensor([t[1] for t in self.buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([t[3] for t in self.buffer]).to(self.device)

        # Compute returns
        returns = []
        G = 0
        for t in reversed(self.buffer):
            G = t[2] + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy
            probs = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute advantages
            values = self.value_net(states).squeeze()
            advantages = returns - values.detach()

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Combined loss
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer = []
        return loss.item()

# Train PPO
env = gym.make('CartPole-v1')
agent = PPOAgent(state_dim=4, action_dim=2)

episodes = 500
rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, log_prob)
        total_reward += reward
        state = next_state

        if done:
            break

    loss = agent.update()
    rewards.append(total_reward)

    if ep % 50 == 0:
        print(f"Episode {ep}, Avg Reward: {np.mean(rewards[-50:]):.2f}")
```

## 📚 Key References

### Papers
- **Sutton et al. (1999)** - "Policy Gradient Methods" - [PDF](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- **Schulman et al. (2017)** - "Proximal Policy Optimization" - [arXiv](https://arxiv.org/abs/1707.06347)
- **Mnih et al. (2016)** - "Asynchronous Methods for Deep RL (A3C)" - [arXiv](https://arxiv.org/abs/1602.01783)
- **Schulman et al. (2015)** - "Trust Region Policy Optimization (TRPO)" - [arXiv](https://arxiv.org/abs/1502.05477)

### Tutorials & Blogs
- [Spinning Up: Policy Gradients](https://spinningup.openai.com/en/latest/algorithms/vpg.html) - Clear derivations and code
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - Implementation details
- [Lilian Weng: Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) - Comprehensive overview
- [Hugging Face Deep RL Course - Unit 8: PPO](https://huggingface.co/learn/deep-rl-course/unit8/introduction) - Interactive tutorial
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Critical for real implementations

### Video Lectures
- [David Silver RL Course - Lecture 7: Policy Gradients](https://www.youtube.com/watch?v=KHZVXao4qXs) - Theory foundation
- [DeepMind x UCL RL Lecture Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) - Full course

### Code Examples
- [CleanRL PPO Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) - Single-file, well-documented
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) - Production-ready
- [PyTorch Actor-Critic Example](https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py) - Minimal example

## 🏋️ Exercises

1. **Implement REINFORCE from scratch** - Monte Carlo policy gradient
2. **Add baseline (value function) to REINFORCE** - Reduce variance
3. **Implement A2C** - Synchronous version of A3C with advantage estimation
4. **Extend PPO to continuous actions** - Use Gaussian policy (see Lesson 9)
5. **Solve LunarLander-v2 with PPO** - Achieve average reward >200

## 🔧 Troubleshooting Tips

### Common Issues

**1. Policy collapses to deterministic too quickly**
- **Symptom:** Agent stops exploring, gets stuck in local optimum
- **Solution:** Increase entropy coefficient (e.g., `ent_coef=0.01` or higher)
- **Solution:** Reduce learning rate to slow down policy updates
- **Check:** Are you normalizing advantages? This helps stability

**2. Training is unstable / performance oscillates**
- **Cause:** Policy updates too large, violating trust region
- **Solution:** Reduce learning rate (try 3e-4 → 1e-4)
- **Solution:** Increase number of epochs per update (`k_epochs`)
- **Solution:** Reduce clip range (`eps_clip` from 0.2 → 0.1)
- **Check:** Are you using GAE (Generalized Advantage Estimation)? Helps reduce variance

**3. Value network not learning / high value loss**
- **Solution:** Use separate learning rates for actor and critic
- **Solution:** Increase value loss coefficient (e.g., `vf_coef=0.5` → `1.0`)
- **Solution:** Normalize returns: `(returns - returns.mean()) / (returns.std() + 1e-8)`

**4. No learning / rewards stay flat**
- **Check:** Is your policy network outputting valid probabilities? (Use softmax for discrete)
- **Check:** Are gradients flowing? Print `policy.parameters()` gradients
- **Solution:** Increase batch size or number of steps per update
- **Solution:** Check reward scaling - very large/small rewards can cause issues

**5. "RuntimeError: one of the variables needed for gradient computation has been modified"**
- **Cause:** Reusing old log probabilities after policy update
- **Solution:** Detach old values: `old_log_probs = log_probs.detach()`
- **Solution:** Store log probs before updating, don't recompute from updated policy

### Debugging Checklist

```python
# Add these to your training loop
print(f"Policy loss: {policy_loss.item():.4f}")
print(f"Value loss: {value_loss.item():.4f}")
print(f"Entropy: {dist.entropy().mean().item():.4f}")
print(f"Approx KL: {((ratio - 1) - ratio.log()).mean().item():.4f}")
print(f"Clip fraction: {((ratio > 1+eps_clip) | (ratio < 1-eps_clip)).float().mean():.2%}")

# Visualize policy distribution
import matplotlib.pyplot as plt
plt.bar(range(action_dim), probs.detach().cpu().numpy())
plt.title('Action Probabilities')
plt.show()
```

### PPO-Specific Tips

**Key hyperparameters that matter:**
- `learning_rate`: 3e-4 is a good default, reduce if unstable
- `n_steps`: 2048 for simple tasks, 4096+ for complex
- `batch_size`: 64 for simple, 256+ for complex
- `n_epochs`: 4-10 (how many times to reuse each batch)
- `clip_range`: 0.2 is standard, reduce to 0.1 if unstable
- `ent_coef`: 0.01 for exploration, reduce over time if needed
- `gamma`: 0.99 for most tasks, 0.995-0.999 for long horizons

## ✅ Self-Check

Before moving to Lesson 9, you should be able to:

- [ ] Explain the policy gradient theorem and its intuition
- [ ] Implement REINFORCE algorithm from scratch
- [ ] Understand why baselines reduce variance without adding bias
- [ ] Explain the difference between on-policy and off-policy methods
- [ ] Describe how PPO's clipped objective prevents destructive updates
- [ ] Train a PPO agent that solves CartPole (avg reward >195)
- [ ] Debug common policy gradient issues (collapse, instability)
- [ ] Understand when to use policy gradients vs DQN
- [ ] Explain advantage functions and their role in variance reduction

## 🚀 Next Steps

Now that you understand policy gradient methods, you're ready for [Lesson 9: Gymnasium and Continuous Control](lesson_09_gym_continuous.md), where you'll learn:
- Mastering the Gymnasium API
- Continuous action spaces with Gaussian policies
- Solving classic control tasks (Pendulum, BipedalWalker)
- Creating custom environments

**Optional challenge:** Before moving on, try to:
- Implement Generalized Advantage Estimation (GAE)
- Add entropy bonus and tune the coefficient
- Compare REINFORCE, A2C, and PPO on the same task
- Solve LunarLander-v2 with PPO (target: >200 reward)

**Connection to Rocket League:** PPO is the algorithm you'll use in Lesson 13 for training Rocket League agents. The continuous control skills from Lesson 9 are essential for handling car controls (throttle, steering, pitch, yaw, roll).

---

**Duration:** 6-8 hours | **Next:** [Lesson 9 →](lesson_09_gym_continuous.md)
