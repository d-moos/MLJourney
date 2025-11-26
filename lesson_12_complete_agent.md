# Lesson 12: Building a Complete 2D/3D Game Agent

**Duration:** 8-10 hours

**Prerequisites:** Lessons 1-11

## 🎯 Learning Objectives

1. Design and implement a complete RL pipeline
2. Train agents for complex 2D/3D environments
3. Debug and optimize training
4. Use TensorBoard for experiment tracking
5. Implement checkpointing and model management
6. Evaluate agent performance systematically
7. Deploy trained models

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

This lesson is about turning everything you have learned so far into a **complete,
repeatable training pipeline**. Up to now you have trained individual algorithms on
single environments; a real project (like Rocket League) needs more structure.

### Complete RL Pipeline

You can think of an RL project as moving through six interconnected stages:

1. **Environment Setup**
   - Define **state and action spaces** (Lesson 10): what information the agent sees,
     and what controls it has.
   - Design the **reward function** (Lessons 10 & 11): what behavior you want to
     encourage, and how you will measure success.
   - Specify **termination conditions**: when an episode ends (goal scored, time limit,
     car destroyed, etc.).

2. **State Preprocessing**
   - **Normalization:** scale inputs to reasonable ranges (e.g., divide positions by
     field size, velocities by max speed). This stabilizes learning.
   - **Frame stacking:** for pixel or fast-moving tasks, stack recent observations so
     the agent can infer velocity.
   - **Feature extraction:** choose or learn compact state representations
     (e.g., CNNs for pixels, hand-crafted Rocket League features).

3. **Agent Architecture & Algorithm**
   - Choose a **model**: MLP, CNN, RNN, or combination, based on state type.
   - Pick an **algorithm**: DQN variants for discrete actions, PPO/SAC/TD3 for
     continuous control.
   - Set initial **hyperparameters** (see below): learning rates, batch sizes, etc.

4. **Training Loop**
   - **Collect data:** interact with the environment using the current policy (possibly
     with exploration noise or ε-greedy).
   - **Store transitions:** maintain replay buffers or trajectory batches.
   - **Update networks:** run gradient steps at regular intervals.
   - **Log metrics:** track rewards, losses, Q-values, policy entropy, etc., with
     TensorBoard/WandB.

5. **Evaluation**
   - Use **separate evaluation runs** with exploration disabled to measure true
     performance.
   - Track **metrics over time**: average return, success rate, episode length.
   - **Visualize behavior:** record videos or trajectories; many RL failures are only
     obvious when you watch the agent.

6. **Deployment**
   - Export trained models (e.g., PyTorch `state_dict`) and configuration.
   - Optimize for **inference**: smaller networks, quantization, batching.
   - Integrate into the **target system**: game client, simulation server, or robot.

The practical code later in this lesson implements exactly this loop in a reusable
`CompleteRLTrainer` class.

### Hyperparameter Tuning

Even with the right algorithm, **bad hyperparameters can completely kill learning**.
Hyperparameter tuning is the process of systematically searching this space.

Some of the most important hyperparameters control **how quickly and smoothly** your
agent learns.

The **learning rate** determines the size of each gradient step and is often in the
range `1e-5` to `1e-3` for deep RL. If the learning rate is **too high**, updates
become very noisy and the parameters can oscillate or even diverge, with losses
shooting to infinity or becoming `NaN`. If it is **too low**, learning becomes very
slow and the optimizer may get stuck in a poor local optimum because each step is too
small to escape.

The **batch size** (how many transitions you use per gradient step) is typically
between `32` and `256` for many algorithms. **Small batches** give more noisy gradient
estimates but allow more frequent updates for the same amount of data, which can help
exploration and responsiveness. **Large batches** produce smoother, more stable
gradients but require more memory and make each training iteration slower.

Your **network architecture** (depth and width) controls the model's capacity. If the
network is **too small**, it may underfit complex environments and fail to represent
useful value or policy functions. If it is **too large**, training becomes slower and
the model may overfit or become harder to stabilize, especially in RL where the data
distribution is non-stationary.

The **exploration parameters** (like ε in ε-greedy for DQN or the entropy bonus
coefficient in PPO) govern the trade-off between trying new actions and exploiting what
the agent already believes is best. With **too little exploration**, the agent may
quickly exploit a suboptimal strategy it discovered early and then get stuck there.
With **too much** exploration, the agent keeps taking random actions and may never
settle on a good policy.

Finally, the **discount factor** γ, usually in `[0.95, 0.999]`, controls how much the
agent values long-term rewards relative to immediate ones. A **lower γ** places more
weight on short-term rewards, which simplifies credit assignment but may cause the
agent to ignore strategies that only pay off far in the future. A **higher γ** makes
the agent care more about long-term outcomes, but it can also increase the variance of
returns and make training less stable.

There are several common **tuning strategies**. In a basic **grid search**, you choose
a small set of candidate values for each hyperparameter and try all combinations. This
is simple to implement but becomes very expensive as you add more parameters or more
candidate values. A **random search** instead samples hyperparameters from chosen
distributions; for the same number of trials this often explores the space more
effectively than a grid.

More advanced strategies include **population-based training (PBT)**, where you
maintain a population of agents, periodically copy weights from the best-performing
ones, and randomly mutate their hyperparameters. Over time this evolves both weights
and hyperparameters. **Bayesian or automated tuning** libraries (such as Optuna) try to
model the relationship between hyperparameters and performance so that each new trial
is chosen in a promising region of the space rather than at random.

Regardless of which strategy you use, some best practices always apply. You should log
**everything** (both the configuration and resulting metrics) so that you can later
analyze what worked and reproduce good runs. When debugging, try to change only a
**small number of hyperparameters at a time**, so you can attribute any improvement or
regression to a specific change. And whenever possible, **start from known-good
defaults** taken from stable baselines or published implementations, then adapt them to
your game instead of tuning from scratch.

### Training Stability

Deep RL is notorious for being unstable. It helps to recognize common failure modes
and know how to respond to them.

One frequent issue is **exploding or vanishing gradients**. When gradients explode,
you may see the loss become extremely large or `NaN`, and the network weights can blow
up to huge values. Vanishing gradients, on the other hand, make learning grind to a
halt because updates become effectively zero. To mitigate these problems you can apply
**gradient clipping** (for example, clipping the global norm to 0.5 or 1.0), use
careful initialization and input normalization so activations stay in reasonable
ranges, and prefer empirically stable architectures and optimizers (such as PPO with
Adam and appropriate learning rates).

Another problem is **reward instability**, where the average episode return jumps
wildly between runs or even within the same run, making it hard to compare different
experiments. A common fix is to apply **reward normalization or clipping** (for
example, clipping individual rewards to a range like `[-10, 10]`) and to design reward
terms so that they have roughly similar magnitudes across different types of events.

You may also encounter **catastrophic forgetting**, where the agent learns a new skill
but seems to forget older ones (for example, it learns to score but no longer
defends). This is especially common when the data distribution shifts over time. To
reduce forgetting, you can mix in **older experiences** in the replay buffer instead of
training only on the most recent transitions, and you can design curricula and
evaluation tasks that regularly exercise the full set of desired skills.

Finally, **non-stationarity** can arise when the environment or opponents change over
time, as often happens in self-play. Performance may improve for a while and then
suddenly collapse when the opponent changes. Techniques like **target networks** and
slowly-updated value estimates can make learning more robust. In self-play settings it
also helps to maintain a **diverse pool of opponents** rather than always training
against the very latest model.

Throughout training, regular **checkpoints** are essential: save models frequently so
you can roll back if a run collapses, compare different hyperparameter settings, and
analyze what went wrong without having to retrain from scratch.

## 💻 Practical Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time

class CompleteRLTrainer:
    """Complete training pipeline for RL agents."""

    def __init__(self, env_name, agent, config):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.agent = agent
        self.config = config

        # Logging
        self.writer = SummaryWriter(f'runs/{env_name}_{time.strftime("%Y%m%d-%H%M%S")}')

        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def train(self, num_episodes):
        """Main training loop."""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            for step in range(self.config['max_steps']):
                # Select action
                action = self.agent.get_action(state)

                # Environment step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)

                # Update agent
                if len(self.agent.buffer) >= self.config['batch_size']:
                    loss = self.agent.update()
                    self.log_training(episode, step, loss)

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if done:
                    break

            # Episode logging
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)

            # Periodic evaluation
            if episode % self.config['eval_freq'] == 0:
                eval_reward = self.evaluate()
                self.log_evaluation(episode, eval_reward)

            # Save checkpoint
            if episode % self.config['save_freq'] == 0:
                self.save_checkpoint(episode)

            # Log episode stats
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Length = {avg_length:.2f}")

    def evaluate(self, num_episodes=10):
        """Evaluate agent performance."""
        eval_rewards = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0

            for _ in range(self.config['max_steps']):
                action = self.agent.get_action(state, training=False)
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            eval_rewards.append(total_reward)

        return np.mean(eval_rewards)

    def log_training(self, episode, step, loss):
        """Log training metrics."""
        global_step = episode * self.config['max_steps'] + step
        self.writer.add_scalar('Training/Loss', loss, global_step)

    def log_evaluation(self, episode, eval_reward):
        """Log evaluation metrics."""
        self.writer.add_scalar('Evaluation/Reward', eval_reward, episode)

    def save_checkpoint(self, episode):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.agent.policy.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards)
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_{episode}.pth')

# Example: Train on LunarLander
config = {
    'max_steps': 1000,
    'batch_size': 64,
    'eval_freq': 50,
    'save_freq': 100,
}

# Assuming PPOAgent from Lesson 8
from lesson_08 import PPOAgent

agent = PPOAgent(state_dim=8, action_dim=4)
trainer = CompleteRLTrainer('LunarLander-v2', agent, config)
trainer.train(num_episodes=1000)
```

## 📚 Key References

### Libraries & Frameworks
- [Stable-Baselines3 Source Code](https://github.com/DLR-RM/stable-baselines3) - Production-ready RL library
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single-file implementations for learning
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html) - Scalable RL framework
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started) - Experiment tracking

### Tutorials & Blogs
- [Stable-Baselines3 Tutorial](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html) - Complete examples
- [Hyperparameter Tuning for RL](https://medium.com/@aureliantactics/hyperparameter-tuning-for-reinforcement-learning-1b5d1b5e6e35) - Practical guide
- [Optuna for RL](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html) - Automated tuning
- [WandB for RL](https://wandb.ai/site/solutions/reinforcement-learning) - Advanced experiment tracking

### Code Examples
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - Trained agents and hyperparameters
- [CleanRL Benchmark](https://github.com/vwxyzjn/cleanrl#benchmarks) - Performance comparisons
- [Stable-Baselines3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) - Additional algorithms

### Video Resources
- [Stable-Baselines3 Tutorial Series](https://www.youtube.com/results?search_query=stable+baselines3+tutorial) - Practical walkthroughs
- [TensorBoard for ML](https://www.youtube.com/results?search_query=tensorboard+tutorial) - Visualization tutorials

## 🏋️ Exercises

### Project: Complete Agent for Your Choice

Pick one environment and build a complete solution:

**2D Options:**
- **Atari Pong** - Target: Beat built-in AI consistently
- **Atari Breakout** - Target: Average score >400
- **CarRacing-v2** - Target: Average score >900

**3D Options:**
- **MuJoCo Humanoid** - Target: Walk forward consistently
- **PyBullet environments** - Free alternative to MuJoCo
- **Unity ML-Agents** - Custom 3D environments

**Requirements:**
1. **State preprocessing pipeline** - Normalization, frame stacking, etc.
2. **Reward shaping** (if needed) - Dense rewards for sparse tasks
3. **Full training with logging** - TensorBoard or WandB
4. **Hyperparameter tuning** - At least 3 different configurations
5. **Evaluation suite** - Systematic performance measurement
6. **Visualization of learned behaviors** - Render episodes, plot trajectories
7. **Documentation** - README with setup, training, evaluation instructions

**Deliverables:**
- Trained model achieving target performance
- Training curves and analysis (loss, reward, episode length)
- Demo video (at least 3 episodes)
- Code repository with clear structure
- Hyperparameter comparison table

## 🔧 Troubleshooting Tips

### Common Issues

**1. Training doesn't converge**
- **Check:** Are you logging enough metrics? (reward, loss, episode length, exploration)
- **Solution:** Try different algorithms (DQN → PPO → SAC)
- **Solution:** Tune learning rate (try 1e-3, 3e-4, 1e-4)
- **Check:** Is environment too hard? Start with simpler version

**2. Hyperparameter tuning is slow**
- **Solution:** Use Optuna for automated search
- **Solution:** Start with coarse grid search, then refine
- **Solution:** Use smaller networks and fewer steps for initial search
- **Tip:** Focus on: learning rate, batch size, network size, entropy coefficient

**3. Can't reproduce results**
- **Solution:** Set all random seeds: `env.seed()`, `torch.manual_seed()`, `np.random.seed()`
- **Solution:** Log hyperparameters with every run
- **Solution:** Save full config file with each checkpoint
- **Check:** Are you using deterministic algorithms? (may be slower)

**4. TensorBoard logs are messy**
- **Solution:** Use clear naming: `runs/{env_name}_{algorithm}_{timestamp}`
- **Solution:** Log to different scalars: `Training/Reward`, `Evaluation/Reward`
- **Solution:** Use `SummaryWriter.add_hparams()` to log hyperparameters
- **Tip:** Smooth curves in TensorBoard UI for clearer trends

**5. Model performance degrades after loading**
- **Check:** Are you loading optimizer state? May need to reset it
- **Check:** Are you setting `model.eval()` for evaluation?
- **Solution:** Save full checkpoint: model, optimizer, config, normalization stats
- **Check:** Are observation normalization stats saved/loaded?

### Debugging Checklist

```python
# Comprehensive logging
writer.add_scalar('Training/Reward', episode_reward, episode)
writer.add_scalar('Training/Loss', loss, global_step)
writer.add_scalar('Training/Episode_Length', episode_steps, episode)
writer.add_scalar('Training/Epsilon', epsilon, episode)  # For DQN
writer.add_scalar('Training/Entropy', entropy, episode)  # For PPO

# Log hyperparameters
writer.add_hparams(
    {'lr': lr, 'batch_size': batch_size, 'gamma': gamma},
    {'hparam/reward': final_reward}
)

# Checkpoint everything
checkpoint = {
    'episode': episode,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'episode_rewards': episode_rewards,
    'random_state': np.random.get_state(),
    'torch_rng_state': torch.get_rng_state()
}
torch.save(checkpoint, f'checkpoints/checkpoint_{episode}.pth')

# Evaluation mode
model.eval()
with torch.no_grad():
    for eval_episode in range(num_eval_episodes):
        # Evaluate without exploration
        pass
model.train()
```

## ✅ Self-Check

Before moving to Lesson 13, you should be able to:

- [ ] Design a complete RL pipeline from scratch
- [ ] Implement proper logging with TensorBoard
- [ ] Save and load model checkpoints correctly
- [ ] Tune hyperparameters systematically
- [ ] Evaluate agent performance with proper metrics
- [ ] Debug training issues (no learning, instability, overfitting)
- [ ] Visualize learned behaviors and training progress
- [ ] Document and reproduce experiments

## 🚀 Next Steps

Now that you can build complete RL agents, you're ready for [Lesson 13: Scaling to Rocket League](lesson_13_rocket_league.md), where you'll learn:
- Setting up RLGym for Rocket League
- Designing state/action spaces for car soccer
- Training with self-play and parallel environments
- Deploying agents in RLBot tournaments

**Optional challenge:** Before moving on, try to:
- Build a complete agent for LunarLander-v2 (target: >200 reward)
- Implement automated hyperparameter tuning with Optuna
- Create a training dashboard with WandB
- Train an Atari agent from pixels (DQN + CNN)

**Final preparation for Rocket League:**
- Review PPO implementation (Lesson 8) - you'll use this algorithm
- Review continuous control (Lesson 9) - Rocket League has continuous actions
- Review parallel environments (Lesson 11) - essential for fast training
- Review reward shaping (Lesson 10) - critical for complex behaviors

---

**Duration:** 8-10 hours | **Next:** [Lesson 13 →](lesson_13_rocket_league.md)
