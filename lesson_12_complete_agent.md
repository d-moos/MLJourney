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

### Complete RL Pipeline

```
1. Environment Setup
   - State/action space design
   - Reward function
   - Termination conditions

2. State Preprocessing
   - Normalization
   - Frame stacking
   - Feature extraction

3. Agent Architecture
   - Network design
   - Hyperparameters
   - Algorithm choice

4. Training Loop
   - Data collection
   - Model updates
   - Logging

5. Evaluation
   - Performance metrics
   - Visualization
   - Analysis

6. Deployment
   - Model export
   - Inference optimization
   - Real-time execution
```

### Hyperparameter Tuning

**Key hyperparameters:**
- Learning rate: 1e-5 to 1e-3
- Batch size: 32 to 256
- Network architecture: depth and width
- Exploration (epsilon, entropy bonus)
- Discount factor gamma: 0.95 to 0.999

**Tuning strategies:**
- Grid search
- Random search
- Population-based training
- Optuna (automated)

### Training Stability

**Common issues:**
- Exploding/vanishing gradients
- Reward instability
- Forgetting (catastrophic)
- Non-stationarity

**Solutions:**
- Gradient clipping
- Reward normalization
- Regular checkpoints
- Target networks

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
