# Lesson 11: Advanced RL Concepts

**Duration:** 6-8 hours

**Prerequisites:** Lessons 1-10

## 🎯 Learning Objectives

1. Master reward shaping techniques
2. Implement curriculum learning
3. Use parallel environments for faster training
4. Understand self-play for competitive games
5. Apply transfer learning in RL
6. Use population-based training
7. Implement intrinsic motivation

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

So far you have learned **core RL algorithms** (Q-learning, DQN, PPO) and how to apply
them to Gym-style problems. For complex games like Rocket League or large-scale
environments, we often need *extra tools* to make learning practical. This lesson covers
four such tools:

- **Reward shaping** – making rewards denser without changing what “optimal” means
- **Curriculum learning** – starting easy and gradually increasing difficulty
- **Parallel environments** – collecting experience much faster
- **Self-play & intrinsic motivation** – creating your own opponents and rewards

These are not new algorithms; they are **scaffolding** around your algorithm that can
make the difference between “never learns” and “works well in practice”.

### Reward Shaping (Advanced View)

In Lesson 10 you saw how adding extra reward terms can guide behavior, but can also be
dangerous: the agent may optimize the shaping term and ignore the real goal.

**Potential-based reward shaping** is a principled way to add shaping while
*provably preserving* the optimal policy.

We define a potential function \(\Phi(s)\) on states (think: “how close am I to the
goal?”). Then we modify the reward as

```text
F(s, s') = γ Φ(s') - Φ(s)
r' = r + F(s, s')
```

Key property (Ng et al., 1999): if \(\Phi\) is any real-valued function on states, then
adding this shaping term **does not change which policies are optimal**. It only alters
the *learning dynamics* by providing denser feedback.

Intuition:

- If you move to a state with higher potential (e.g., closer to scoring), you get a
  positive shaping reward.
- If you move away, you get a negative shaping reward.
- Because the shaping term is a discounted difference of a potential, it forms a
  “telescoping sum” along trajectories and cancels out when comparing full returns.

For Rocket League, \(\Phi(s)\) could be something like **minus the distance from the
ball to the opponent’s goal**, possibly with extra terms for ball height or control.

### Curriculum Learning

Training a strong agent **from scratch on the full game** is often extremely hard:
the state space is huge, rewards are sparse, and exploration is difficult.

Curriculum learning addresses this by training on a sequence of tasks of increasing
difficulty:

1. Design a set of tasks \(T_1, T_2, ..., T_n\) from easy to hard.
2. Train the agent on \(T_1\) until it reaches a performance threshold.
3. Move to \(T_2\), possibly mixing in some \(T_1\) to prevent forgetting.
4. Continue until the agent can handle the full task.

**Rocket League example curriculum:**

1. **Hit a stationary ball** into an empty goal from short range.
2. **Hit a slow-moving ball** rolling towards/away from you.
3. **Practice aerial touches** in simplified setups.
4. **Play full 1v1 games** with opponents.

Good curricula:

- Decompose skills (driving straight, turning, hitting, recovering, aerials).
- Start with tasks where *random behavior sometimes gets reward*, so learning can begin.
- Progress when success is consistently high (e.g., 80%+), not on a fixed episode count.

### Parallel Environments

Single-env training is often **too slow** for large problems: you spend most of your
time waiting for the environment to step. Running **many environments in parallel**
solves this by collecting experiences simultaneously.

Conceptually:

```python
envs = [make_env() for _ in range(num_envs)]
states = [env.reset() for env in envs]
# Each training iteration:
#   - choose actions for all envs
#   - step all envs
#   - accumulate a big batch of transitions
```

Benefits:

- Much **faster data collection** for CPU-bound environments
- More **diverse experiences** in each batch (different random seeds, states)
- Smoother learning curves (each update averages over many trajectories)

This is exactly what libraries like Stable-Baselines3’s `SubprocVecEnv` do under the
hood: spawn multiple processes, each running its own environment instance.

### Self-Play

In competitive games, fixed scripted opponents quickly become too easy or exploitable.
**Self-play** lets the agent continuously generate challenging opponents:

- At each training stage the agent plays against **copies of itself** or a **pool of
  past versions**.
- As the agent improves, the opponents it faces also become stronger.
- This naturally creates a **curriculum of increasingly difficult opponents**.

However, self-play introduces challenges:

- The opponent is always changing, making the environment **non-stationary**.
- The agent can **forget** how to beat older strategies if training always focuses on
  the latest opponent.
- Large systems (like AlphaStar) use **league training**: a pool of different agent
  roles and skill levels to maintain diversity.

Practical tips:

- Save snapshots of your agent periodically and sample opponents from this pool.
- Evaluate against a **fixed benchmark** (e.g., built-in bots) to measure real progress.

### Intrinsic Motivation (Curiosity and Exploration Bonuses)

In many games external rewards are rare (sparse). Intrinsic motivation gives the agent
an **internal reward signal** for “interesting” states, encouraging exploration.

Two common forms:

**1. Curiosity-based rewards**

Train a model that tries to **predict what will happen** given the current state and
action. If the prediction is bad, that means the outcome was surprising, so we reward it:

```text
r_intrinsic = || f(s') - f̂(s') ||²
```

Here `f̂(s')` is the forward model’s prediction and `f(s')` is some learned feature
representation of the true next state. High error → high intrinsic reward.

Effect:

- The agent seeks out transitions where its world model is wrong.
- Over time, as the model improves, the bonus for familiar states shrinks.

**2. Count-based / novelty bonuses**

If you can track how often you visit states (or state clusters), you can reward rare
states more:

```text
r_intrinsic(s) = 1 / √(count(s))
```

- Early visits to a state give a large bonus.
- As `count(s)` grows, the bonus fades, pushing the agent to explore new regions.

In practice you rarely use intrinsic reward **alone**; you mix it with the task reward:

```text
r_total = r_extrinsic + β · r_intrinsic
```

with a small coefficient `β` (e.g., 0.01). Tuning `β` is important: too high and the
agent just wanders around; too low and it ignores curiosity.

Intrinsic motivation is especially useful in environments like large open worlds,
exploration-heavy games, or Rocket League training maps where goals are far apart.

## 💻 Practical Implementation

```python
import torch
import numpy as np
from multiprocessing import Process, Pipe

# Parallel environments
class ParallelEnv:
    def __init__(self, env_fn, num_envs):
        self.num_envs = num_envs
        self.envs = [env_fn() for _ in range(num_envs)]

    def reset(self):
        return np.array([env.reset()[0] for env in self.envs])

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        states = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] or r[3] for r in results])
        return states, rewards, dones

# Curriculum learning
class CurriculumTrainer:
    def __init__(self, tasks, threshold=0.8):
        self.tasks = tasks  # List of (task, difficulty)
        self.current_task = 0
        self.threshold = threshold

    def get_current_task(self):
        return self.tasks[self.current_task]

    def update(self, success_rate):
        if success_rate > self.threshold and self.current_task < len(self.tasks) - 1:
            self.current_task += 1
            print(f"Advancing to task {self.current_task}")

# Intrinsic curiosity
class CuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Forward model: predict s' from s, a
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def compute_intrinsic_reward(self, state, action, next_state):
        state_action = torch.cat([state, action], dim=-1)
        predicted_next = self.forward_model(state_action)
        error = torch.mean((predicted_next - next_state) ** 2, dim=-1)
        return error  # High error = surprising = high reward

# Self-play
class SelfPlayTrainer:
    def __init__(self, agent_fn):
        self.current_agent = agent_fn()
        self.opponent_pool = []

    def add_opponent(self, agent):
        self.opponent_pool.append(agent)

    def sample_opponent(self):
        if len(self.opponent_pool) == 0:
            return self.current_agent
        return np.random.choice(self.opponent_pool)

    def should_save_opponent(self, elo_gain):
        # Save if agent improved significantly
        return elo_gain > 50
```

## 📚 Key References

### Papers
- **Ng et al. (1999)** - "Policy Invariance Under Reward Shaping" - [PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
- **Burda et al. (2018)** - "Curiosity-driven Exploration by Self-supervised Prediction" - [arXiv](https://arxiv.org/abs/1705.05363)
- **Pathak et al. (2017)** - "Curiosity-driven Exploration" - [arXiv](https://arxiv.org/abs/1705.05363)
- **Vinyals et al. (2019)** - "Grandmaster level in StarCraft II using multi-agent RL (AlphaStar)" - [Nature](https://www.nature.com/articles/s41586-019-1724-z)
- **Silver et al. (2017)** - "Mastering Chess and Shogi by Self-Play (AlphaZero)" - [arXiv](https://arxiv.org/abs/1712.01815)
- **Bengio et al. (2009)** - "Curriculum Learning" - [ICML](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)

### Tutorials & Blogs
- [Stable-Baselines3: Vectorized Environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) - Parallel training guide
- [Curriculum Learning Survey](https://arxiv.org/abs/2010.13166) - Comprehensive overview
- [OpenAI: Curiosity-Driven Learning](https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/) - Blog post on ICM
- [Lilian Weng: Exploration Strategies](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/) - Count-based, curiosity, and more
- [Self-Play in Deep RL](https://blog.openai.com/competitive-self-play/) - OpenAI blog

### Code Examples
- [Stable-Baselines3: SubprocVecEnv](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv) - Parallel environments
- [CleanRL: PPO with Curiosity](https://github.com/vwxyzjn/cleanrl) - ICM implementation
- [RLlib: Self-Play](https://docs.ray.io/en/latest/rllib/rllib-env.html#self-play) - Multi-agent self-play

### Video Resources
- [DeepMind AlphaStar Explained](https://www.youtube.com/results?search_query=alphastar+deepmind) - Self-play and league training
- [Curiosity-Driven Learning Explained](https://www.youtube.com/results?search_query=curiosity+driven+reinforcement+learning) - ICM tutorials

## 🏋️ Exercises

1. **Implement potential-based reward shaping** - Ensure policy invariance
2. **Create curriculum for CartPole variants** - Progressively harder pole lengths/masses
3. **Use SubprocVecEnv for parallel training** - Speed up PPO with 8+ environments
4. **Implement self-play for tic-tac-toe** - Agent plays against itself
5. **Add curiosity module to DQN** - Intrinsic motivation for exploration

## 🔧 Troubleshooting Tips

### Common Issues

**1. Parallel environments not faster**
- **Check:** Are you using `SubprocVecEnv` (multiprocessing) or `DummyVecEnv` (single process)?
- **Solution:** Use `SubprocVecEnv` for CPU-bound envs, `DummyVecEnv` for GPU-bound
- **Check:** Is environment step time >> network forward time? Parallelization helps most when env is slow
- **Overhead:** Too many processes (>16) can cause overhead; find sweet spot

**2. Curriculum learning not progressing**
- **Symptom:** Agent stuck on early tasks, never advances
- **Solution:** Lower success threshold (e.g., 0.8 → 0.6)
- **Solution:** Add manual progression after N episodes if stuck
- **Check:** Are early tasks too hard? Start even simpler

**3. Self-play leads to forgetting**
- **Symptom:** Agent beats old versions but loses to even older ones
- **Solution:** Maintain diverse opponent pool (save checkpoints at different skill levels)
- **Solution:** Use league training (AlphaStar style) with multiple branches
- **Check:** Are you updating opponent too frequently? Slow down updates

**4. Curiosity module causes instability**
- **Symptom:** Agent explores forever, never exploits
- **Solution:** Reduce intrinsic reward coefficient (start with 0.01)
- **Solution:** Anneal curiosity bonus over time
- **Check:** Is forward model overfitting? Use dropout or smaller network

**5. Reward shaping breaks optimal policy**
- **Symptom:** Agent optimizes shaped reward, ignores true objective
- **Solution:** Use potential-based shaping: `F(s,s') = γΦ(s') - Φ(s)` (provably policy-invariant)
- **Check:** Is shaped reward magnitude >> true reward? Balance coefficients
- **Example:** Agent circles goal for "distance to goal" reward instead of entering

### Debugging Checklist

```python
# Verify parallel environments
from stable_baselines3.common.vec_env import SubprocVecEnv
import time

def make_env():
    return lambda: gym.make('CartPole-v1')

# Time single vs parallel
single_env = gym.make('CartPole-v1')
vec_env = SubprocVecEnv([make_env() for _ in range(8)])

start = time.time()
for _ in range(1000):
    single_env.step(single_env.action_space.sample())
print(f"Single env: {time.time() - start:.2f}s")

start = time.time()
for _ in range(1000):
    vec_env.step([vec_env.action_space.sample() for _ in range(8)])
print(f"Parallel env (8x): {time.time() - start:.2f}s")

# Monitor curriculum progression
print(f"Current task: {curriculum.current_task}")
print(f"Success rate: {success_rate:.2%}")
print(f"Threshold: {curriculum.threshold:.2%}")

# Check curiosity rewards
intrinsic_reward = curiosity_module.compute_intrinsic_reward(state, action, next_state)
print(f"Extrinsic: {reward:.2f}, Intrinsic: {intrinsic_reward:.2f}")
```

## ✅ Self-Check

Before moving to Lesson 12, you should be able to:

- [ ] Implement potential-based reward shaping correctly
- [ ] Design curriculum learning for progressive task difficulty
- [ ] Use parallel environments to speed up training
- [ ] Understand self-play dynamics and opponent pool management
- [ ] Implement intrinsic motivation (curiosity or count-based)
- [ ] Debug common issues with advanced RL techniques
- [ ] Know when to use each technique (curriculum, self-play, curiosity)
- [ ] Understand trade-offs (e.g., curiosity can delay exploitation)

## 🚀 Next Steps

Now that you understand advanced RL techniques, you're ready for [Lesson 12: Building a Complete 2D/3D Game Agent](lesson_12_complete_agent.md), where you'll learn:
- Designing complete RL pipelines
- Training and debugging complex agents
- Hyperparameter tuning strategies
- Evaluation and deployment

**Optional challenge:** Before moving on, try to:
- Implement AlphaZero-style self-play for a simple game
- Use curiosity to solve sparse-reward environments (e.g., Montezuma's Revenge)
- Design a curriculum for a complex task (e.g., BipedalWalker)
- Compare training speed with 1, 4, 8, 16 parallel environments

**Connection to Rocket League:** In Lesson 13, you'll use:
- **Curriculum learning** to progress from hitting stationary balls to aerials
- **Self-play** to train competitive agents
- **Parallel environments** to speed up training (8-32 game instances)
- **Reward shaping** to guide learning of complex mechanics

---

**Duration:** 6-8 hours | **Next:** [Lesson 12 →](lesson_12_complete_agent.md)
