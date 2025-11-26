# Lesson 13: Scaling to Rocket League - Building Complex Game AI

**Duration:** 8-12 hours

**Prerequisites:** Lessons 1-12

## 🎯 Learning Objectives

1. Understand Rocket League as an RL problem
2. Set up RLGym environment
3. Design state and action spaces for car soccer
4. Implement reward functions for complex behaviors
5. Train agents with self-play and opponent bots
6. Scale training with parallel environments
7. Deploy and compete in RLBot tournaments

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

In this capstone lesson you apply everything from the course to a **real, complex,
multi-agent game**: Rocket League. Compared to classic benchmarks like CartPole or
Atari, Rocket League presents several additional challenges that strongly motivate the
advanced techniques from Lessons 10–12.

### Rocket League RL Challenges

Rocket League is a rich but extremely challenging environment for reinforcement
learning. Several properties of the game make it much harder than classic benchmarks
like CartPole or Atari.

First, the game has a **high-dimensional state**. To play well, your agent must reason
about full 3D car and ball physics: positions and velocities in three dimensions,
rotations (pitch, yaw, roll), boost amounts, team scores, and other contextual
information such as kickoff positions or remaining time. All of this information has
to be compressed into a vector that a neural network can process while still
preserving what is relevant for good decision-making.

Second, the game combines **continuous and discrete actions**. Controls like throttle,
steering, pitch, yaw, and roll are naturally continuous in the range `[-1, 1]`, while
actions such as jump, boost, and handbrake are binary on/off switches. This mixture
makes simple discrete-action algorithms (like plain DQN) awkward to apply and strongly
favors **policy-gradient methods with continuous action spaces** such as PPO or SAC.

Third, Rocket League has **long horizons**. A match can last up to five minutes, and
actions taken early (like grabbing boost or rotating to a safe defensive position) may
only pay off much later when a scoring opportunity or save appears. This makes
**credit assignment** difficult: it is hard for the algorithm to know which earlier
decisions deserve credit for later rewards.

Fourth, the game involves **multi-agent interactions**. In 1v1, 2v2, or 3v3 modes your
environment includes both teammates and opponents whose behavior may be scripted,
stochastic, or also learning. This means the dynamics of the environment can change as
policies change, and coordination between agents (for example, not both chasing the
ball) becomes an important part of the problem.

Fifth, Rocket League features **complex physics and mechanics** such as ball bounces,
wall rides, aerials, demolitions, and recovery maneuvers after landing or being bumped.
To master the game, an agent must discover and exploit these mechanics, which greatly
increases the diversity of possible trajectories.

Finally, the game provides **sparse primary rewards**: goals are rare events compared
with the huge number of timesteps in a match. Without additional shaping, an agent may
play thousands of steps before ever seeing a non-zero reward signal. Because of all
these factors, **naively applying a standard RL algorithm** often leads to agents that
drive in circles, chase the ball aimlessly, or learn brittle tricks that fail against
slightly different opponents. The rest of this lesson shows how to structure
observations, rewards, and training so that learning becomes feasible.

### RLGym Environment

Libraries like **RLGym** wrap Rocket League into a Gym-style interface. A typical
observation dictionary might look like:

```python
{
    'player': [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
               pitch, yaw, roll, boost_amount, ...],
    'ball': [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z],
    'teammates': [...],
    'opponents': [...]
}
```

Conceptually, the observation can be divided into several parts. The **player** section
encodes your own car's position and orientation in the arena, its linear velocity and
rotation, and resources such as current boost amount. The **ball** section encodes the
ball's position and velocity; these are often normalized by field dimensions and
maximum speeds (as discussed in Lesson 10) so that all features lie in comparable
ranges. The **teammates** and **opponents** sections provide similar information for
other cars on the field.

In practice, most implementations **flatten and normalize** this information into a
single feature vector (as in the `StateEncoder` later in this lesson). Good feature
design makes it easier for the policy network to learn meaningful behavior.

The **action space** is typically 8 controls:

- Throttle: `[-1, 1]`
- Steer: `[-1, 1]`
- Pitch: `[-1, 1]`
- Yaw: `[-1, 1]`
- Roll: `[-1, 1]`
- Jump: `{0, 1}`
- Boost: `{0, 1}`
- Handbrake: `{0, 1}`

This is a high-dimensional, partly discrete, partly continuous control problem, which
is exactly the setting where **policy-gradient methods with continuous actions** (PPO,
SAC, TD3) shine.

### Reward Function Design

Because goals are rare, you almost always need **shaped rewards** to give the agent
useful feedback at intermediate steps. A common pattern is a **multi-component reward**
that mixes several terms, each targeting a different aspect of good play:

```python
reward = (
    1000 * goal_scored
    + 100 * ball_touch
    + 10 * velocity_toward_ball
    + 5 * velocity_toward_goal
    + 1 * boost_pickup
    - 50 * demo_opponent
)
```

Each term has an intuitive meaning. The `goal_scored` component is a large, sparse
reward tied directly to the main objective of the game: putting the ball in the
opponent's net. The `ball_touch` term provides a smaller but more frequent reward that
encourages basic competence such as reliably reaching and touching the ball. The
`velocity_toward_ball` and `velocity_toward_goal` terms reward situations where the
ball or your car are moving in a useful direction, even before an actual shot occurs;
this helps the agent learn to push the ball upfield and create pressure.

The `boost_pickup` term encourages the agent to collect boost pads rather than driving
around empty, which is crucial for maintaining speed and enabling aerial plays. The
`demo_opponent` term can be either positive or negative depending on your training
goals; in this example it is penalized to discourage reckless demolitions that might
hurt overall positioning.

The exact coefficients multiplying these terms are **task-dependent**, and you should
expect to iterate on them. After each change, it is important to watch replays to make
sure the agent is not exploiting a shaping term in an unintended way (as discussed in
Lessons 10 and 11).

To manage difficulty, you can implement **curriculum rewards** that change over time.
In an early training stage you might reward **any ball touch** and ignore goals
entirely, so the agent first learns to reliably reach the ball. In a second stage you
can start rewarding touches that move the ball towards the opponent's goal, teaching
the agent to push in the correct direction. Later, you add large rewards for actually
scoring goals and gradually reduce the basic touch reward so that the agent focuses on
converting pressure into points. In advanced stages you can introduce terms for team
coordination or defensive skills. This mirrors curriculum learning: start with simple
skills and gradually align the reward with full-game objectives.

### Training Strategies

Given the complexity and cost of each environment step, training strategy matters a
lot. Three ideas are especially important.

**1. Self-Play**

In self-play, you train your agent against **copies or past versions of itself** rather
than only against fixed scripted bots. As the current agent improves, the opponents it
faces naturally become stronger as well, creating an automatic curriculum of
difficulty. This setup reduces reliance on hand-crafted bots and can uncover novel
tactics that emerge purely from competition between learning agents.

However, as you saw in Lesson 11, self-play also carries risks such as catastrophic
forgetting or cycling between strategies. To mitigate these, you typically maintain an
**opponent pool** containing snapshots of your agent from different training stages and
regularly evaluate against **fixed baselines** such as built-in bots or frozen older
agents. This gives you a stable measure of progress even when both sides are learning.

**2. Bot and Data Diversity**

Pure self-play can lead to narrow meta-strategies that only work against copies of your
own agent. To broaden behavior, you can mix in a variety of opponents and data
sources. For example, you might alternate games against built-in **Rookie** and
**All-Star** bots, against **past agent checkpoints** from different points in
training, and optionally pretrain the policy using **human replay data** via behavior
cloning before fine-tuning with RL. This diversity makes it harder for the agent to
overfit to a single opponent style and encourages more robust strategies.

**3. Parallel and Distributed Training**

Rocket League environments are relatively expensive to step, especially when running
the full game. To train in a reasonable time, you typically run **8–32 parallel
instances** (or more on clusters). Each environment runs in its own process or even on
different machines, while a central learner gathers trajectories, performs gradient
updates, and broadcasts updated parameters back to the workers.

This architecture is conceptually the same as the **parallel environments** in
Lesson 11, just at a larger scale suitable for a complex 3D game. By collecting data
in parallel you can reach tens of millions of timesteps in days instead of weeks.

Together, good **state encoding**, **reward shaping**, **curriculum**, **self-play**, and
**parallel training** turn Rocket League from an impossible RL problem into a
challenging but manageable capstone project.

## 💻 Practical Implementation

### Setup

```bash
pip install rlgym rlgym-tools rocket-learn
```

### 1. Basic RLGym Environment

```python
import rlgym
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.terminal_conditions import TimeoutCondition
from rlgym.utils.state_setters import DefaultState

# Create environment
env = rlgym.make(
    game_speed=100,  # Speed up for training
    spawn_opponents=True,
    team_size=1,  # 1v1
    obs_builder=AdvancedObs(),
    reward_fn=CombinedReward(
        (VelocityReward(), 0.1),
        (EventReward(goal=1.0, touch=0.1), 1.0)
    ),
    terminal_conditions=[TimeoutCondition(300)],  # 5 minutes
    state_setter=DefaultState()
)

# Check spaces
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape}")
```

### 2. Custom Reward Function

```python
from rlgym.utils.reward_functions import RewardFunction
import numpy as np

class RocketLeagueReward(RewardFunction):
    def __init__(self):
        self.last_ball_touch = {}

    def reset(self, initial_state):
        self.last_ball_touch = {}

    def get_reward(self, player, state, previous_action):
        reward = 0.0

        # Ball touch reward
        if player.ball_touched:
            reward += 1.0
            self.last_ball_touch[player.car_id] = state.ball.position

        # Goal reward
        if state.last_touch == player.car_id:
            if state.blue_score > 0:  # Scored
                reward += 100.0

        # Velocity toward ball
        ball_dir = state.ball.position - player.car_data.position
        ball_dir = ball_dir / (np.linalg.norm(ball_dir) + 1e-8)
        velocity_toward = np.dot(player.car_data.linear_velocity, ball_dir)
        reward += 0.01 * velocity_toward

        # Boost management
        if player.boost_amount > 50:
            reward += 0.001

        return reward

# Use custom reward
env = rlgym.make(reward_fn=RocketLeagueReward())
```

### 3. State Encoder

```python
class StateEncoder:
    """Encode RLGym state for neural network input."""

    def encode(self, obs):
        """
        Convert RLGym observation to normalized feature vector.
        """
        # Normalize positions (divide by field dimensions)
        positions = obs[:6] / np.array([4096, 5120, 2044, 4096, 5120, 2044])

        # Normalize velocities
        velocities = obs[6:12] / 2300  # Max car velocity

        # Rotation (already in [-1, 1])
        rotation = obs[12:18]

        # Boost (already in [0, 1])
        boost = obs[18:19]

        # Ball state
        ball_state = obs[19:31]
        ball_positions = ball_state[:3] / np.array([4096, 5120, 2044])
        ball_velocities = ball_state[3:] / 6000  # Max ball velocity

        return np.concatenate([
            positions, velocities, rotation, boost,
            ball_positions, ball_velocities
        ])
```

### 4. Training with PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import rlgym

def make_env():
    def _init():
        return rlgym.make(
            game_speed=100,
            reward_fn=RocketLeagueReward(),
            spawn_opponents=True
        )
    return _init

# Create parallel environments
num_envs = 8
env = SubprocVecEnv([make_env() for _ in range(num_envs)])

# Train with PPO
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

# Train
model.learn(total_timesteps=10_000_000)
model.save("rocket_league_agent")
```

### 5. Self-Play Training

```python
class SelfPlayCallback:
    """Periodically update opponent with current policy."""

    def __init__(self, update_freq=10000):
        self.update_freq = update_freq
        self.opponent_policy = None

    def __call__(self, locals_, globals_):
        if locals_['self'].num_timesteps % self.update_freq == 0:
            # Save current policy as opponent
            self.opponent_policy = locals_['self'].policy.state_dict()
            print("Updated opponent policy")
        return True

# Use in training
model.learn(
    total_timesteps=10_000_000,
    callback=SelfPlayCallback()
)
```

### 6. Evaluation and Deployment

```python
def evaluate_agent(model, num_games=10):
    """Evaluate agent against bots."""
    env = rlgym.make(
        game_speed=1,  # Real-time
        spawn_opponents=True
    )

    wins = 0
    total_goals = 0

    for game in range(num_games):
        obs = env.reset()
        done = False
        episode_goals = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Track goals
            if info.get('blue_score', 0) > episode_goals:
                episode_goals = info['blue_score']
                total_goals += 1

        # Check win
        if info['blue_score'] > info['orange_score']:
            wins += 1

    print(f"Win rate: {wins/num_games:.2%}")
    print(f"Avg goals per game: {total_goals/num_games:.2f}")

# Evaluate
model = PPO.load("rocket_league_agent")
evaluate_agent(model)
```

## 📚 Key References

### RLGym Resources
- [RLGym Documentation](https://rlgym.github.io/) - Complete API reference
- [RLGym GitHub](https://github.com/lucas-emery/rocket-league-gym) - Source code and examples
- [Rocket-Learn](https://github.com/Rolv-Arild/rocket-learn) - Distributed training framework
- [RLGym-Tools](https://github.com/AechPro/rlgym-tools) - Additional utilities

### Community & Competitions
- [RLGym Discord](https://discord.gg/rlgym) - Active community, get help here!
- [RLBot](https://rlbot.org/) - Bot framework and competitions
- [RLBot Discord](https://discord.gg/rlbot) - Bot development community
- [Rocket League Replays](https://ballchasing.com/) - Download replays for imitation learning

### Tutorials & Blogs
- [RLGym Getting Started Guide](https://rlgym.github.io/getting_started.html) - Official tutorial
- [Necto Bot Case Study](https://github.com/Rolv-Arild/Necto) - SSL-level bot, open source
- [RLBot Python Tutorial](https://www.youtube.com/watch?v=DwBB7871rX0) - Beginner-friendly video

### Papers (Complex Game AI)
- **Berner et al. (2019)** - "Dota 2 with Large Scale Deep RL" - [arXiv](https://arxiv.org/abs/1912.06680)
  - Similar complexity, multi-agent, long time horizons
- **Jaderberg et al. (2019)** - "Human-level performance in 3D multiplayer games (Quake III)" - [arXiv](https://arxiv.org/abs/1807.01281)
  - FPS game, continuous control, self-play
- **Vinyals et al. (2019)** - "Grandmaster level in StarCraft II (AlphaStar)" - [Nature](https://www.nature.com/articles/s41586-019-1724-z)
  - Multi-agent, curriculum, league training

### Video Resources
- [RLGym YouTube Channel](https://www.youtube.com/@RLGym) - Official tutorials and showcases
- [RLBot Setup Guide](https://www.youtube.com/watch?v=YJ69QZ-EX7k) - Installation walkthrough
- [Rocket League Bot Showcase](https://www.youtube.com/results?search_query=rlbot+showcase) - See what's possible!

## 🏋️ Exercises

### Exercise 1: Basic Agent (Easy)
Train an agent that can hit the ball consistently in 1v0 mode.

**Success criteria:**
- Touches ball >5 times per game
- Stays on correct side of field

### Exercise 2: Aerial Mechanics (Medium)
Implement reward shaping for aerial hits.

```python
class AerialReward(RewardFunction):
    def get_reward(self, player, state, previous_action):
        # Reward being in air near ball
        # Reward aerial ball touches
        # Your implementation
        pass
```

### Exercise 3: Team Play (Hard)
Train 2v2 agents with passing behaviors.

In this exercise you focus on **team play** rather than just solo ball chasing. One key
challenge is **credit assignment**: when a goal is scored after a series of passes and
rotations, it can be difficult to determine which player and which earlier actions
deserve credit. Another challenge is **coordination**: the agents must learn not to
both chase the ball at the same time and instead take complementary roles such as
attacker and defender. Finally, you may need some form of **communication or shared
information**, for example via a centralized critic that sees the state of both cars
and can encourage coordinated strategies.

### Exercise 4: Advanced Mechanics (Hard)
Implement curriculum learning for a progression of mechanics: start with consistent
**ground hits**, then move to **wall hits**, then **simple aerials**, and finally
**advanced aerials** such as air roll shots. The idea is to design separate training
tasks or reward stages for each mechanic so that the agent can master them one by one
instead of trying to learn everything at once.

### Exercise 5: RLBot Competition (Very Hard)
**Final Project:** Enter RLBot tournament

**Requirements:**
1. Train agent to beat All-Star bot consistently
2. Implement real-time inference (<16ms per action)
3. Handle match state management
4. Create match recording/replay system
5. Write strategy documentation

**Bonus:**
- Multi-modal learning (vision + state)
- Opponent modeling
- In-game adaptation

## 🔧 Troubleshooting Tips

### Common Issues

**1. Training is unstable**

If training loss or episode returns fluctuate wildly, the first step is often to
**normalize all inputs** so that state features lie in comparable ranges. You can also
apply **reward normalization** so that very large or very small rewards do not dominate
learning. Reducing the **learning rate** and increasing the **batch size** both make
updates smaller and more stable, which is particularly important in a noisy, complex
environment like Rocket League.

**2. Agent doesn't learn basic mechanics**

If your agent never learns to hit the ball or drive coherently, the reward function may
be too complicated or too focused on goals. Try **simplifying the reward function** so
that you strongly reward easy-to-achieve behaviors such as ball touches. Introducing a
clear **curriculum**, where you first train on simple tasks and only later add more
complex objectives, can also help. Additionally, double-check **action space scaling**
and **observation encoding** to ensure that actions map correctly to in-game controls
and that important information (like ball position) is not missing or incorrectly
scaled.

**3. Slow training**

Rocket League is computationally heavy, so training may feel extremely slow. You can
often speed things up by increasing the environment's **`game_speed`** parameter (for
example, to 100x) so that more in-game time passes per real second. Running **more
parallel environments** allows you to collect experience faster, and using **GPU
acceleration** speeds up neural network computations. If performance is still an
issue, consider **reducing observation complexity** so that each step is cheaper to
process.

**4. Agent gets stuck in local optimum**

Sometimes an agent discovers a mediocre but easy strategy (for example, camping in
front of the ball) and then fails to improve further. In this case you may need more
**exploration**, such as increasing the entropy coefficient in PPO to encourage more
varied actions. Combining this with **self-play against diverse opponents** and
**curiosity-driven exploration** can push the agent out of local optima. If things go
badly wrong, you can also **reset to earlier checkpoints** that performed better and
try new hyperparameters or reward designs from there.

### Performance Optimization

```python
# Multi-GPU training
from stable_baselines3.common.vec_env import VecNormalize

env = SubprocVecEnv([make_env() for _ in range(32)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Use larger networks for complex tasks
policy_kwargs = dict(
    net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])]
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    device="cuda"
)
```

## ✅ Self-Check

Before considering yourself ready for RL game AI:

- [ ] Can train agent that beats rookie bots consistently
- [ ] Understand reward shaping trade-offs for Rocket League
- [ ] Can implement custom observation builders and reward functions
- [ ] Know how to use parallel training (8+ environments)
- [ ] Can debug training issues systematically (instability, no learning)
- [ ] Understand self-play dynamics and opponent pool management
- [ ] Can deploy trained models in real-time (<16ms inference)
- [ ] Have trained an agent that can hit the ball consistently

## 🎓 Capstone Project Rubric

### Minimum Requirements (Pass)
- [ ] Agent can hit stationary ball >50% of attempts
- [ ] Training pipeline with logging (TensorBoard)
- [ ] Custom reward function implemented
- [ ] Parallel training (at least 4 environments)
- [ ] Evaluation script with metrics (goals, touches, win rate)
- [ ] Documentation (README with setup and training instructions)

### Target Performance (Good)
- [ ] Agent beats Rookie bots >70% of games
- [ ] Can hit moving ball consistently
- [ ] Self-play training implemented
- [ ] Curriculum learning (stationary → moving ball)
- [ ] Hyperparameter tuning documented
- [ ] Demo video showing learned behaviors

### Stretch Goals (Excellent)
- [ ] Agent beats All-Star bots >50% of games
- [ ] Can perform basic aerials
- [ ] Multi-agent training (2v2 or 3v3)
- [ ] Opponent modeling or adaptation
- [ ] Real-time deployment in RLBot framework
- [ ] Detailed analysis of learned strategies

### Expected Training Time & Resources
- **Hardware:** GPU recommended (RTX 3060 or better), 16GB+ RAM
- **Training time:**
  - Basic ball hitting: 2-4 hours (1-2M steps)
  - Beat Rookie bots: 8-12 hours (5-10M steps)
  - Beat All-Star bots: 24-48 hours (20-50M steps)
- **Parallel environments:** 8-16 for good speed/stability trade-off
- **Game speed:** 100x for training (real-time for evaluation)

## 🚀 Next Steps

**Congratulations!** You've completed the RL workshop from basics to Rocket League!

### Continue Learning:

**1. Advanced RL Topics:**
- **Multi-agent RL:** QMIX, MADDPG, MAPPO for team coordination
- **Model-based RL:** World Models, MuZero, Dreamer for sample efficiency
- **Meta-learning:** Learn to adapt quickly to new opponents/tasks
- **Sim-to-real transfer:** Apply RL to real robots (if you have access)

**2. Related Fields:**
- **Imitation learning:** Behavior cloning from human replays, inverse RL
- **Offline RL:** Learn from datasets (ballchasing.com replays!)
- **Hierarchical RL:** Skills and options for complex behaviors
- **Transformer-based RL:** Decision Transformer, Trajectory Transformer

**3. Competitions & Community:**
- [RLBot Tournaments](https://rlbot.org/) - Compete with your bot!
- [MineRL Competition](https://minerl.io/) - Minecraft RL challenge
- [AI Crowd Challenges](https://www.aicrowd.com/) - Various RL competitions
- [Kaggle RL Competitions](https://www.kaggle.com/competitions?search=reinforcement) - Data science + RL

**4. Research & Open Source:**
- Read recent papers from **NeurIPS, ICML, ICLR** (RL track)
- Implement algorithms from [Spinning Up](https://spinningup.openai.com/)
- Contribute to [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [CleanRL](https://github.com/vwxyzjn/cleanrl), or [RLGym](https://github.com/lucas-emery/rocket-league-gym)
- Share your Rocket League bot on RLBot Discord!

### Your RL Journey

You now have the skills to:
- ✅ Implement RL algorithms from scratch (Q-learning, DQN, PPO)
- ✅ Train agents for complex environments (Atari, MuJoCo, Rocket League)
- ✅ Debug and optimize training (hyperparameters, reward shaping, curriculum)
- ✅ Build complete RL pipelines (preprocessing, logging, evaluation, deployment)
- ✅ Apply RL to real-world problems (game AI, robotics, control)

### Recommended Next Projects

1. **Improve your Rocket League bot:**
   - Add aerial mechanics with curriculum learning
   - Implement team play (2v2, 3v3)
   - Use imitation learning from pro replays
   - Enter an RLBot tournament!

2. **Apply RL to a new domain:**
   - Train a trading bot (stock market, crypto)
   - Build a recommendation system with RL
   - Solve a robotics task (if you have access)
   - Create a custom game and train an agent

3. **Contribute to research:**
   - Reproduce a recent paper
   - Improve an existing algorithm
   - Publish your findings (blog, paper, GitHub)

**Keep experimenting, keep learning, and most importantly—have fun building AI agents!**

---

**Estimated completion time:** 8-12 hours (basic agent) to 40+ hours (competitive bot)

**Workshop complete!** 🎉 Return to [README](README.md) for more resources.

**Share your results:** Post your trained Rocket League bot on the RLGym/RLBot Discord and show what you've learned!
