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

Some key sources of difficulty:

- **High-dimensional state:** you must reason about full 3D car and ball physics:
  positions, velocities, rotations, boost amounts, team scores, etc.
- **Continuous and discrete actions mixed:** throttle, steering, pitch, yaw, roll are
  continuous; jump/boost/handbrake are binary. This favors algorithms like PPO or SAC.
- **Long horizons:** games last up to 5 minutes, so actions taken early may only pay
  off much later (credit assignment is hard).
- **Multi-agent interactions:** 1v1, 2v2, or 3v3 means your environment includes
  teammates and opponents that may also learn or behave unpredictably.
- **Complex physics and mechanics:** bounces, wall rides, aerials, demos, recoveries,
  kickoffs, and more.
- **Sparse primary rewards:** goals are rare events compared to the number of timesteps.

Because of this, **naively applying a standard RL algorithm** often leads to agents that
drive in circles, chase the ball aimlessly, or learn brittle tricks. The rest of this
lesson shows how to structure observations, rewards, and training to make learning
feasible.

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

Conceptually:

- The **player** part encodes your own cars position/orientation, linear velocity,
  rotation, and resources like boost.
- The **ball** part encodes position and velocity; you often normalize these by field
  dimensions and maximum speeds (Lesson 10).
- **Teammates** and **opponents** provide equivalent information for other cars.

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
useful feedback at intermediate steps. A common pattern is a **multi-component reward**:

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

Interpretation:

- `goal_scored`: large, sparse reward for the main objective.
- `ball_touch`: encourages basic competence (just touching the ball reliably).
- `velocity_toward_ball` and `velocity_toward_goal`: reward moving the ball in a useful
  direction, even before a shot.
- `boost_pickup`: encourages collecting resources rather than driving empty.
- `demo_opponent`: can be positive or negative depending on whether demos help your
  strategy; here it is penalized.

The exact coefficients are **task-dependent**; you should expect to iterate. Always
watch replays to ensure the agent is not exploiting a shaping term in a weird way
(Lesson 10 & 11).

To manage difficulty, you can implement **curriculum rewards** that change over time:

- **Stage 1:** reward **any ball touch**; ignore goals.
- **Stage 2:** reward touches that move the ball towards the opponent goal.
- **Stage 3:** add large rewards for scoring; reduce basic touch reward.
- **Stage 4:** introduce team coordination objectives or defensive skills.

This mirrors curriculum learning: start with simple skills and gradually align the
reward with full-game objectives.

### Training Strategies

Given the complexity and cost of each environment step, training strategy matters a
lot. Three ideas are especially important:

**1. Self-Play**

- Train your agent against **copies or past versions of itself**.
- As it improves, its opponents naturally become stronger, creating an automatic
  curriculum of difficulty.
- This reduces reliance on hand-crafted scripted bots and can uncover novel tactics.

You still need safeguards against forgetting (Lesson 11): maintain an opponent pool and
evaluate against fixed baselines (e.g., built-in bots or frozen earlier agents).

**2. Bot and Data Diversity**

Pure self-play can lead to narrow meta-strategies. To broaden behavior, mix in:

- Built-in **Rookie** and **All-Star** bots.
- **Past agent checkpoints** from different training stages.
- Optionally, **human replay data** for behavior cloning warm-starts.

This diversity makes it harder for the agent to overfit to a single opponent style.

**3. Parallel and Distributed Training**

Rocket League environments are relatively expensive to step. To train in a reasonable
time, you typically run **8–32 parallel instances** (or more on clusters):

- Each environment runs in its own process (or even on different machines).
- A central learner gathers trajectories, performs updates, and broadcasts new
  parameters.

This is conceptually the same as the **parallel environments** in Lesson 11, just at a
larger scale. It allows you to reach tens of millions of timesteps in days instead of
weeks.

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

**Challenges:**
- Credit assignment (who caused goal?)
- Coordination (don't both go for ball)
- Communication (centralized critic?)

### Exercise 4: Advanced Mechanics (Hard)
Implement curriculum learning for:
1. Ground hits
2. Wall hits
3. Simple aerials
4. Advanced aerials (air roll shots)

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
- Normalize all inputs
- Use reward normalization
- Reduce learning rate
- Increase batch size

**2. Agent doesn't learn basic mechanics**
- Simplify reward function
- Use curriculum learning
- Check action space scaling
- Verify observation encoding

**3. Slow training**
- Increase game_speed (100x)
- Use more parallel environments
- Enable GPU acceleration
- Reduce observation complexity

**4. Agent gets stuck in local optimum**
- Increase exploration (entropy coefficient)
- Use self-play with diverse opponents
- Implement curiosity-driven exploration
- Reset to earlier checkpoints

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
