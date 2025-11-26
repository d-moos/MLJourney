# Lesson 10: Game State Representation and Feature Engineering

**Duration:** 5-6 hours

**Prerequisites:** Lessons 1-9

## 🎯 Learning Objectives

1. Design state representations for games
2. Extract features from game observations
3. Process visual observations with CNNs
4. Handle partial observability (frame stacking)
5. Normalize and preprocess inputs
6. Engineer reward functions
7. Design observation spaces for complex games

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

This lesson is about two tightly connected design problems in RL for games:

1. **What information do we give the agent?** (state / observation)
2. **What behavior do we reward?** (reward function)

Good choices here often matter **more** than small changes to the RL algorithm.

### State Representation Principles

We want the observation `s_t` to be a *useful summary* of the game at time `t`. A good
state representation is often described as **Markov**, **compact**, **normalized**, and
**informative**.

Being **Markov** means that, given `s_t`, the **future** does not depend on the
**past**. In other words, `s_t` contains all the information you need to make good
decisions and to predict what will happen next. For Rocket League, a single cropped
visual frame is *not* Markov: from one still image you cannot tell the ball's velocity
or where it was a moment ago. If you add explicit velocity features or stack several
recent frames together, the state becomes **closer** to having the Markov property.

The representation should also be **compact**, meaning it should not include
unnecessary details that do not affect the task. The exact grass texture or the
audience in the stands rarely matters for control, so including those pixels just
increases input size and makes learning slower without adding useful signal.

It is helpful for features to be **normalized**, with different quantities mapped onto
roughly similar numeric ranges. For example, you might divide positions by the field
size so they lie in `[-1, 1]`, and divide velocities by some maximum speed. Neural
networks generally train more reliably when inputs are on comparable scales, because
gradients are better behaved and one large-magnitude feature does not dominate the
others.

Finally, the state must be **informative** about the task itself. For a Rocket
League-like game, this usually means focusing on things like the relative positions of
cars and ball, who currently has possession, where the goals are, and how much boost
you have. If key information is missing, the agent may never learn good behavior,
because no policy can succeed without seeing what actually matters. Bad state designs
can therefore make learning nearly impossible (when they omit critical information) or
just very slow (when they include huge amounts of irrelevant input).

### Feature Types

There are three broad approaches to constructing state features; in practice you often
**combine** them.

The first approach uses **raw pixels (vision)** as input. Here the observation is one
or more images rendered from the game. You typically process these images with a
convolutional neural network (CNN), as in DQN for Atari. To keep things efficient, the
images are often converted to **grayscale** and resized to a smaller resolution (for
example, 84×84 pixels) so that the input tensor is not too large. When using pixels it
is also common to apply **frame stacking** (discussed below) so that the network can
infer motion from a short history of frames rather than seeing only a single snapshot.
The big advantage of raw pixels is that they are very general and require little
game-specific knowledge; they also match how humans visually perceive the game. The
downsides are that pixel inputs are high-dimensional, make networks slower to train,
and can be harder to debug than more structured, human-designed features.

The second approach is to build a **structured state** using hand-crafted features. In
this case the observation is a vector of numbers that directly describes important game
objects. For Rocket League, you might include the ball's position and velocity, the
car's position, velocity, and rotation, whether the car is on the ground or in the air,
and quantities like boost amount or jump/flip status. These feature vectors are usually
**much lower dimensional** than raw pixels and much easier to normalize. They also
make it easier for you, as the designer, to reason about what information the network
is seeing and to debug strange behaviors.

The third approach is to add **derived features** on top of raw pixels or structured
state. These are higher-level quantities computed from more basic inputs, such as the
distance from the car to the ball or goal, the angle between the car's forward
direction and the ball, or a flag indicating whether the ball is currently moving
towards or away from the opponent's goal. Derived features often encode **what we care
about** for the task more directly than raw positions or velocities. However, too much
hand-designed structure can bias the agent towards your intuitions and potentially hide
useful signals that a more flexible representation might discover on its own, so there
is always a trade-off.

### Frame Stacking and Temporal Information

Games are dynamic. From a **single** frame, you cannot tell whether the ball is moving
towards or away from you. Two common solutions:

1. **Include velocities** explicitly as features (in structured representations).
2. **Stack recent frames** when using pixels:

   ```
   state_t = [frame_t, frame_{t-1}, frame_{t-2}, frame_{t-3}]
   ```

Frame stacking lets the CNN infer motion (like optical flow) from differences between
frames. This is exactly what the original DQN paper did for Atari.

For very long-term dependencies (e.g., multi-step strategies), you might explore RNNs or
transformers later, but frame stacking is a strong baseline.

### Reward Shaping

The **reward function** defines what you want the agent to care about. In principle, we
could give reward only when the game is won or lost:

```text
r = 1  if goal_scored else 0
```

This is a **sparse reward**. It is perfectly aligned with the game objective, but very
hard to learn from: the agent may play thousands of steps before ever scoring a goal.

To speed things up we often add **dense shaping rewards** that give feedback at every
step, for example:

```text
r = 0.1 * velocity_toward_opponent_goal
    + 0.5 * touch_ball
    + 1.0 * goal_scored
```

This encourages the agent to move the ball in the right direction, touch it often, and
still strongly rewards actual goals.

However, reward shaping is **dangerous** if done carelessly. Because the agent only
optimizes what you explicitly reward, it may learn to exploit the shaping terms instead
of actually winning the game. For example, if you reward only "ball velocity towards
the opponent goal", the agent might happily smash the ball in ways that create short,
fast bursts of movement towards the goal even if this leaves the ball in terrible
positions right afterward. From the point of view of the shaping reward this behavior
looks great, but the team may rarely score.

A more principled approach is **potential-based shaping** (Ng et al., 1999), where you
define a potential function \(\Phi(s)\) (e.g., negative distance to the opponent goal)
and add a shaping term

```text
F(s, s') = γ Φ(s') - Φ(s)
```

This preserves the optimal policy of the original MDP while still providing denser
feedback.

In practice, for Rocket League-like games you will usually start with a simple **base
reward** that gives positive points for scoring a goal and negative points when a goal
is scored against you. On top of this you can add a **small number** of carefully
chosen shaping terms, such as a small reward for moving the ball towards the opponent's
goal, for making controlled touches, or for keeping the ball in play. During
experiments it is important to watch replays or visualize trajectories to check whether
the agent is exploiting the reward in unintended ways, and to adjust or remove shaping
terms that encourage obviously undesirable behavior.

## 💻 Practical Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import cv2

class StateProcessor:
    """Process game observations."""

    def __init__(self, img_size=(84, 84), stack_size=4):
        self.img_size = img_size
        self.stack_size = stack_size
        self.frame_buffer = []

    def process_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize
        resized = cv2.resize(gray, self.img_size)
        # Normalize
        normalized = resized / 255.0
        return normalized

    def get_state(self, frame):
        processed = self.process_frame(frame)
        self.frame_buffer.append(processed)

        if len(self.frame_buffer) > self.stack_size:
            self.frame_buffer.pop(0)

        # Pad if needed
        while len(self.frame_buffer) < self.stack_size:
            self.frame_buffer.append(processed)

        return np.stack(self.frame_buffer, axis=0)

# CNN for visual observations
class ConvNetwork(nn.Module):
    def __init__(self, input_channels=4, num_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_out((input_channels, 84, 84))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        return self.fc(self.conv(x))

# Feature engineering for Rocket League
class RocketLeagueFeatures:
    """Extract features from Rocket League state."""

    def extract_features(self, obs):
        """
        obs: dict with keys like 'ball', 'car', 'boost_pads', etc.
        """
        features = []

        # Ball features
        ball_pos = obs['ball']['position']  # [x, y, z]
        ball_vel = obs['ball']['velocity']
        features.extend(ball_pos)
        features.extend(ball_vel)

        # Car features
        car_pos = obs['car']['position']
        car_vel = obs['car']['velocity']
        car_rotation = obs['car']['rotation']  # [pitch, yaw, roll]
        boost = obs['car']['boost']

        features.extend(car_pos)
        features.extend(car_vel)
        features.extend(car_rotation)
        features.append(boost / 100.0)  # Normalize

        # Derived features
        ball_to_car = np.array(ball_pos) - np.array(car_pos)
        distance_to_ball = np.linalg.norm(ball_to_car)
        features.append(distance_to_ball / 10000.0)  # Normalize

        return np.array(features, dtype=np.float32)

# Reward shaping example
class RewardShaper:
    def __init__(self):
        self.prev_ball_pos = None

    def compute_reward(self, obs, raw_reward):
        shaped_reward = raw_reward

        # Progress toward goal
        ball_pos = obs['ball']['position']
        goal_pos = [0, 5120, 0]  # Example goal position

        if self.prev_ball_pos is not None:
            prev_dist = np.linalg.norm(np.array(self.prev_ball_pos) - np.array(goal_pos))
            curr_dist = np.linalg.norm(np.array(ball_pos) - np.array(goal_pos))
            progress = prev_dist - curr_dist
            shaped_reward += 0.01 * progress

        self.prev_ball_pos = ball_pos
        return shaped_reward
```

## 📚 Key References

### Papers
- **Ng et al. (1999)** - "Policy Invariance Under Reward Shaping" - [PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
- **Mnih et al. (2015)** - "Human-level control through deep RL" - [Nature](https://www.nature.com/articles/nature14236) - CNN architecture for Atari

### Tutorials & Blogs
- [Lilian Weng: RL Overview](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) - Feature engineering section
- [Andrej Karpathy: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) - Processing visual observations
- [Stable-Baselines3: Preprocessing](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#wrappers) - Observation wrappers

### Code Examples
- [Gymnasium Atari Preprocessing](https://gymnasium.farama.org/environments/atari/complete_list/) - Frame stacking, grayscale, resizing
- [OpenCV Python Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) - Image processing basics
- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) - Building CNNs

### Rocket League Specific
- [RLGym Observation Builders](https://rlgym.github.io/api_doc/rlgym.utils.obs_builders.html) - Pre-built state encoders
- [RLGym Reward Functions](https://rlgym.github.io/api_doc/rlgym.utils.reward_functions.html) - Reward shaping examples

## 🏋️ Exercises

1. **Implement frame stacking wrapper** - Stack last 4 frames for temporal information
2. **Design features for Pong** - Ball position, velocity, paddle positions
3. **Create reward shaping for your custom env** - Use potential-based shaping
4. **Build CNN for Atari games** - Implement DQN architecture from Mnih et al.
5. **Extract Rocket League ball prediction features** - Future ball positions, velocities

## 🔧 Troubleshooting Tips

### Common Issues

**1. CNN not learning from images**
- **Check:** Are images normalized to [0, 1]? Divide by 255.0
- **Check:** Are you stacking frames? Single frame lacks temporal info
- **Solution:** Use grayscale (1 channel) instead of RGB (3 channels) to reduce complexity
- **Check:** Is input shape correct? PyTorch expects (batch, channels, height, width)

**2. State representation too large / slow**
- **Solution:** Reduce image size (84x84 is standard for Atari)
- **Solution:** Use grayscale instead of RGB
- **Solution:** Reduce frame stack size (4 is typical, but 2-3 may work)
- **Check:** Are you using GPU for CNN forward passes?

**3. Reward shaping causes unintended behavior**
- **Symptom:** Agent exploits shaped reward instead of solving task
- **Solution:** Use potential-based shaping: `F(s,s') = γΦ(s') - Φ(s)`
- **Solution:** Reduce shaping coefficient, increase sparse reward weight
- **Example:** Agent circles ball for "velocity toward ball" reward instead of scoring

**4. Features not informative enough**
- **Check:** Is state Markovian? Does it contain all info needed to act optimally?
- **Solution:** Add derived features (distances, angles, relative velocities)
- **Solution:** Include recent history (frame stacking or LSTM)
- **Check:** Are features normalized to similar scales?

**5. Memory errors with frame stacking**
- **Cause:** Storing too many high-res images in replay buffer
- **Solution:** Store frames as uint8 instead of float32
- **Solution:** Reduce buffer size or image resolution
- **Solution:** Use lazy frame stacking (store indices, stack on sampling)

### Debugging Checklist

```python
# Verify state processor
processor = StateProcessor()
frame = env.render()  # Get RGB frame
processed = processor.get_state(frame)
print(f"Original shape: {frame.shape}")
print(f"Processed shape: {processed.shape}")  # Should be (stack_size, H, W)
print(f"Value range: [{processed.min():.2f}, {processed.max():.2f}]")  # Should be [0, 1]

# Visualize processed frames
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(processed[i], cmap='gray')
    axes[i].set_title(f'Frame t-{3-i}')
plt.show()

# Check feature extraction
features = feature_extractor.extract_features(obs)
print(f"Feature vector shape: {features.shape}")
print(f"Feature stats: mean={features.mean():.2f}, std={features.std():.2f}")
```

## ✅ Self-Check

Before moving to Lesson 11, you should be able to:

- [ ] Design Markovian state representations for games
- [ ] Process visual observations (grayscale, resize, normalize)
- [ ] Implement frame stacking for temporal information
- [ ] Build CNNs for processing image observations
- [ ] Extract structured features from game state
- [ ] Apply potential-based reward shaping correctly
- [ ] Understand trade-offs between sparse and dense rewards
- [ ] Debug common state representation issues

## 🚀 Next Steps

Now that you understand state representation and feature engineering, you're ready for [Lesson 11: Advanced RL Concepts](lesson_11_advanced_rl.md), where you'll learn:
- Curriculum learning for complex tasks
- Parallel environments for faster training
- Self-play for competitive games
- Intrinsic motivation and exploration bonuses

**Optional challenge:** Before moving on, try to:
- Train DQN on Atari Pong using CNN and frame stacking
- Design custom features for a complex environment
- Implement and compare different reward shaping strategies
- Build an observation wrapper that normalizes and stacks frames

**Connection to Rocket League:** In Lesson 13, you'll use the `RocketLeagueFeatures` class you saw here to encode car and ball state. Reward shaping will be critical for learning complex behaviors like aerials!

---

**Duration:** 5-6 hours | **Next:** [Lesson 11 →](lesson_11_advanced_rl.md)
