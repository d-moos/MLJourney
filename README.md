# Reinforcement Learning Workshop: From Basics to Rocket League Bots

A comprehensive, hands-on curriculum for learning to build RL agents from the ground up.

## 📚 Course Overview

This workshop takes you from Python fundamentals through advanced reinforcement learning, culminating in building AI agents for complex 3D environments like Rocket League. Each lesson includes theory, practical code, exercises, and references to deepen your understanding.

**Prerequisites:** Basic programming knowledge (variables, loops, functions)

**Total Time Investment:** 60-80 hours over 8-12 weeks

**Tools You'll Need:**
- Python 3.8+
- PyTorch
- NumPy, Matplotlib
- OpenAI Gym / Gymnasium
- Jupyter Notebook (recommended)

## 🗺️ Curriculum Structure

### **Phase 1: Foundations (Weeks 1-2)**

| Lesson | Topic | Time | Key Skills |
|--------|-------|------|------------|
| [Lesson 1](lesson_01_python_numpy.md) | Python & NumPy Fundamentals | 4-5h | Arrays, vectorization, scientific computing |
| [Lesson 2](lesson_02_pytorch_intro.md) | Introduction to PyTorch | 4-6h | Tensors, autograd, GPU acceleration |
| [Lesson 3](lesson_03_supervised_learning.md) | Supervised Learning Fundamentals | 4-5h | Training loops, loss functions, optimization |
| [Lesson 4](lesson_04_classifier.md) | Building a Classifier | 5-6h | End-to-end neural network training |

### **Phase 2: Reinforcement Learning Basics (Weeks 3-5)**

| Lesson | Topic | Time | Key Skills |
|--------|-------|------|------------|
| [Lesson 5](lesson_05_rl_theory.md) | RL Theory & MDPs | 5-7h | Markov processes, value functions, Bellman equations |
| [Lesson 6](lesson_06_tabular_qlearning.md) | Tabular Q-Learning | 4-6h | Q-tables, exploration vs exploitation |
| [Lesson 7](lesson_07_dqn.md) | Deep Q-Networks (DQN) | 6-8h | Experience replay, target networks |
| [Lesson 8](lesson_08_policy_gradients.md) | Policy Gradients & PPO | 6-8h | REINFORCE, actor-critic, PPO algorithm |

### **Phase 3: Practical Game AI (Weeks 6-8)**

| Lesson | Topic | Time | Key Skills |
|--------|-------|------|------------|
| [Lesson 9](lesson_09_gym_continuous.md) | Gym & Continuous Control | 4-5h | Environment APIs, continuous action spaces |
| [Lesson 10](lesson_10_game_states.md) | Game State Representation | 5-6h | Feature engineering, state preprocessing |
| [Lesson 11](lesson_11_advanced_rl.md) | Advanced RL Concepts | 6-8h | Reward shaping, curriculum learning, parallelization |

### **Phase 4: Complex Environments (Weeks 9-12)**

| Lesson | Topic | Time | Key Skills |
|--------|-------|------|------------|
| [Lesson 12](lesson_12_complete_agent.md) | Complete 2D/3D Game Agent | 8-10h | Integration, debugging, hyperparameter tuning |
| [Lesson 13](lesson_13_rocket_league.md) | Scaling to Rocket League | 8-12h | RLGym, multi-agent RL, deployment |

## 🎯 Learning Path

```
Python/NumPy → PyTorch → Supervised Learning → Simple Classifier
                                                       ↓
RL Theory → Q-Learning → DQN → Policy Gradients → PPO
                                                       ↓
        Gym Environments → Feature Engineering → Advanced Techniques
                                                       ↓
                              2D/3D Game AI → Rocket League Bots
```

## 📖 How to Use This Guide

1. **Work sequentially** - Each lesson builds on previous concepts
2. **Code along** - Type out examples, don't just read them
3. **Do ALL exercises** - They're designed to test understanding
4. **Experiment** - Modify code, break things, see what happens
5. **Join communities** - Share progress on Discord/Reddit RL communities

## 🛠️ Setup Instructions

### Quick Start

```bash
# Create virtual environment
python -m venv rl_workshop
source rl_workshop/bin/activate  # On Windows: rl_workshop\Scripts\activate

# Install core dependencies
pip install torch torchvision numpy matplotlib jupyter
pip install gymnasium[classic-control]
pip install tensorboard

# For later lessons
pip install stable-baselines3
pip install rlgym  # For Rocket League (Lesson 13)
```

### Recommended Directory Structure

```
rl-workshop/
├── notebooks/          # Jupyter notebooks for experiments
├── code/              # Your implementations
│   ├── lesson_01/
│   ├── lesson_02/
│   └── ...
├── models/            # Saved model checkpoints
└── logs/              # Training logs and tensorboard data
```

## 📚 Core References

### Textbooks
- **Sutton & Barto** - "Reinforcement Learning: An Introduction" (2nd ed.) - [Free PDF](http://incompleteideas.net/book/the-book-2nd.html)
- **Goodfellow et al.** - "Deep Learning" - [Free online](https://www.deeplearningbook.org/)

### Online Courses (supplementary)
- **Spinning Up in Deep RL** (OpenAI) - https://spinningup.openai.com/
- **David Silver's RL Course** (DeepMind) - https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver

### Communities
- r/reinforcementlearning
- r/learnmachinelearning
- OpenAI Discord
- Hugging Face Discord

## 🎓 Assessment Milestones

Track your progress with these milestones:

- [ ] **Week 2:** Train a neural network to classify MNIST digits (>95% accuracy)
- [ ] **Week 4:** Implement Q-learning that solves FrozenLake-v1
- [ ] **Week 6:** Train a DQN agent that masters CartPole-v1
- [ ] **Week 8:** Implement PPO from scratch and solve LunarLander
- [ ] **Week 10:** Build an agent for a custom 2D game
- [ ] **Week 12:** Train a basic Rocket League bot that can hit the ball

## 💡 Tips for Success

1. **Be patient with training** - RL agents can take hours/days to train
2. **Visualize everything** - Plot rewards, losses, Q-values
3. **Start simple** - Get basic versions working before adding complexity
4. **Debug systematically** - Use print statements, tensorboard, unit tests
5. **Manage expectations** - Not every algorithm works on every problem

## 🚨 Common Pitfalls

- Skipping theory → not understanding why things fail
- Not normalizing inputs → unstable training
- Wrong learning rates → divergence or no learning
- Insufficient exploration → suboptimal policies
- Not saving checkpoints → losing trained models

## 🎮 Final Project Ideas

After completing the curriculum, try these projects:

1. **Retro Games** - Train agents for Atari/SNES games using Gymnasium
2. **Custom Environments** - Build your own game and train an agent
3. **Multi-agent Competition** - Create agents that compete against each other
4. **Real-world Control** - Sim-to-real transfer for robotics
5. **Rocket League Tournament** - Compete in RLBot tournaments

## 📞 Getting Help

- **Debugging:** Each lesson has a troubleshooting section
- **Theory Questions:** Check Sutton & Barto or post on r/reinforcementlearning
- **Code Issues:** Include error messages and minimal reproducible examples
- **Stuck on Exercises:** Try for 30min, then check solutions (coming soon)

## 🔄 Updates and Contributions

This is a living curriculum. As you progress:
- Note what works well and what's confusing
- Suggest improvements via issues/PRs
- Share your project results!

---

**Ready to start?** Head to [Lesson 1: Python & NumPy Fundamentals](lesson_01_python_numpy.md)

**Questions before starting?** Review the setup instructions above or ask in the communities listed.

Good luck on your RL journey! 🚀
