# Getting Started with the RL Workshop

Welcome to your comprehensive journey from Python basics to building Rocket League AI agents!

## 📋 Quick Start Checklist

Before starting Lesson 1, make sure you have:

- [ ] Python 3.8 or higher installed
- [ ] A code editor (VS Code, PyCharm, or similar)
- [ ] Basic understanding of programming concepts (variables, loops, functions)
- [ ] 60-80 hours available over the next 8-12 weeks
- [ ] Enthusiasm for learning reinforcement learning!

## 🛠️ Initial Setup

### 1. Create Your Project Directory

```bash
mkdir rl-workshop
cd rl-workshop
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv rl_env

# Activate it
# On Windows:
rl_env\Scripts\activate
# On Mac/Linux:
source rl_env/bin/activate
```

### 3. Install Core Dependencies

```bash
# Phase 1 (Lessons 1-4): Fundamentals
pip install numpy matplotlib jupyter
pip install torch torchvision
pip install scikit-learn seaborn

# Phase 2 (Lessons 5-6): Basic RL
pip install gymnasium

# Phase 3 (Lessons 7-9): Deep RL
pip install tensorboard

# Phase 4 (Lessons 10-13): Advanced & Games
pip install stable-baselines3
pip install rlgym rlgym-tools  # For Rocket League
```

### 4. Verify Installation

```python
# test_setup.py
import numpy as np
import torch
import gymnasium as gym

print(f"✓ NumPy version: {np.__version__}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ Gymnasium installed")

# Test simple environment
env = gym.make('CartPole-v1')
print(f"✓ CartPole environment created")
print("\n🎉 Setup complete! You're ready to start learning.")
```

## 📚 How to Use This Workshop

### Learning Approach

**For each lesson:**

Before you dive in, remember that any unfamiliar ML, deep learning, or RL terms are defined in the [Glossary](GLOSSARY.md).

1. **Read the theory section** (30-60 min)
   - Take notes
   - Sketch diagrams
   - Don't just skim!

2. **Type out the code examples** (1-2 hours)
   - Don't copy-paste
   - Experiment with modifications
   - Run everything

3. **Complete the exercises** (1-3 hours)
   - Start with easy ones
   - Challenge yourself with hard ones
   - Document your solutions

4. **Review and reflect** (30 min)
   - What did you learn?
   - What was confusing?
   - How does it connect to previous lessons?

### Time Management

**Recommended schedule (12 weeks):**

- **Weeks 1-2:** Lessons 1-4 (Foundations)
  - 2 lessons per week
  - 4-6 hours per week

- **Weeks 3-5:** Lessons 5-8 (RL Theory & Algorithms)
  - 1-2 lessons per week
  - 6-8 hours per week

- **Weeks 6-8:** Lessons 9-11 (Practical Applications)
  - 1 lesson per week
  - 5-7 hours per week

- **Weeks 9-12:** Lessons 12-13 (Complex Projects)
  - Final projects
  - 8-10 hours per week

### Study Tips

**✅ DO:**
- Code along with examples
- Experiment and break things
- Ask questions in communities
- Take regular breaks
- Review previous lessons
- Celebrate small wins

**❌ DON'T:**
- Skip theory sections
- Copy-paste without understanding
- Rush through exercises
- Give up when stuck
- Work for hours without breaks
- Compare your pace to others

## 🎯 Learning Milestones

Track your progress with these concrete achievements:

### Week 2 Milestone
- [ ] NumPy array operations feel natural
- [ ] Can build and train a simple PyTorch network
- [ ] Achieved >95% on MNIST classification

### Week 4 Milestone
- [ ] Understand Bellman equations intuitively
- [ ] Implemented Q-learning from scratch
- [ ] Agent solves FrozenLake

### Week 6 Milestone
- [ ] Built a DQN that solves CartPole
- [ ] Understand experience replay and target networks
- [ ] Can debug RL training issues

### Week 8 Milestone
- [ ] Implemented PPO algorithm
- [ ] Understand policy gradients
- [ ] Agent solves LunarLander

### Week 10 Milestone
- [ ] Created custom Gym environment
- [ ] Designed reward functions
- [ ] Implemented parallel training

### Week 12 Milestone
- [ ] Built complete RL pipeline for 2D/3D game
- [ ] Trained Rocket League agent
- [ ] Can explain RL concepts to others

## 💡 Getting Help

### When You're Stuck

**1. Debug Systematically**
- Check error messages carefully
- Print intermediate values
- Simplify to minimal example
- Use a debugger

**2. Consult Resources**
- Lesson troubleshooting sections
- Official documentation
- Stack Overflow
- GitHub issues

**3. Ask in Communities**
- r/reinforcementlearning
- r/learnmachinelearning
- OpenAI Discord
- RLGym Discord (for Rocket League)

**4. Take a Break**
- Sometimes stepping away helps
- Come back with fresh perspective
- Sleep on difficult problems

### Common Beginner Questions

**Q: I don't have a GPU. Can I still do this?**
A: Yes! Lessons 1-6 don't need GPU. For later lessons, use Google Colab (free GPU) or train on CPU with smaller networks/fewer episodes.

**Q: How much math do I need to know?**
A: Basic algebra and understanding of derivatives helps. The lessons explain math concepts as needed.

**Q: Can I skip the early lessons?**
A: Only if you're already comfortable with NumPy, PyTorch, and neural networks. The lessons build on each other.

**Q: What if I don't understand something?**
A: Re-read the section, try the code yourself, check the references, and ask in communities. It's normal to not understand everything on first pass.

**Q: How do I know if I'm ready to move on?**
A: Complete the self-check at the end of each lesson. If you can do most items, you're ready.

## 🎮 Project Ideas

After completing the workshop, try these projects:

**Beginner Projects:**
- Tic-tac-toe agent with tabular Q-learning
- Flappy Bird clone with DQN
- Snake game with PPO

**Intermediate Projects:**
- Atari game agent (Pong, Breakout)
- Custom racing game environment
- Multi-agent tag game

**Advanced Projects:**
- 3D robot navigation (PyBullet)
- StarCraft II micro-management
- RLBot tournament entry

## 📖 Supplementary Resources

### Books (Optional but Recommended)
- Sutton & Barto - "RL: An Introduction" (Free online)
- Goodfellow et al. - "Deep Learning" (Free online)

### Video Courses
- David Silver's RL Course (YouTube)
- DeepMind x UCL lectures
- Spinning Up in Deep RL videos

### Practice Platforms
- [OpenAI Gym](https://gymnasium.farama.org/)
- [Kaggle RL competitions](https://www.kaggle.com/competitions)
- [Hugging Face Deep RL Course](https://huggingface.co/deep-rl-course/unit0/introduction)

## 🚀 Your Learning Journey Starts Now!

You're about to embark on an exciting journey. Remember:

- **Be patient with yourself** - RL is challenging
- **Celebrate progress** - Every lesson completed is an achievement
- **Stay curious** - Experiment and explore
- **Have fun** - Build cool agents!

Ready? Let's begin with [Lesson 1: Python & NumPy Fundamentals →](lesson_01_python_numpy.md)

Good luck! 🎓
