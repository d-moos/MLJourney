# Lesson 5: Reinforcement Learning Theory - MDPs and Bellman Equations

**Duration:** 5-7 hours

**Prerequisites:** Lessons 1-4

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. Understand Markov Decision Processes (MDPs) formally
2. Grasp the concepts of states, actions, rewards, and policies
3. Master value functions: V(s) and Q(s,a)
4. Derive and understand Bellman equations
5. Distinguish between policy-based and value-based methods
6. Understand the exploration vs exploitation tradeoff
7. Know the theoretical foundations for all future RL algorithms

## 📖 Theory

If you encounter unfamiliar ML, deep learning, or RL terms in this lesson, see the [Glossary](GLOSSARY.md) for quick definitions and links to the relevant lessons.

### What is Reinforcement Learning?

**Key idea:** An agent learns to make decisions by interacting with an environment to maximize cumulative reward.

```
    ┌─────────┐        action        ┌─────────────┐
    │         │ ──────────────────→  │             │
    │  Agent  │                       │ Environment │
    │         │ ←──────────────────  │             │
    └─────────┘   state, reward       └─────────────┘
```

**Compared to supervised learning.** In supervised learning every training
example comes with a **label** that tells you the correct answer, and each
example is usually independent of the others. In reinforcement learning we
do **not** get such direct labels. The agent only receives **rewards**, and
those rewards can be **delayed** by many time steps. Decisions are also
**sequential**: each action changes the future states the agent will see.
This creates an **exploration problem** (the agent must try actions to
discover their long‑term consequences) and a **credit assignment problem**
(when a reward finally arrives, we must figure out which past actions
deserve the credit or blame).

### Markov Decision Processes (MDPs)

Formally, we describe an RL problem as a **Markov Decision Process
(MDP)**, which is defined by the tuple (S, A, P, R, γ).

The set **S** is the **state space**. It contains all the different
situations the agent can find itself in. For the CartPole environment,
for example, one way to represent the state is as a vector of four
numbers: `[position, velocity, angle, angular_velocity]`. Each unique
combination of these numbers is a different state.

The set **A** is the **action space**. It lists everything the agent is
allowed to do. In some environments A is **discrete**, such as choosing
between "move left" or "move right". In others it is **continuous**, such
as selecting a steering angle between -30° and +30° or choosing a real
valued acceleration.

The function **P** is the **transition function** P(s' | s, a). It tells
us, for each state s and action a, the probability of landing in a next
state s'. The key assumption is the **Markov property**: P(s' | s, a) does
not depend on the full past history (s_{t-1}, s_{t-2}, ...), only on the
current state and action. Intuitively, the current state is a
"sufficient summary" of the past for predicting the future.

The function **R** is the **reward function** R(s, a, s'). It outputs a
single number that is the **immediate feedback** for taking action a in
state s and ending up in state s'. This is how we encode the task: high
rewards for desirable outcomes, low or negative rewards for undesirable
ones. The agent's goal is to discover behaviour that produces large
long‑term rewards according to R.

Finally, **γ** is the **discount factor**, with 0 ≤ γ ≤ 1. It controls how
strongly the agent cares about rewards that will arrive in the future
compared to rewards that arrive right now. If γ = 0, the agent is
completely **myopic**: it only cares about the immediate reward. As γ
approaches 1, the agent becomes increasingly **far‑sighted**, valuing
future rewards almost as much as present ones.

### The RL Objective

The overall goal in reinforcement learning is to find a policy π that
maximizes the **expected cumulative discounted reward**. Instead of only
caring about the very next reward, we care about the whole future, but we
discount it using a factor γ between 0 and 1:

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

Here G_t is called the **return** from time t: it is a single number that
summarizes how good the rest of the episode is expected to be after time t
if we follow a particular policy.

**Why discount?**

There are three main reasons to multiply future rewards by powers of γ
instead of simply summing them without discount.

First, the further into the future a reward is, the more **uncertain** it
becomes. The game might end early, the environment might change, or our
current policy might not actually reach that future situation. Giving
slightly less weight to far‑future rewards reflects this uncertainty.

Second, discounting gives us nice **mathematical properties**. With
0 ≤ γ < 1 the infinite sum of rewards is guaranteed to be finite, which
makes theoretical analysis and many algorithms (like value iteration)
well‑behaved and numerically stable.

Third, in many practical problems we have a natural **preference for
earlier rewards**. Getting +1 point now is often better than maybe getting
the same point much later (just like receiving 10€ today usually feels
better than a promise of 10€ next year). The discount factor encodes this
time preference: smaller γ means “more impatient”, while γ close to 1
means “willing to wait a long time for future rewards”.

### Policies

A **policy** π is a rule for choosing actions based on the current
state. You can think of it as the agent's behaviour: "When I see this
state, what do I do?" In math notation we write a policy as a function
that maps states to actions.

In a **deterministic policy**, the agent always takes the same action in a
given state. Formally we write a = π(s). If you put the agent back into
exactly the same situation, it will make exactly the same choice. A
classic example is a shortest-path navigation policy that always moves
directly toward the goal.

In a **stochastic policy**, written as π(a|s) = P(A_t = a | S_t = s), the
output is not a single action but a **probability distribution** over
actions. The agent will sometimes pick different actions even from the
same state. This randomness is extremely useful for **exploration** (the
agent keeps trying new things) and is essential in **continuous action
spaces**, where the policy typically outputs the mean and spread of a
Gaussian distribution over actions instead of a single fixed value.

### Value Functions

Value functions estimate "how good" states or state-action pairs are:

#### State-Value Function V^π(s)

Expected return starting from state s and following policy π:

```
V^π(s) = E_π[G_t | S_t = s]
        = E_π[R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... | S_t = s]
```

**Interpretation:** "How good is this state if I follow policy π?"

#### Action-Value Function Q^π(s,a)

Expected return starting from state s, taking action a, then following policy π:

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
          = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s, A_t = a]
```

**Interpretation:** "How good is taking action a in state s, then following policy π?"

#### Relationship

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)
```

The value of a state is the expected Q-value over actions chosen by π.

### Bellman Equations

The Bellman equations express recursive relationships in value functions:

#### Bellman Expectation Equation for V^π

```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')]
```

**In words:** The value of a state equals the expected immediate reward plus the discounted value of the next state.

#### Bellman Expectation Equation for Q^π

```
Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a')
```

#### Bellman Optimality Equation

The optimal value function V*(s) satisfies:

```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]
```

The optimal Q-function Q*(s,a) satisfies:

```
Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s',a')
```

**Optimal policy:**

```
π*(s) = argmax_a Q*(s,a)
```

### Policy vs Value-Based Methods

There are three broad families of RL algorithms that you will encounter
throughout the course: **value‑based**, **policy‑based**, and
**actor–critic** methods.

**Value‑based methods** focus on learning a value function such as V(s)
or Q(s, a). Once they have good estimates of these values, they derive a
policy by choosing, in each state, the action with the highest estimated
value: π(s) = argmax_a Q(s, a). Classic examples are tabular
Q‑learning, SARSA, and DQN.

**Policy‑based methods** skip the intermediate value function and instead
learn the policy π_θ(a|s) directly. They treat the policy as a
parameterized function (usually a neural network) and adjust its
parameters θ to maximize expected return using policy gradient
techniques. REINFORCE and PPO are examples of pure policy‑based methods.

**Actor–critic methods** combine both ideas. They maintain an **actor**
(the policy network) and a **critic** (a value network that estimates
V(s) or Q(s, a)). The critic provides lower‑variance learning signals
for the actor, while the actor focuses on choosing good actions. A2C,
A3C, TD3, and SAC all fall into this actor–critic family.

### Exploration vs Exploitation

In every RL problem there is a tension between **exploitation** and
**exploration**. Exploitation means choosing the action that currently
looks best according to our value estimates. Exploration means trying
other actions in case they turn out to be even better than we thought.

If we exploit too aggressively, we can get stuck in a **local optimum**:
the agent keeps repeating a merely decent strategy and never discovers a
much better one. If we explore too much, we keep trying random actions and
never fully capitalize on what we have already learned.

Several standard strategies are used to balance these two goals:

**ε‑greedy.** With ε‑greedy exploration we say: “Most of the time behave
greedily, but occasionally try something random.” On each step we flip a
biased coin. With probability ε we pick a random action (pure
exploration); with probability 1−ε we choose argmax_a Q(s, a) (pure
exploitation). Early in training ε is usually large (for example 1.0) and
then decays over time as the value estimates become more reliable.

**Softmax / Boltzmann exploration.** Instead of a sharp greedy‑versus‑
random decision, softmax exploration treats Q‑values like **preferences**
and converts them into a probability distribution:
P(a) ∝ exp(Q(s, a) / τ). The **temperature** τ controls how random
the choices are: large τ makes all actions almost equally likely, small
τ makes the distribution peak more strongly around the best actions.

**Upper Confidence Bound (UCB).** UCB methods come from multi‑armed
bandits. They explicitly keep track of both the *estimated value* of each
action and the *uncertainty* in that estimate. At each step they choose
the action with the highest **upper confidence bound**, which adds a
“bonus” for rarely tried actions. This gives a principled way to trade
off exploitation and exploration.

**Entropy bonus.** In policy‑gradient methods we often add an extra term
to the loss that encourages the policy’s action distribution to have high
**entropy** (be more spread out). The agent is then rewarded slightly for
keeping its options open rather than collapsing to a deterministic policy
too early. Over time this entropy bonus can be reduced as the policy
matures.

### Types of RL Algorithms

You will often see RL algorithms described as **model-based** or
**model-free**, and as **on-policy** or **off-policy**.

**Model-based methods** try to learn an explicit model of the
environment: the transition probabilities P(s' | s, a) and sometimes
the reward function R(s, a). Once they have such a model, they can
simulate outcomes and **plan** ahead using methods like dynamic
programming or tree search. Model-based approaches can be very
**sample-efficient** (they squeeze a lot of learning out of each real
interaction) but they are often complex to implement and can be brittle
if the learned model is inaccurate.

**Model-free methods** skip the explicit model and learn a value function
or policy **directly from experience**. Q-learning, DQN, and PPO are all
model-free. They tend to require more real environment interaction, but
they are usually simpler and more robust in high-dimensional tasks like
Atari or Rocket League.

An **on-policy** algorithm learns about the behavior of the policy it is
currently executing. SARSA is on-policy: it updates Q-values assuming it
will keep following the same ε-greedy strategy it used to collect the
data.

An **off-policy** algorithm learns about a *different* policy from the
one that generated the data. Q-learning is off-policy: even if the agent
behaves ε-greedily, the target uses max_a Q(s, a) as if it were acting
greedily. Off-policy methods can be more **sample-efficient** because
they can reuse experience collected under many different behavior
policies.

## 💻 Practical Implementation

### Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

np.random.seed(42)
```

### 1. Simple MDP Example - Grid World

```python
class GridWorld:
    """
    Simple 4x4 grid world MDP.

    States: positions in 4x4 grid (16 states)
    Actions: up=0, right=1, down=2, left=3
    Rewards: -1 per step, +10 for goal, -10 for pit
    """
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4

        # Special states
        self.goal_state = (3, 3)
        self.pit_state = (2, 2)
        self.start_state = (0, 0)

        self.state = self.start_state

    def state_to_index(self, state):
        """Convert (row, col) to state index."""
        return state[0] * self.size + state[1]

    def index_to_state(self, index):
        """Convert state index to (row, col)."""
        return (index // self.size, index % self.size)

    def get_next_state(self, state, action):
        """Get next state given current state and action."""
        row, col = state

        # Action effects
        if action == 0:    # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)

        return (row, col)

    def get_reward(self, state):
        """Get reward for entering a state."""
        if state == self.goal_state:
            return 10.0
        elif state == self.pit_state:
            return -10.0
        else:
            return -1.0

    def is_terminal(self, state):
        """Check if state is terminal."""
        return state == self.goal_state or state == self.pit_state

    def reset(self):
        """Reset to start state."""
        self.state = self.start_state
        return self.state

    def step(self, action):
        """Take action and return (next_state, reward, done)."""
        next_state = self.get_next_state(self.state, action)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)

        self.state = next_state
        return next_state, reward, done

    def render(self, value_function=None):
        """Visualize the grid world."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw grid
        for i in range(self.size):
            for j in range(self.size):
                # Color based on state type
                if (i, j) == self.goal_state:
                    color = 'green'
                elif (i, j) == self.pit_state:
                    color = 'red'
                elif (i, j) == self.start_state:
                    color = 'blue'
                else:
                    color = 'white'

                rect = Rectangle((j, self.size-1-i), 1, 1,
                               linewidth=2, edgecolor='black',
                               facecolor=color, alpha=0.3)
                ax.add_patch(rect)

                # Add value if provided
                if value_function is not None:
                    idx = self.state_to_index((i, j))
                    value = value_function[idx]
                    ax.text(j+0.5, self.size-1-i+0.5, f'{value:.1f}',
                           ha='center', va='center', fontsize=12)

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.size+1))
        ax.set_yticks(range(self.size+1))
        ax.grid(True)
        plt.title('Grid World\nBlue=Start, Green=Goal (+10), Red=Pit (-10)')

        return fig, ax

# Create and visualize
env = GridWorld()
fig, ax = env.render()
plt.savefig('gridworld.png')
print("Grid World created and visualized")
```

### 2. Computing Value Functions via Bellman Equations

```python
def policy_evaluation(env, policy, gamma=0.9, theta=0.001):
    """
    Evaluate a policy using iterative Bellman updates.

    Args:
        env: GridWorld environment
        policy: array of shape (n_states, n_actions) with action probabilities
        gamma: discount factor
        theta: convergence threshold

    Returns:
        V: state-value function
    """
    V = np.zeros(env.n_states)

    iteration = 0
    while True:
        delta = 0
        V_new = np.zeros(env.n_states)

        # Update each state
        for s_idx in range(env.n_states):
            state = env.index_to_state(s_idx)

            if env.is_terminal(state):
                continue

            # Bellman expectation equation
            v = 0
            for action in range(env.n_actions):
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(next_state)
                next_s_idx = env.state_to_index(next_state)

                # V(s) = Σ_a π(a|s) [R(s,a) + γ V(s')]
                v += policy[s_idx, action] * (reward + gamma * V[next_s_idx])

            V_new[s_idx] = v
            delta = max(delta, abs(V[s_idx] - V_new[s_idx]))

        V = V_new
        iteration += 1

        if delta < theta:
            break

    print(f"Policy evaluation converged in {iteration} iterations")
    return V

# Test with random policy
random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
V_random = policy_evaluation(env, random_policy)

fig, ax = env.render(V_random)
plt.title('State Values under Random Policy')
plt.savefig('value_function_random.png')
print("Random policy value function computed")
```

### 3. Value Iteration Algorithm

```python
def value_iteration(env, gamma=0.9, theta=0.001):
    """
    Find optimal value function using value iteration.

    Uses Bellman optimality equation:
    V*(s) = max_a [R(s,a) + γ V*(s')]

    Returns:
        V: optimal state-value function
        policy: optimal deterministic policy
    """
    V = np.zeros(env.n_states)

    iteration = 0
    while True:
        delta = 0
        V_new = np.zeros(env.n_states)

        for s_idx in range(env.n_states):
            state = env.index_to_state(s_idx)

            if env.is_terminal(state):
                continue

            # Compute max over actions
            action_values = []
            for action in range(env.n_actions):
                next_state = env.get_next_state(state, action)
                reward = env.get_reward(next_state)
                next_s_idx = env.state_to_index(next_state)

                q_value = reward + gamma * V[next_s_idx]
                action_values.append(q_value)

            V_new[s_idx] = max(action_values)
            delta = max(delta, abs(V[s_idx] - V_new[s_idx]))

        V = V_new
        iteration += 1

        if delta < theta:
            break

    # Extract optimal policy
    policy = np.zeros((env.n_states, env.n_actions))

    for s_idx in range(env.n_states):
        state = env.index_to_state(s_idx)

        if env.is_terminal(state):
            continue

        # Find best action
        action_values = []
        for action in range(env.n_actions):
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(next_state)
            next_s_idx = env.state_to_index(next_state)

            q_value = reward + gamma * V[next_s_idx]
            action_values.append(q_value)

        best_action = np.argmax(action_values)
        policy[s_idx, best_action] = 1.0

    print(f"Value iteration converged in {iteration} iterations")
    return V, policy

V_optimal, optimal_policy = value_iteration(env)

fig, ax = env.render(V_optimal)
plt.title('Optimal State Values (V*)')
plt.savefig('value_function_optimal.png')
print("Optimal value function computed")
```

### 4. Visualize Optimal Policy

```python
def visualize_policy(env, policy):
    """Visualize policy as arrows."""
    fig, ax = env.render(V_optimal)

    # Arrow directions
    arrows = {
        0: (0, 0.3),    # up
        1: (0.3, 0),    # right
        2: (0, -0.3),   # down
        3: (-0.3, 0),   # left
    }

    for s_idx in range(env.n_states):
        state = env.index_to_state(s_idx)

        if env.is_terminal(state):
            continue

        row, col = state
        action = np.argmax(policy[s_idx])
        dx, dy = arrows[action]

        # Draw arrow
        ax.arrow(col + 0.5, env.size - 1 - row + 0.5,
                dx, dy,
                head_width=0.2, head_length=0.1,
                fc='black', ec='black', linewidth=2)

    plt.title('Optimal Policy (arrows show best actions)')
    plt.savefig('optimal_policy.png')
    print("Optimal policy visualized")

visualize_policy(env, optimal_policy)
```

### 5. Q-Value Function

```python
def compute_q_values(env, V, gamma=0.9):
    """
    Compute Q-values from state-value function.

    Q(s,a) = R(s,a) + γ V(s')
    """
    Q = np.zeros((env.n_states, env.n_actions))

    for s_idx in range(env.n_states):
        state = env.index_to_state(s_idx)

        for action in range(env.n_actions):
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(next_state)
            next_s_idx = env.state_to_index(next_state)

            Q[s_idx, action] = reward + gamma * V[next_s_idx]

    return Q

Q_optimal = compute_q_values(env, V_optimal)

# Visualize Q-values for a specific state
state = (1, 1)
s_idx = env.state_to_index(state)
action_names = ['Up', 'Right', 'Down', 'Left']

plt.figure(figsize=(10, 4))
plt.bar(action_names, Q_optimal[s_idx])
plt.xlabel('Action')
plt.ylabel('Q-value')
plt.title(f'Q-values for state {state}')
plt.savefig('q_values_state.png')
print(f"Q-values for state {state}: {Q_optimal[s_idx]}")
```

### 6. Discount Factor Visualization

```python
def compare_discount_factors():
    """Show effect of different discount factors."""
    gammas = [0.5, 0.9, 0.99]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, gamma in enumerate(gammas):
        V, policy = value_iteration(env, gamma=gamma)

        ax = axes[i]
        for row in range(env.size):
            for col in range(env.size):
                state = (row, col)
                s_idx = env.state_to_index(state)

                if state == env.goal_state:
                    color = 'green'
                elif state == env.pit_state:
                    color = 'red'
                else:
                    color = 'white'

                rect = Rectangle((col, env.size-1-row), 1, 1,
                               linewidth=2, edgecolor='black',
                               facecolor=color, alpha=0.3)
                ax.add_patch(rect)

                ax.text(col+0.5, env.size-1-row+0.5, f'{V[s_idx]:.1f}',
                       ha='center', va='center', fontsize=10)

        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.set_title(f'γ = {gamma}')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('discount_factor_comparison.png')
    print("Discount factor comparison saved")

compare_discount_factors()
```

### 7. Bellman Equation Verification

```python
def verify_bellman_equation(env, V, policy, gamma=0.9):
    """
    Verify that V satisfies the Bellman equation for given policy.
    """
    print("Verifying Bellman equation...")

    max_error = 0
    for s_idx in range(env.n_states):
        state = env.index_to_state(s_idx)

        if env.is_terminal(state):
            continue

        # Left side: V(s)
        v_left = V[s_idx]

        # Right side: Σ_a π(a|s) [R(s,a) + γ V(s')]
        v_right = 0
        for action in range(env.n_actions):
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(next_state)
            next_s_idx = env.state_to_index(next_state)

            v_right += policy[s_idx, action] * (reward + gamma * V[next_s_idx])

        error = abs(v_left - v_right)
        max_error = max(max_error, error)

    print(f"Maximum Bellman equation error: {max_error:.6f}")
    if max_error < 1e-3:
        print("✓ Bellman equation satisfied!")
    else:
        print("✗ Bellman equation not satisfied")

verify_bellman_equation(env, V_optimal, optimal_policy)
```

## 📚 Key References

### Textbooks (Essential!)
- **Sutton & Barto** - "Reinforcement Learning: An Introduction" (2nd ed.)
  - [Free PDF](http://incompleteideas.net/book/the-book-2nd.html)
  - Chapters 3-4 are crucial for this lesson

### Papers
- **Bellman, R. (1957)** - "A Markovian Decision Process"
- **Watkins (1989)** - "Learning from Delayed Rewards" (PhD thesis)

### Video Lectures
- **David Silver's RL Course** (DeepMind) - Lectures 1-3
  - [YouTube playlist](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

### Interactive Resources
- [GridWorld Playground](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)
- [Spinning Up: RL Introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

## 🏋️ Exercises

### Exercise 1: Bellman Equations (Easy)

Given:
- State s with V(s) = 5
- Two actions: a₁ and a₂
- π(a₁|s) = 0.7, π(a₂|s) = 0.3
- Taking a₁ leads to s₁' (V=8) with reward r=2
- Taking a₂ leads to s₂' (V=3) with reward r=5
- γ = 0.9

**Verify that V(s) satisfies the Bellman equation.**

### Exercise 2: Optimal Policy (Medium)

Modify the GridWorld to have:
- Two goals: (3,3) with reward +10 and (0,3) with reward +5
- Pit at (2,2) with reward -10
- Movement cost -1 per step

Find the optimal policy with γ=0.9. Which goal does the agent prefer and why?

### Exercise 3: Value Iteration Implementation (Medium)

Implement value iteration completely from scratch (without using the provided code). Test it on a custom 5x5 GridWorld with multiple pits and goals.

### Exercise 4: Discount Factor Analysis (Medium-Hard)

For the standard GridWorld:

1. Run value iteration with γ ∈ {0.1, 0.5, 0.9, 0.95, 0.99}
2. For each γ, record:
   - Optimal value of start state V*(s₀)
   - Number of iterations to converge
   - Average path length to goal
3. Plot and explain the relationships

### Exercise 5: Stochastic Environment (Hard)

Modify GridWorld to be stochastic:
- Intended action succeeds with probability 0.8
- With probability 0.1 each: move perpendicular to intended direction

Implement and compare:
1. Value iteration in stochastic environment
2. Policy extracted from Q-values
3. Does the optimal policy change? Why or why not?

```python
# Starter code
class StochasticGridWorld(GridWorld):
    def get_transition_prob(self, state, action, next_state):
        """
        Return P(next_state | state, action).

        Intended action: 0.8 probability
        Perpendicular actions: 0.1 each
        """
        # Your implementation here
        pass
```

## 🔧 Troubleshooting Tips

### Common Conceptual Issues

1. **Confusing V(s) and Q(s,a)**
   - V(s): value of being in state s
   - Q(s,a): value of taking action a in state s
   - Relationship: V(s) = max_a Q(s,a) for optimal policy

2. **Discount factor confusion**
   - γ close to 0: myopic (immediate rewards)
   - γ close to 1: farsighted (long-term planning)
   - γ = 1 only valid for episodic tasks

3. **Bellman equation vs Bellman optimality**
   - Bellman equation: for a given policy π
   - Bellman optimality: for optimal policy π*

4. **Policy evaluation vs improvement**
   - Evaluation: compute V^π for fixed π
   - Improvement: find better π from V^π
   - Iteration: alternate until convergence

### Debugging Value Iteration

```python
# Add verbose mode to track convergence
def value_iteration_verbose(env, gamma=0.9):
    V = np.zeros(env.n_states)
    deltas = []

    for iteration in range(1000):
        delta = 0
        V_new = np.zeros(env.n_states)

        for s_idx in range(env.n_states):
            # ... (value iteration update)

            delta = max(delta, abs(V[s_idx] - V_new[s_idx]))

        V = V_new
        deltas.append(delta)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: delta = {delta:.6f}")

    # Plot convergence
    plt.plot(deltas)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Max value change (delta)')
    plt.title('Value Iteration Convergence')
    plt.grid(True)
    plt.show()

    return V
```

## ✅ Self-Check

Before moving to Lesson 6, you should be able to:

- [ ] Define an MDP formally
- [ ] Explain the Markov property
- [ ] Compute returns with discounting
- [ ] Distinguish V(s) from Q(s,a)
- [ ] Write out Bellman equations from memory
- [ ] Explain the difference between Bellman expectation and optimality equations
- [ ] Implement value iteration
- [ ] Extract policy from value function
- [ ] Understand exploration vs exploitation

## 🚀 Next Steps

You now have the theoretical foundation for all RL algorithms!

**Next:** [Lesson 6: Tabular Q-Learning Implementation](lesson_06_tabular_qlearning.md)
- Implement Q-learning from scratch
- Apply it to GridWorld and FrozenLake
- Understand temporal difference learning
- Explore ε-greedy exploration

The theory becomes practice!

---

**Estimated completion time:** 5-7 hours (including exercises)

**Next lesson:** [Lesson 6: Tabular Q-Learning →](lesson_06_tabular_qlearning.md)
