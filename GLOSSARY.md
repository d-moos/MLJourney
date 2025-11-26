# Glossary of ML and RL Terms

This glossary collects the main machine learning (ML), deep learning, and reinforcement
learning (RL) terms used across Lessons 1–13. For deeper explanations, see the
referenced lessons.

## General ML & Neural Networks

- **Dataset** – A collection of examples `(x, y)` used for training and evaluating models.
- **Feature** – A measurable property of an input (e.g., pixel intensity, position).
- **Label / Target** – The desired output `y` associated with an input `x` in supervised learning.
- **Supervised learning** – Learning a function from labeled pairs `(x, y)` (Lesson 3).
- **Regression** – Predicting continuous values (e.g., price, value estimate) (Lesson 3).
- **Classification** – Predicting discrete classes (e.g., digit 0–9) (Lesson 3–4).
- **Loss function** – A scalar measure of prediction error used to train models (Lesson 3).
- **Mean Squared Error (MSE)** – Loss equal to the squared difference between prediction and target.
- **Mean Absolute Error (MAE)** – Loss equal to the absolute difference between prediction and target.
- **Huber loss** – Loss that is quadratic for small errors and linear for large errors (robust MSE).
- **Cross-entropy loss** – Loss for probability distributions, common in classification and policies.
- **Optimizer** – Algorithm that updates parameters to reduce loss (e.g., SGD, Adam) (Lesson 3).
- **Gradient descent / SGD** – Update rule using the gradient of the loss with respect to parameters.
- **Learning rate** – Step size used by the optimizer when updating parameters.
- **Epoch** – One full pass over the training dataset.
- **Batch / mini-batch** – A subset of data used for one gradient update.
- **Overfitting** – When a model memorizes training data but generalizes poorly (Lesson 3).
- **Underfitting** – When a model is too simple to capture patterns in the data.
- **Regularization** – Techniques that reduce overfitting (e.g., weight decay, dropout, data aug).
- **Activation function** – Non-linear function applied between layers (e.g., ReLU, sigmoid).
- **ReLU (Rectified Linear Unit)** – Activation `ReLU(x) = max(0, x)`; standard in deep nets.
- **Dropout** – Regularization that randomly zeros some activations during training.
- **Batch normalization** – Layer that normalizes activations per mini-batch for stability (Lesson 4).
- **Fully connected / linear layer** – Layer where each input connects to each output neuron.
- **Convolutional Neural Network (CNN)** – Network using convolution layers, good for images (Lesson 4).
- **Tensor** – Generalized multi-dimensional array used in PyTorch (Lesson 2).

## Reinforcement Learning Basics

- **Environment** – The world the agent interacts with; maps actions to next states and rewards.
- **Agent** – The decision-making entity that chooses actions based on states.
- **Episode** – One run of the environment from start to a terminal or truncated state.
- **Markov Decision Process (MDP)** – Mathematical model of RL tasks defined by `(S, A, P, R, γ)` (Lesson 5).
- **State** – The information describing the current situation of the environment.
- **Action** – A choice the agent can make in a given state.
- **Reward** – Scalar feedback signal indicating immediate desirability of a transition.
- **Return** – Discounted sum of future rewards from a time step onward (Lesson 5).
- **Discount factor (γ)** – Number in `[0, 1]` that controls how much future rewards matter.
- **Policy** – Mapping from states to actions or action probabilities (deterministic or stochastic).
- **Value function `V(s)`** – Expected return from state `s` under a policy (Lesson 5).
- **Action-value / Q-function `Q(s, a)`** – Expected return from taking action `a` in state `s` (Lesson 5).
- **Bellman equation** – Recursive relationship expressing values in terms of immediate reward and next-state values.
- **Model-based RL** – Methods that learn or assume a model of `P` and `R` for planning.
- **Model-free RL** – Methods that learn values or policies directly from experience.
- **On-policy** – Learning about the policy currently being executed (e.g., SARSA, some actor-critic).
- **Off-policy** – Learning about a different (often greedy) policy than the behavior policy (e.g., Q-learning, DQN).
- **Exploration vs exploitation** – Tradeoff between trying new actions and using known good ones.
- **ε-greedy** – Strategy that takes a random action with probability ε, otherwise the best-known action.
- **Temporal-Difference (TD) learning** – Methods that update estimates using bootstrapped targets (Lesson 6).
- **TD error** – Difference between target and current value estimate; the "surprise" in TD learning.
- **Monte Carlo return** – Return computed by summing full future rewards to the end of an episode.

## Deep RL & Advanced Techniques

- **Q-network** – Neural network approximating the Q-function `Q(s, a)` (Lesson 7).
- **Policy network** – Neural network representing a parameterized policy `πθ(a|s)` (Lesson 8).
- **Value network** – Neural network approximating a state-value function `V(s)`.
- **Replay buffer / experience replay** – Memory that stores past transitions for off-policy training (Lesson 7).
- **Target network** – Slowly updated copy of a network used to compute stable targets (Lesson 7).
- **Double DQN** – DQN variant that reduces overestimation by decoupling action selection and evaluation.
- **Dueling network** – Architecture that separately estimates state value and action advantages.
- **Policy gradient** – Methods that directly optimize policy parameters via gradient of expected return (Lesson 8).
- **Advantage function `A(s, a)`** – Measures how much better an action is compared to the average in a state.
- **Actor-critic** – Algorithms with separate policy (actor) and value (critic) networks (Lesson 8).
- **PPO (Proximal Policy Optimization)** – Policy-gradient algorithm with a clipped objective for stable updates.
- **Entropy bonus** – Extra term in the loss that encourages more random (higher entropy) policies.
- **Parallel / vectorized environments** – Multiple environments run simultaneously to collect data faster (Lesson 11).
- **Reward shaping** – Modifying or augmenting rewards to guide learning (Lessons 10–11).
- **Potential-based reward shaping** – Shaping using `F(s, s') = γ Φ(s') − Φ(s)` that preserves the optimal policy.
- **Curriculum learning** – Training on a sequence of tasks from easy to hard to gradually build skills (Lesson 11).
- **Intrinsic motivation** – Internal reward signal encouraging exploration (e.g., curiosity bonuses).
- **Curiosity bonus** – Intrinsic reward proportional to prediction error of a learned dynamics or feature model.
- **Count-based exploration** – Intrinsic reward that decreases as a state (or region) is visited more often.
- **Self-play** – Training agents by playing against their own current or past versions (Lesson 11 & 13).
- **State encoder** – Function that converts raw observations into normalized feature vectors for networks.

If you encounter a term in the lessons that is not clarified in context, check here first,
then follow the link to the lesson where it is used in more depth.

