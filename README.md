# Deep Reinforcement Learning with Advantage Actor-Critic (A2C)

## Overview
The project focuses on applying the A2C algorithm, a reinforcement learning technique that combines elements of policy gradients (Actor) and value estimation (Critic), to balance a pole on a cart in the OpenAI Gym's CartPole-v1 environment. The goal is for an agent to learn a policy that maximizes the cumulative reward by making sequential decisions.

## Environment Setup

### Installation

The necessary libraries are installed using pip:
- `gym`: Provides the CartPole-v1 environment and tools for reinforcement learning.
- `imageio`: Used for creating and saving videos of the agent's performance.

### Running the Random Agent

To visualize the CartPole-v1 environment using a random agent:
- A loop runs for 500 steps where the agent takes random actions (`env.action_space.sample()`).
- Each frame of the environment is captured (`env.render(mode='rgb_array')`) and stored in `frames`.
- The frames are saved as an MP4 video (`imageio.mimsave('./cartpole_random.mp4', frames, fps=25)`).

## A2C Algorithm

### Model Architecture (`ActorCritic`)

The neural network model (`ActorCritic`) consists of:
- `fc1`: A fully connected layer that takes input observations and outputs a hidden representation (`hidden_size`).
- `fc_actor`: Another fully connected layer that outputs action probabilities using a softmax activation.
- `fc_critic`: A fully connected layer that outputs a value estimate for the state.

### Training (`A2CAgent`)

The `A2CAgent` class facilitates training of the A2C model:
- **Initialization**: Sets up parameters like number of episodes (`num_episodes`), maximum steps per episode (`max_steps`), learning rate (`lr`), and network architecture.
- **Choosing Actions**: The `choose_action` method selects actions based on the current policy (actor network).
- **Compute Returns**: Calculates discounted returns (`returns`) for each time step to estimate the advantage.
- **Training Loop**: Iterates through episodes and steps within each episode:
  - Interacts with the environment, collects rewards and log probabilities of actions.
  - Computes advantages and calculates actor and critic losses.
  - Updates model parameters using backpropagation (`loss.backward()` and `optimizer.step()`).

## Evaluation

### Running Trained Agent

To evaluate the trained agent:
- **Initialization**: Sets up the environment and the trained A2C model.
- **Evaluation Loop**: Runs multiple episodes (`num_episodes`) to evaluate the agent's performance:
  - Renders each frame of the environment.
  - Uses the trained model to select actions based on the learned policy (`choose_action`).
  - Accumulates rewards and prints the reward for each episode.

## Results

After training and evaluating the model, the README suggests calculating the average reward over all episodes to assess the agent's performance.

---

This Markdown document provides a structured explanation of setting up the environment, implementing the A2C algorithm, training the agent, evaluating its performance, and interpreting results. Adjust the sections and details as per your actual implementation and any additional information you find relevant.
