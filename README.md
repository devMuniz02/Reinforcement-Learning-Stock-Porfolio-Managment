# Reinforcement-Learning-Stock-Portfolio-Management

A Python package for stock portfolio management using **Reinforcement Learning (RL)** and **Behavioral Cloning (BC)**. This project allows users to train, evaluate, and test agents in a stock trading environment with support for custom data, multiple algorithms, and reward structures.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Examples and Results](#examples-and-results)
- [License](#license)

---

## Features

- **Reinforcement Learning (RL):** Train agents using algorithms like PPO, SAC, or DQN for optimizing stock portfolios.
- **Behavioral Cloning (BC):** Learn trading strategies from historical expert data.
- **Custom Environments:** Easily integrate new stock data or reward structures.
- **Visualization Tools:** Monitor rewards, stock prices, and portfolio values during training and evaluation.
- **Multi-Environment Support:** Train using vectorized environments for faster convergence.

---

## Quick Start

Here’s how to get started with training and evaluating your model:

### 1. **Training an RL Model**
```python
from rl_portfolio.env import create_training_env
from rl_portfolio.agents import train_rl_agent

# Create training environment
history_length = 10
reward_type = "simple"
start_date = "2020-01-01"
end_date = "2020-12-31"
stocks = ["AAPL", "GOOG", "TSLA"]
n_envs = 4

env, vec_env = create_training_env(history_length, reward_type, start_date, end_date, stocks, n_envs)

# Train the model
model = train_rl_agent(vec_env, algorithm="PPO", timesteps=100_000)
```

### 2. **Evaluating the RL Model**
```python
from stable_baselines3.common.evaluation import evaluate_policy

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
```

### 3. **Using Behavioral Cloning (BC)**
```python
from rl_portfolio.bc import train_bc_agent

# Train the Behavioral Cloning model
bc_model = train_bc_agent(expert_data="expert_trades.csv", history_length=10)

# Test the BC model
bc_reward = bc_model.test(env)
print(f"BC Model Reward: {bc_reward}")
```

---

## How It Works

This package leverages custom Gym environments to simulate stock trading. Agents learn by interacting with these environments:

1. **Reinforcement Learning (RL):**
   - Uses frameworks like Stable-Baselines3 to train agents on policy optimization (e.g., PPO, SAC).
   - Rewards are calculated based on portfolio performance.

2. **Behavioral Cloning (BC):**
   - Trains models on expert trajectories to mimic historical trading strategies.

3. **Metrics:**
   - Tracks rewards, Sharpe ratios, and portfolio values during training and evaluation.

---

## Examples and Results

### 1. **Training Reward Curve**
This graph shows the cumulative reward achieved during training using PPO:

![Training Reward Curve](https://github.com/yourusername/Reinforcement-Learning-Stock-Portfolio-Management/images/reward_curve.png)

---

### 2. **Portfolio Value Over Time**
Performance of RL and BC models compared to a baseline (e.g., Buy and Hold):

| **Algorithm**       | **Final Portfolio Value** | **Sharpe Ratio** |
|----------------------|---------------------------|------------------|
| PPO (RL)            | $125,000                  | 1.45             |
| Behavioral Cloning   | $110,000                  | 1.20             |
| Buy and Hold (Baseline) | $100,000               | 1.05             |

---

### 3. **Trade Visualization**
The following plot shows trades executed by the RL agent during evaluation:

![Trade Visualization](https://github.com/yourusername/Reinforcement-Learning-Stock-Portfolio-Management/images/trades.png)

---

### Check Out My Other Projects
Explore more of my AI and ML work [here](https://github.com/devMuniz02/AI-ML-Code-and-projects/).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
