# Core Libraries
import os
import sys
import time
import datetime
import warnings

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

# Financial Data
import yfinance as yf
import quantstats as qs
import ta

# Machine Learning - Supervised Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Machine Learning - Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy

# Reinforcement Learning and Environments
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import ARS, MaskablePPO, RecurrentPPO, QRDQN, TRPO

# Imitation Learning
from imitation.algorithms import bc
from imitation.testing.reward_improvement import is_significant_reward_improvement
from imitation.data.types import Transitions

# Interactive Brokers API
from ib_insync import *

from typing import Callable

from collections import Counter

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def calculate_accuracy(best_actions, model_actions):
    correct_predictions = sum(1 for best, model in zip(best_actions, model_actions) if best == model)
    accuracy = correct_predictions / len(best_actions) * 100
    return accuracy

def train_model(
    model_name,
    model=None,
    create_model=False,
    vec_env=None,
    env=None,  # Required for BC
    transitions=None,  # Required for BC
    iterations=1,
    train_timesteps=10_000,
    log_frec=10,
    log_base_dir="./logs",
    n_steps=10,
    batch_size=50,
    learning_rate=0.001,
    ent_coef=0.10,
    seed=1,
    bc_batches=10_000,
    bc_log_interval=1_000
):
    """
    Train PPO, A2C, DQN, or BC models.

    Args:
        model_name (str): The name of the model ('PPO', 'A2C', 'DQN', or 'BC').
        model: A pre-initialized model instance (optional if create_model is True).
        create_model (bool): Whether to dynamically create the model.
        vec_env: The vectorized training environment (required for PPO, A2C, DQN).
        env: The single environment (required for BC).
        transitions: Expert data for Behavior Cloning (required for BC).
        iterations (int): Number of iterations for training RL models.
        train_timesteps (int): Total timesteps for training RL models.
        log_frec (int): Logging frequency.
        log_base_dir (str): Base directory for TensorBoard logs.
        n_steps (int): Number of steps for RL models.
        batch_size (int): Batch size for RL models.
        learning_rate (float): Learning rate for RL models.
        ent_coef (float): Entropy coefficient for RL models.
        seed (int): Random seed for reproducibility.
        bc_batches (int): Number of batches for BC training.
        bc_log_interval (int): Logging interval for BC.

    Returns:
        model: The trained model.
    """
    if create_model:
        if model_name == 'PPO':
            if vec_env is None:
                raise ValueError("vec_env is required for PPO.")
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                ent_coef=ent_coef,
                verbose=1
            )
        elif model_name == 'A2C':
            if vec_env is None:
                raise ValueError("vec_env is required for A2C.")
            model = A2C(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                ent_coef=ent_coef,
                verbose=1
            )
        elif model_name == 'DQN':
            if vec_env is None:
                raise ValueError("vec_env is required for DQN.")
            model = DQN(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                verbose=1
            )
        elif model_name == 'BC':
            if env is None or transitions is None:
                raise ValueError("env and transitions are required for BC.")
            model = BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                rng=np.random.default_rng(seed),
                batch_size=batch_size
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    if model is None:
        raise ValueError("Model instance must be provided or create_model must be True.")

    if model_name in ['PPO', 'A2C', 'DQN']:
        # Train RL models
        for i in range(iterations):
            log_dir = f"{log_base_dir}/{model_name}/{i}_run/"
            if i == 0:
                model.learn(
                    total_timesteps=train_timesteps,
                    progress_bar=False,
                    log_interval=log_frec,
                    tb_log_name=f"{i}_run",
                    reset_num_timesteps=True
                )
            else:
                model.learn(
                    total_timesteps=train_timesteps,
                    progress_bar=False,
                    log_interval=log_frec,
                    tb_log_name=f"{i}_run",
                    reset_num_timesteps=False
                )
    elif model_name == 'BC':
        # Evaluate BC policy before training
        env.reset(seed)
        mean_reward_bc_before, std_reward_bc_before = evaluate_policy(
            model.policy, env, n_eval_episodes=1, return_episode_rewards=False, deterministic=True
        )
        print("BC Learner rewards before training:")
        print(f"Mean reward: {mean_reward_bc_before} +/- {std_reward_bc_before:.2f}")

        # Train BC model
        model.train(
            n_batches=bc_batches,
            log_interval=bc_log_interval,
            progress_bar=False
        )

        # Evaluate BC policy after training
        env.reset(seed)
        mean_reward_bc_after, std_reward_bc_after = evaluate_policy(
            model.policy, env, n_eval_episodes=1, return_episode_rewards=False, deterministic=True
        )
        print("BC Learner rewards after training:")
        print(f"Mean reward: {mean_reward_bc_after} +/- {std_reward_bc_after:.2f}")

    return model