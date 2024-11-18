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


def save_model(model, folder_path, file_name="best_model"):
    """
    Save the trained model to the specified folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model.save(os.path.join(folder_path, file_name))
    
def collect_expert_data(env, seed):
    """
    Collect expert trajectories and generate transitions for Behavior Cloning (BC).

    Args:
        env: The trading environment used for collecting expert data.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - transitions (Transitions): Transitions object for Behavior Cloning.
            - expert_actions (np.array): Expert actions collected during the trajectory.
    """
    # Set the random seed for the environment
    obs, _ = env.reset(seed=seed)
    done = False

    # Initialize storage for expert data
    expert_obs = []
    expert_actions = []

    while not done:
        # Calculate daily percentage change for each stock to determine the best action
        percentage_changes = np.abs(
            (env.df_unscaled['Close'].pct_change().iloc[env.steps + env.history_length] + 1).fillna(0).values
        )
        percentage_changes = np.append(percentage_changes, 1)  # Add cash/no-action option

        # Determine the current held position
        current_position = expert_actions[-1] if expert_actions else None

        # Adjust for commissions
        adjusted_changes = [
            (change - env.buy_com - env.sell_com) if action != current_position and action != env.num_stocks
            else (change - env.sell_com) if action == env.num_stocks and action != current_position
            else change
            for action, change in enumerate(percentage_changes)
        ]

        # Choose the action with the highest adjusted percentage change
        best_action = np.argmax(adjusted_changes)

        # Store the observation and best action
        expert_obs.append(obs)
        expert_actions.append(best_action)

        # Take the best action and proceed in the environment
        obs, _, done, _, _ = env.step(best_action)

    # Convert expert data to numpy arrays
    expert_obs = np.array(expert_obs)
    expert_actions = np.array(expert_actions)

    # Collect transitions for Behavior Cloning
    expert_observations = []
    expert_actions_list = []
    terminals = []

    # Reset the environment
    observation, _ = env.reset(seed=seed)

    for action in expert_actions:
        # Step the environment using the expert action
        next_observation, reward, done, _, _ = env.step(action)

        # Store the observation, action, and done status
        expert_observations.append(observation)
        expert_actions_list.append(action)
        terminals.append(done)

        # Update the current observation
        observation = next_observation

        # Reset the environment if the episode is done
        if done:
            observation, _ = env.reset(seed=seed)

    # Convert collected data into Transitions
    transitions = Transitions(
        obs=np.array(expert_observations),
        acts=np.array(expert_actions_list),
        next_obs=np.zeros_like(expert_observations),  # Placeholder for next_obs
        dones=np.array(terminals),
        infos=np.array([{}] * len(expert_observations))  # Placeholder for infos
    )

    return transitions, expert_actions
