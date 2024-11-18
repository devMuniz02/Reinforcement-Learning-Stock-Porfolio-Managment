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

def evaluate_best(venv, expert_actions, SEED):
    print("Best")
    i = 0
    best_actions = []
    observation, info = venv.reset(seed=SEED)
    total_reward = 0
    while True:
        # Select action
        action = expert_actions[i]  # Adjust index if needed
        best_actions.append(action)
        # Step the environment
        observation, reward, done, _, _ = venv.step(action)
        total_reward += reward
        i += 1
        if done:
            break
    best_returns = venv.returns()
    venv.render(mode='total')
    venv.render(mode='rewards')
    return best_actions, best_returns, total_reward

def evaluate_buy(venv, SEED):
    print("Buy")
    i = 0
    buy_actions = []
    observation, info = venv.reset(seed=SEED)
    total_reward = 0
    while True:
        action = 1  # Always "Buy"
        buy_actions.append(action)
        observation, reward, done, _, _ = venv.step(action)
        total_reward += reward
        i += 1
        if done:
            break
    buy_returns = venv.returns()
    venv.render(mode='total')
    return buy_actions, buy_returns, total_reward

def evaluate_model(venv, trainer, name, SEED, has_policy=False):
    """
    Evaluates the provided model or trainer in a given environment.

    Args:
        venv: The environment to evaluate the model on. Must support `reset`, `step`, and `render` methods.
        trainer: The model or agent used for evaluation. If `has_policy` is True, it must have a `policy.predict` method.
        name (str): A descriptive name for the evaluation (e.g., for logging or debugging purposes).
        SEED (int): The random seed to ensure reproducibility of environment behavior.
        has_policy (bool, optional): Indicates whether the `trainer` has a `policy` attribute. Defaults to False.

    Returns:
        tuple:
            - model_actions (list): A list of actions taken by the model during evaluation.
            - returns (any): The cumulative return of the environment, as provided by `venv.returns()`.
            - total_reward (float): The total reward accumulated during the evaluation.
    """
    print(name)
    i = 0
    model_actions = []
    observation, info = venv.reset(seed=SEED)
    total_reward = 0
    while True:
        if has_policy:
            action = trainer.policy.predict(observation, deterministic=True)[0]
        else:
            action = trainer.predict(observation, deterministic=True)[0]
        model_actions.append(action)
        observation, reward, done, _, _ = venv.step(action)
        total_reward += reward
        i += 1
        if done:
            break
    returns = venv.returns()
    venv.render(mode='total')
    venv.render(mode='rewards')
    return model_actions, returns, total_reward

def evaluate_various(venv, expert_actions, sqil_trainer, bc_trainer, learner_gail, learner_airl, SEED):
    best_actions, best_returns, best_total_reward = evaluate_best(venv, expert_actions, SEED)
    buy_actions, buy_returns, buy_total_reward = evaluate_buy(venv, SEED)
    
    sqil_actions, sqil_returns, sqil_total_reward = evaluate_model(venv, sqil_trainer, "SQIL", SEED, has_policy=True)
    bc_actions, bc_returns, bc_total_reward = evaluate_model(venv, bc_trainer, "BC", SEED, has_policy=True)
    gail_actions, gail_returns, gail_total_reward = evaluate_model(venv, learner_gail, "GAIL", SEED, has_policy=False)
    airl_actions, airl_returns, airl_total_reward = evaluate_model(venv, learner_airl, "AIRL", SEED, has_policy=False)
    
    sqil_accuracy = calculate_accuracy(best_actions, sqil_actions)
    bc_accuracy = calculate_accuracy(best_actions, bc_actions)
    gail_accuracy = calculate_accuracy(best_actions, gail_actions)
    airl_accuracy = calculate_accuracy(best_actions, airl_actions)
    buy_accuracy = calculate_accuracy(best_actions, buy_actions)
    
    print(f"SQIL Accuracy: {sqil_accuracy}%")
    print(f"BC Accuracy: {bc_accuracy}%")
    print(f"GAIL Accuracy: {gail_accuracy}%")
    print(f"AIRL Accuracy: {airl_accuracy}%")
    print(f"Buy Accuracy: {buy_accuracy}%")
    
    print(f"SQIL Total Reward: {sqil_total_reward}")
    print(f"BC Total Reward: {bc_total_reward}")
    print(f"GAIL Total Reward: {gail_total_reward}")
    print(f"AIRL Total Reward: {airl_total_reward}")
    print(f"Buy Total Reward: {buy_total_reward}")
    print(f"Best Total Reward: {best_total_reward}")
    
def evaluate_all(name, model, has_policy, SEED):
    print('Valid')
    evaluate_best(valid_venv,expert_actions_valid,SEED)
    evaluate_buy(valid_venv,SEED)
    evaluate_model(valid_venv,name=name,has_policy=has_policy,SEED=SEED,trainer=model)
    print('Train')
    evaluate_best(train_venv,expert_actions_train,SEED)
    evaluate_buy(train_venv,SEED)
    evaluate_model(train_venv,name=name,has_policy=has_policy,SEED=SEED,trainer=model)
    print('Test')
    evaluate_best(test_venv,expert_actions_test,SEED)
    evaluate_buy(test_venv,SEED)
    evaluate_model(test_venv,name=name,has_policy=has_policy,SEED=SEED,trainer=model)