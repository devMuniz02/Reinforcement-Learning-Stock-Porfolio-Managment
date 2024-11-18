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

def sqs_reward_func(reward_temp):
    """Square of reward_temp, with sign preservation."""
    if reward_temp > 0:
        return reward_temp ** 2
    else:
        return -reward_temp ** 2

def sqh_reward_func(reward_temp):
    """Half the square of reward_temp, with sign preservation."""
    if reward_temp > 0:
        return (reward_temp ** 2) / 2
    else:
        return -(reward_temp ** 2) / 2

def smp_reward_func(reward_temp):
    """Increments or decrements reward by 1 based on the sign of reward_temp."""
    if reward_temp > 0:
        return 1
    else:
        return -1

def bin_reward_func(reward_temp):
    """Increments reward by 1 if reward_temp is positive, no penalty otherwise."""
    if reward_temp > 0:
        return 1
    else:
        return 0

def lnr_reward_func(reward_temp):
    """Linear reward, simply returns reward_temp."""
    return reward_temp

def stp_reward_func(reward_temp):
    """Fixed reward increment by 1."""
    return 1

class TradingEnvUnique(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple trading environment where the agent must decide whether to
    sell, hold, or buy a financial asset.
    """

    metadata = {"render.modes": ["console", "human"]}

    # Define actions for each stock
    SELL = 0   # Decide to sell the asset
    BUY = 1    # Decide to buy the asset

    LNR = 'LNR'
    SQS = 'SQS'
    SQH = 'SQH'
    SMP = 'SMP'
    BIN = 'BIN'

    def calculate_reward(self, reward_temp):
      # Call the selected reward function
      return self.reward_func(reward_temp)

    def __init__(self, df_unscaled, history_length, reward_type, render_mode="console", buy_com=0.01, sell_com=0.01):
        super().__init__()

        self.reward_func_map = {
        'SQS': sqs_reward_func,
        'SQH': sqh_reward_func,
        'SMP': smp_reward_func,
        'BIN': bin_reward_func,
        'LNR': lnr_reward_func,
        'STP': stp_reward_func
        }

        # Set the reward function based on reward_type, with a default if reward_type is not found
        self.reward_func = self.reward_func_map.get(reward_type, lnr_reward_func)

        self.df_unscaled = df_unscaled
        self.TimeSize = np.shape(df_unscaled)[0]
        self.ColSize = np.shape(df_unscaled)[1]
        self.action_space = spaces.Discrete(2)  # Only 2 actions: SELL and BUY
        self.steps = 0
        self.history_length = history_length
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.history_length*self.ColSize,), dtype=np.float64)
        self.render_mode = render_mode
        self.cashval = 200
        self.shareval = 2
        self.buy_com = buy_com
        self.sell_com = sell_com

        # Initialize portfolio for the stock
        self.portfolio = []
        self.cash = []
        self.shares = []
        cash = self.cashval
        shares = self.shareval
        stock_portfolio = cash + shares * self.df_unscaled['Close'].tolist()[self.history_length]
        self.portfolio.append(stock_portfolio)
        self.cash.append(cash)
        self.shares.append(shares)

        # Initialize action counts
        self.Actions = []
        self.BuyNum = 0
        self.SellNum = 0
        self.rewards = []
        self.sum_rewards = []

    def step(self, action):
        reward = 0
        reward_temp = 0

        # Simulate price change for the stock
        # Perform one step of the environment
        # action: 0 (SELL), 1 (BUY)
        self.Actions.append(action)
        # Simulate price change
        next_close_price = self.df_unscaled['Close'].iloc[self.history_length+self.steps+1]  # Next day's close price
        close_price = self.df_unscaled['Close'].iloc[self.history_length+self.steps]  # Current day's close price

        if action == self.BUY:
            # Perform the buy action at market close price
            self.shares.append(self.shares[-1] + (1 - self.buy_com) * self.cash[-1] / close_price)
            self.cash.append(0)
            # Update portfolio value
            self.portfolio.append(self.cash[-1] + self.shares[-1] * close_price)
            self.BuyNum += 1

            # Calculate the portfolio value at the same day's market close
            reward_temp = 100 * (next_close_price-close_price) / close_price

        elif action == self.SELL:
            # Perform the sell action at market close price
            self.cash.append(self.cash[-1] + (1 - self.sell_com) * self.shares[-1] * close_price)
            self.shares.append(0)
            # Update portfolio value at market open (initially)
            self.portfolio.append(self.cash[-1])
            self.SellNum += 1

            # Calculate the portfolio value at the same day's market close
            reward_temp = -100 * (next_close_price-close_price) / close_price

        reward_temp -= 0.001

        reward_temp = self.calculate_reward(reward_temp)

        reward += reward_temp
        if np.isinf(reward):
            print((next_close_price,close_price,self.steps))
            print(f"{reward_temp} Inf reward detected: {reward}")
            df = self.df[0]
            # Define the column of interest
            column_of_interest = 'Close'

            # Find rows where the value in 'Close' is 0, inf, or NaN
            mask = df[column_of_interest].isin([0, np.inf]) | df[column_of_interest].isna()
            rows_with_conditions = df[mask]

            print(rows_with_conditions)

        # Check if episode is done after a certain number of steps
        self.steps += 1
        done = self.steps >= (self.TimeSize - self.history_length - 1) # -1 for extra day for next_close_price

        observation = np.array(self.df_unscaled.iloc[self.steps:self.steps + self.history_length])
        self.rewards.append(reward_temp)
        if self.sum_rewards == []:
            self.sum_rewards.append(reward_temp)
        else:
            self.sum_rewards.append(self.sum_rewards[-1]+reward_temp)


        return observation.flatten(), reward, done, False, {}

    def reset(self, seed=None, options = None):
        # Reset environment state to initial state
        self.steps = 0
        self.portfolio = []
        self.cash = []
        self.shares = []

        # Reset portfolio for the stock
        cash = self.cashval
        shares = self.shareval
        stock_portfolio = cash + shares * self.df_unscaled['Close'].tolist()[self.history_length]
        self.portfolio.append(stock_portfolio)
        self.cash.append(cash)
        self.shares.append(shares)

        # Reset action counts
        self.Actions = []
        self.BuyNum = 0
        self.SellNum = 0
        self.rewards = []
        self.sum_rewards = []

        observation = np.array(self.df_unscaled.iloc[self.steps:self.steps + self.history_length])

        return observation.flatten(), {}

    def last_observation(self):
        observation = np.array(self.df_unscaled.iloc[-self.history_length:])
        return observation.flatten()

    def render(self, mode='console'):
        """
        Render the environment.
        """
        if mode == "console":
            print(f"Stock price history: {self.portfolio}")
        elif mode == "multiple":
            print(f'Sell: {self.SellNum} Buy: {self.BuyNum}')
            plt.figure(figsize=(10, 6))
            plt.plot(self.df_unscaled['Close'].tolist()[self.history_length:], color='blue', marker='o', label='Stock price')
            for j, val in enumerate(self.Actions):
                if val == 0:
                    plt.plot(j, self.df_unscaled['Close'].tolist()[j+self.history_length], 'ro')  # Red dot
                elif val == 1:
                    plt.plot(j, self.df_unscaled['Close'].tolist()[j+self.history_length], 'go')  # Yellow dot
            plt.plot(self.portfolio, color='black', marker='.', label='Portfolio')
            plt.title(f'Stock price history')
            plt.xlabel('Time step')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()
            plt.show()
        elif mode == "total":
            plt.figure(figsize=(10, 6))
            print(f'Sell: {self.SellNum} Buy: {self.BuyNum}')
            plt.plot(self.df_unscaled['Close'].tolist()[self.history_length:], color='blue', marker='s')
            for j, val in enumerate(self.Actions):
                if val == 0:
                    plt.plot(j, self.df_unscaled['Close'].tolist()[j+self.history_length], 'ro')  # Red dot
                elif val == 1:
                    plt.plot(j, self.df_unscaled['Close'].tolist()[j+self.history_length], 'go')  # Yellow dot

            plt.plot(self.portfolio, color='black', marker='.', label='Portfolio')
            plt.title(f'Portfolio price history')
            plt.xlabel('Time step')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()
            plt.show()
            plt.figure(figsize=(10, 6))
            plt.plot(self.portfolio, color = 'blue', marker = '*', label='Total portfolio')
            plt.title(f'Total Portfolio price history')
            plt.ylim(0,1.2*np.max(self.portfolio))
            plt.xlabel('Time step')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()
            plt.show()

        elif mode == "rewards":
            plt.figure(figsize=(10, 6))
            plt.plot(self.rewards, color='black', marker='.', label='Rewards')
            plt.title(f'Rewards')
            plt.xlabel('Time step')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid()
            plt.show()
            plt.figure(figsize=(10, 6))
            plt.plot(self.sum_rewards, color = 'blue', marker = '*', label='Total rewards')
            plt.title(f'Total Rewards')
            plt.xlabel('Time step')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid()
            plt.show()
            print(f'Total rewards: {sum(self.sum_rewards)}')

    def returns(self):
        returns = 100*(self.portfolio[-1]-self.portfolio[0])/self.portfolio[0]
        print(f'Total return of portfolio: {"{:.2f}".format(returns)}%')

    def returns_num(self):
        returns = 100*(self.portfolio[-1]-self.portfolio[0])/self.portfolio[0]
        return returns
    
class TradingEnvUniqueMultiple(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple trading environment where the agent must decide whether to
    sell, hold, or buy a financial asset.
    """

    metadata = {"render.modes": ["console", "human"]}

    LNR = 'LNR'
    SQS = 'SQS'
    SQH = 'SQH'
    SMP = 'SMP'
    BIN = 'BIN'

    def calculate_reward(self, reward_temp):
      # Call the selected reward function
      return self.reward_func(reward_temp)

    def __init__(self, df_unscaled, history_length, reward_type, render_mode="console", buy_com=0.01, sell_com=0.01):
        super().__init__()

        self.reward_func_map = {
        'SQS': sqs_reward_func,
        'SQH': sqh_reward_func,
        'SMP': smp_reward_func,
        'BIN': bin_reward_func,
        'LNR': lnr_reward_func,
        'STP': stp_reward_func
        }

        # Set the reward function based on reward_type, with a default if reward_type is not found
        self.reward_func = self.reward_func_map.get(reward_type, lnr_reward_func)

        self.df_unscaled = df_unscaled
        self.num_stocks = int(len(df_unscaled.columns.get_level_values('Ticker').unique()))
        self.TimeSize = int(np.shape(df_unscaled)[0])
        self.ColSize = int(len(df_unscaled.iloc[0].index.get_level_values('Price').unique()))
        self.action_space = spaces.Discrete(self.num_stocks+1)  # Only 2 actions: SELL and BUY
        self.steps = 0
        self.history_length = history_length
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.history_length*self.num_stocks*self.ColSize,), dtype=np.float64)
        self.render_mode = render_mode
        self.cashval = 100
        self.shareval = 0
        self.buy_com = buy_com
        self.sell_com = sell_com

        # Initialize portfolio for the stock
        self.portfolio = []
        self.cash = []
        self.shares = []
        cash = self.cashval
        shares = self.shareval
        stock_portfolio = cash
        self.portfolio.append(stock_portfolio)
        self.cash.append(cash)
        self.shares.append(shares)

        # Initialize action counts
        self.Actions = []
        self.rewards = []
        self.sum_rewards = []

    def step(self, action):
        reward = 0
        reward_temp = 0
        sell_price = 0

        self.Actions.append(int(action))
        if action != self.num_stocks: #actions is a stock
            next_close_price = self.df_unscaled.iloc[self.history_length+self.steps,self.num_stocks+self.Actions[-1]] #price of action stock on next day
            close_price = self.df_unscaled.iloc[self.history_length+self.steps-1,self.num_stocks+self.Actions[-1]] #price today of action stock to buy

            cash_today = self.cash[-1]
            if self.steps != 0: #if not first step
                if self.shares[-1] != 0:
                    sell_price = self.df_unscaled.iloc[self.history_length+self.steps-1,self.num_stocks+self.Actions[-2]] #sell price of current stock
                if self.Actions[-1] != self.Actions[-2]: #if different stock than previous
                    cash_today = (self.cash[-1] + (1 - self.sell_com) * self.shares[-1] * sell_price)
                    self.shares.append((1 - self.buy_com) * cash_today / close_price)
                    self.cash.append(0)

            else: #if fist step then only buy
                self.shares.append((1 - self.buy_com) * cash_today / close_price)
                self.cash.append(0)

            self.portfolio.append(self.cash[-1] + self.shares[-1] * close_price)
            reward_temp = 100 * (next_close_price-close_price) / close_price
            
        else: #action is SELL
            cash_today = self.cash[-1]
            if self.steps != 0:
                if self.shares[-1] != 0: #if you have shares
                    sell_price = self.df_unscaled.iloc[self.history_length+self.steps-1,self.num_stocks+self.Actions[-2]]
                    next_close_price = self.df_unscaled.iloc[self.history_length+self.steps,self.num_stocks+self.Actions[-2]]
                    close_price = self.df_unscaled.iloc[self.history_length+self.steps-1,self.num_stocks+self.Actions[-2]]
                    cash_today = (self.cash[-1] + (1 - self.sell_com) * self.shares[-1] * sell_price)
                    self.shares.append(0)
                    self.cash.append(cash_today)
                    self.portfolio.append(self.cash[-1] + self.shares[-1] * close_price)
                    reward_temp = -100 * (next_close_price-close_price) / close_price #negative because is a loss if it goes up and you sold
                else: # if you dont have shares
                    self.portfolio.append(self.portfolio[-1])
                    self.shares.append(self.shares[-1])
                    self.cash.append(self.cash[-1])
                    reward_temp = 0
            else: #if first step then you dont have shares
                self.portfolio.append(self.portfolio[-1])
                self.shares.append(self.shares[-1])
                self.cash.append(self.cash[-1])
        reward_temp -= 100*0.001

        reward_temp = self.calculate_reward(reward_temp)

        reward += reward_temp
        if np.isinf(reward):
            print((next_close_price,close_price,self.steps))
            print(f"{reward_temp} Inf reward detected: {reward}")

        # Check if episode is done after a certain number of steps
        done = self.steps >= (self.TimeSize - self.history_length - 1) # -1 for extra day for next_close_price
        self.steps += 1
        #10 time size
        #5 history
        #[0 1 2 3 4 5 6 7 8 9] time size
        #[0 1 2 3 4] steps
        #[5 6 7 8 9 10] index
        #[4 5 6 7 8 9] df
        #[5 6 7 8 9 10] self.history_length+self.steps close
        #[6 7 8 9 10 11] self.history_length+self.steps+1 nextclose
        #10-5-1 = 4 sum
        #done = 0 >= (10 - 5 - 1) = 4 
        observation = self.df_unscaled.iloc[self.steps:self.steps + self.history_length].values

        self.rewards.append(reward_temp)
        if self.sum_rewards == []:
            self.sum_rewards.append(reward_temp)
        else:
            self.sum_rewards.append(self.sum_rewards[-1]+reward_temp)


        return observation.flatten(), reward, done, False, {}

    def reset(self, seed=None, options = None):
        # Reset environment state to initial state
        self.steps = 0
        self.portfolio = []
        self.cash = []
        self.shares = []

        # Reset portfolio for the stock
        cash = self.cashval
        shares = self.shareval
        stock_portfolio = cash
        self.portfolio.append(stock_portfolio)
        self.cash.append(cash)
        self.shares.append(shares)

        # Reset action counts
        self.Actions = []
        self.rewards = []
        self.sum_rewards = []

        observation = self.df_unscaled.iloc[self.steps:self.steps + self.history_length].values

        return observation.flatten(), {}

    def last_observation(self):
        observation = self.df_unscaled.iloc[-self.history_length:].values
        return observation.flatten()

    def render(self, mode='console'):
        """
        Render the environment.
        """
        if mode == "console":
            print(f"Stock price history: {self.portfolio}")
        elif mode == "multiple":
            print(f'Sell: {self.SellNum} Buy: {self.BuyNum}')
            plt.figure(figsize=(10, 6))
            plt.plot(self.portfolio, color='black', marker='.', label='Portfolio')
            plt.title(f'Stock price history')
            plt.xlabel('Time step')
            plt.ylabel('Price')
            plt.legend()
            plt.grid()
            plt.show()
        elif mode == "total":
            plt.figure(figsize=(10, 6))
            plt.plot(self.portfolio, color='blue', marker='o', markersize=5, linewidth=2, label='Total Portfolio')
            plt.title('Total Portfolio Price History', fontsize=16)
            plt.ylim(0, 1.2 * np.max(self.portfolio))
            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.show()

        elif mode == "rewards":
            # Rewards Plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.rewards, color='darkorange', marker='*', alpha=0.7, label='Rewards')
            plt.title('Rewards per Time Step', fontsize=16)
            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel('Reward', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.show()

            # Total Rewards Plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.sum_rewards, color='green', marker='*', markersize=6, linewidth=2, label='Total Rewards')
            plt.title('Cumulative Total Rewards', fontsize=16)
            plt.xlabel('Time Step', fontsize=14)
            plt.ylabel('Reward', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.show()

            print(f'Total rewards: {sum(self.rewards)}')

    def returns(self):
        returns = 100*(self.portfolio[-1]-self.portfolio[0])/self.portfolio[0]
        print(f'Total return of portfolio: {"{:.2f}".format(returns.item())}%')

    def returns_num(self):
        returns = 100*(self.portfolio[-1]-self.portfolio[0])/self.portfolio[0]
        return returns
