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
from imitation.algorithms.bc import BC #from imitation.algorithms import bc
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
    if reward_temp > 0:
        return reward_temp
    else:
        return 100-(100*100/(100+reward_temp))

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
            next_close_price = self.df_unscaled.iloc[self.history_length+self.steps]['Close'].iloc[self.Actions[-1]] #price of action stock on next day
            close_price = self.df_unscaled.iloc[self.history_length+self.steps-1]['Close'].iloc[self.Actions[-1]] #price today of action stock to buy

            cash_today = self.cash[-1]
            if self.steps != 0: #if not first step
                if self.shares[-1] != 0:
                    sell_price = self.df_unscaled.iloc[self.history_length+self.steps-1]['Close'].iloc[self.Actions[-2]] #sell price of current stock
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
                    sell_price = self.df_unscaled.iloc[self.history_length+self.steps-1]['Close'].iloc[self.Actions[-2]]
                    next_close_price = self.df_unscaled.iloc[self.history_length+self.steps]['Close'].iloc[self.Actions[-2]]
                    close_price = self.df_unscaled.iloc[self.history_length+self.steps-1]['Close'].iloc[self.Actions[-2]]
                    cash_today = (self.cash[-1] + (1 - self.sell_com) * self.shares[-1] * sell_price)
                    self.shares.append(0)
                    self.cash.append(cash_today)
                    self.portfolio.append(self.cash[-1] + self.shares[-1] * close_price)
                    reward_temp = 0 #-100 * (next_close_price-close_price) / close_price #negative because is a loss if it goes up and you sold
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
        if self.portfolio[-1] == 0:
            done = True
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
def create_env_unique(history_length, reward_type, start_date, end_date, stocks, scaler, scalers, pred_dir):
    df = []
    df_unscaled = []
    interval = '1d'
    num_stocks = len(stocks)
    
    # Ensure dates are in datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_date_temp = start_date - datetime.timedelta(days=history_length)
    
    # Loop through each stock to download and process data
    for i, stock in enumerate(stocks):
        output_filename = os.path.join(pred_dir, f'{stock}_predictions_{history_length}.csv')
        
        # Download or load data
        if os.path.isfile(output_filename) and os.path.getsize(output_filename) > 0:
            print(f"Loading data for {stock} from {output_filename}")
            data = pd.read_csv(output_filename, index_col=0, parse_dates=True)
            unscaled_data_combined = data.copy()
        else:
            print(f"No data found for {stock}, downloading from Yahoo Finance")
            while True:
                try:
                    data = yf.download(stock, start=start_date_temp, end=end_date, interval=interval, progress=False)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel('Ticker')
                    break
                except Exception as e:
                    print("An error occurred:", e)
                    print("Retrying...")
                    time.sleep(15)

        # Add technical indicators
        data['5MA'] = data['Close'].rolling(window=5).mean()
        data['10MA'] = data['Close'].rolling(window=10).mean()
        data['15MA'] = data['Close'].rolling(window=15).mean()
        data['20MA'] = data['Close'].rolling(window=20).mean()
        data['Daily_Return'] = data['Adj Close'].pct_change() * 100
        data['Bollinger_High'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_hband()
        data['Bollinger_Low'] = ta.volatility.BollingerBands(close=data['Close']).bollinger_lband()
        data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
        
        # Save unscaled data
        unscaled_data_combined = data.copy()

        # Scale data if needed
        if scaler:
            if len(scalers) <= i:
                scalers.append([MinMaxScaler() for _ in range(data.shape[1])])
            data_values = data.values
            data_scaled = np.zeros_like(data_values)
            for col_idx in range(data.shape[1]):
                data_scaled[:, col_idx] = scalers[i][col_idx].fit_transform(data_values[:, col_idx].reshape(-1, 1)).flatten()
            data_combined = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
        else:
            data_combined = data

        # Assign MultiIndex columns: (Price, Ticker)
        data_combined.columns = pd.MultiIndex.from_product([data.columns, [stock]], names=["Price", "Ticker"])
        unscaled_data_combined.columns = pd.MultiIndex.from_product([unscaled_data_combined.columns, [stock]], names=["Price", "Ticker"])

        # Append to list for later concatenation
        df.append(data_combined)
        df_unscaled.append(unscaled_data_combined)

    # Concatenate all stock data into one DataFrame
    df_unscaled = pd.concat(df_unscaled, axis=1)

    # Return the environment setup
    def env_fn():
        return #TradingEnvUnique(df_unscaled, history_length, reward_type)

    #env = TradingEnvUnique(df_unscaled, history_length, reward_type)
    #check_env(env)

    return env, env_fn, df.index, scalers, df, df_unscaled

def create_env(history_length, reward_type, start_date, end_date, stocks, scaler=None, scalers=None):
    """
    Creates a trading environment using stock data and technical indicators.

    Parameters:
        history_length (int): Length of history to use for state representation.
        reward_type (str): Type of reward function to use.
        start_date (str): Start date for stock data in 'YYYY-MM-DD' format.
        end_date (str): End date for stock data in 'YYYY-MM-DD' format.
        stocks (list): List of stock tickers to include.
        scaler (bool): Whether to use a scaler (default: None).
        scalers (list): List of scalers (default: None).

    Returns:
        env: The trading environment.
        env_fn: Function to instantiate the environment.
        date_interval: Number of trading days in the data.
        scalers: List of scalers used.
        df: Processed DataFrame with all stock data.
        df_unscaled: Unscaled version of the DataFrame.
    """
    interval = '1d'
    add_date = False

    # Validate and adjust start_date to ensure sufficient data points
    while True:
        try:
            date_interval = yf.download(stocks[0], start=start_date, end=end_date, interval=interval, progress=False, actions=True, rounding=True).shape[0]
            if date_interval < 27:  # If data is insufficient, adjust the start_date
                start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                add_date = True
            else:
                break
        except Exception as e:
            print(f"An error occurred while validating dates: {e}")
            print("Retrying...")
            time.sleep(5)

    # Initialize scalers if not provided
    if scalers is None:
        scalers = []

    # Download data for all stocks
    try:
        data = yf.download(stocks, start=start_date, end=end_date, interval=interval, progress=False, group_by='ticker', actions=True, rounding=True)
    except Exception as e:
        print(f"Failed to fetch stock data: {e}")
        return None, None, None, scalers, None, None

    # Initialize a list to hold processed DataFrames
    processed_data = []

    # Process each stock's data
    for ticker in stocks:
        if ticker not in data or data[ticker].empty:
            print(f"No data available for {ticker}, skipping...")
            continue
        
        stock_data = data[ticker].copy()

        # Add technical indicators
        stock_data['week_day_number'] = stock_data.index.weekday + 1
        stock_data['5MA'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['10MA'] = stock_data['Close'].rolling(window=10).mean()
        stock_data['15MA'] = stock_data['Close'].rolling(window=15).mean()
        stock_data['20MA'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['Daily_Return'] = stock_data['Close'].pct_change() * 100

        # Adding more technical indicators using TA library
        bollinger = ta.volatility.BollingerBands(close=stock_data['Close'])
        stock_data['Bollinger_High'] = bollinger.bollinger_hband()
        stock_data['Bollinger_Low'] = bollinger.bollinger_lband()
        stock_data['RSI'] = ta.momentum.RSIIndicator(close=stock_data['Close']).rsi()
        macd = ta.trend.MACD(close=stock_data['Close'])
        stock_data['MACD'] = macd.macd()
        stock_data['MACD_Signal'] = macd.macd_signal()
        stock_data['12EMA'] = ta.trend.EMAIndicator(close=stock_data['Close'], window=12).ema_indicator()
        stock_data['26EMA'] = ta.trend.EMAIndicator(close=stock_data['Close'], window=26).ema_indicator()
        stock_data['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=stock_data['Close'], volume=stock_data['Volume']
        ).on_balance_volume()

        # Backward and forward fill missing values
        stock_data = stock_data.bfill().ffill()

        # Add MultiIndex for the stock
        stock_data.columns = pd.MultiIndex.from_product([[ticker], stock_data.columns], names=['Ticker', 'Price'])

        # Append processed data
        processed_data.append(stock_data)

    # Combine all stocks' processed data into a single DataFrame with MultiIndex
    df = pd.concat(processed_data, axis=1)

    # Adjust the MultiIndex to have 'Price' as the first level and 'Ticker' as the second level
    df.columns = df.columns.swaplevel('Price', 'Ticker')

    # Sort the MultiIndex to ensure consistency
    df = df.sort_index(axis=1)
    df_unscaled = df.copy()

    # Define the environment creation function
    def env_fn():
        return TradingEnvUniqueMultiple(df_unscaled, history_length, reward_type)

    # Instantiate the environment
    env = TradingEnvUniqueMultiple(df_unscaled, history_length, reward_type)

    # Return results
    return env, env_fn, date_interval, scalers, df, df_unscaled

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

def calculate_accuracy(best_actions, model_actions):
    correct_predictions = sum(1 for best, model in zip(best_actions, model_actions) if best == model)
    accuracy = correct_predictions / len(best_actions) * 100
    return accuracy

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
    
def create_training_env(history_length, reward_type, start_date, end_date, stocks, n_envs):
    """
    Create a vectorized environment for training.
    """
    env, env_fn, date_interval, scalers, df, df_unscaled = create_env(
        history_length, reward_type, start_date, end_date, stocks, True, []
    )
    check_env(env)
    vec_env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    return env, vec_env

def create_evaluation_env(history_length, reward_type, start_date, end_date, stocks, n_envs=1):
    """
    Create a vectorized environment for evaluation.
    """
    env, env_fn, _, _, _, _ = create_env(history_length, reward_type, start_date, end_date, stocks, True, [])
    check_env(env)
    vec_env = make_vec_env(env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    return env, vec_env

def save_model(model, folder_path, file_name="best_model"):
    """
    Save the trained model to the specified folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model.save(os.path.join(folder_path, file_name))

from sb3_contrib import ARS, MaskablePPO, RecurrentPPO, QRDQN, TRPO
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3

from stable_baselines3 import A2C, DQN, HER, PPO
from sb3_contrib import ARS, MaskablePPO, RecurrentPPO, QRDQN, TRPO

from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from sb3_contrib import ARS, MaskablePPO, RecurrentPPO, QRDQN, TRPO
from imitation.algorithms.bc import BC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.algorithms import sqil


def train_model(
    model_name,
    model=None,
    callback = None,
    create_model=False,
    vec_env=None,
    env=None,  # Required for HER, BC, GAIL, AIRL
    transitions=None,  # Required for SQIL, BC, GAIL, AIRL
    iterations=1,
    train_timesteps=10_000,
    log_frec=10,
    log_base_dir="./logs",
    n_steps=10,
    batch_size=50,
    learning_rate=0.001,
    ent_coef=0.10,
    seed=1,
    eval_episodes=100,
    bc_batches = 1000
):
    """
    Train RL models (e.g., PPO, A2C), HER, Behavioral Cloning (BC), SQIL, GAIL, or AIRL.

    Args:
        model_name (str): Name of the model to train. Supported:
            - RL Models: 'PPO', 'A2C', 'DQN', 'DDPG', 'SAC', 'TD3', 'TRPO', 'QRDQN', 'ARS', 'MaskablePPO', 'RecurrentPPO'.
            - HER: Requires single env and RL algo.
            - Imitation Learning: 'BC', 'SQIL', 'GAIL', 'AIRL'.
        model: Predefined model instance (optional if create_model=True).
        create_model (bool): Whether to dynamically create the model.
        vec_env: Vectorized environment for RL models.
        env: Single environment for HER and imitation learning models.
        transitions: Expert transitions (required for SQIL, BC, GAIL, AIRL).
        iterations (int): Number of training iterations.
        train_timesteps (int): Total timesteps per training iteration.
        log_frec (int): Logging frequency.
        log_base_dir (str): Directory for TensorBoard logs.
        n_steps (int): Number of steps for on-policy models.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for training.
        ent_coef (float): Entropy coefficient for RL models.
        seed (int): Random seed for reproducibility.
        eval_episodes (int): Number of evaluation episodes.

    Returns:
        model: Trained model instance.
    """
    os.makedirs(log_base_dir, exist_ok=True)

    # RL Models
    if model_name in [
        "PPO",
        "A2C",
        "DQN",
        "DDPG",
        "SAC",
        "TD3",
        "TRPO",
        "QRDQN",
        "MaskablePPO",
        "RecurrentPPO",
        "ARS",
    ]:
        if vec_env is None:
            raise ValueError(f"{model_name} requires `vec_env`.")
        if create_model:
            if model_name == "PPO":
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    ent_coef=ent_coef,
                    tensorboard_log=log_base_dir,
                    verbose=0,
                    seed=seed,
                    policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
                )
            elif model_name == "A2C":
                model = A2C(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    ent_coef=ent_coef,
                    tensorboard_log=log_base_dir,
                    verbose=0,
                    seed=seed,
                )
            elif model_name == "DQN":
                model = DQN(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    buffer_size=50000,
                    batch_size=batch_size,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )
            elif model_name == "DDPG":
                model = DDPG(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    buffer_size=50000,
                    batch_size=batch_size,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )
            elif model_name == "SAC":
                model = SAC(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    buffer_size=50000,
                    batch_size=batch_size,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )
            elif model_name == "TD3":
                model = TD3(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    buffer_size=50000,
                    batch_size=batch_size,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )
            elif model_name == "TRPO":
                model = TRPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )
            elif model_name == "QRDQN":
                model = QRDQN(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    buffer_size=50000,
                    batch_size=batch_size,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )
            elif model_name == "MaskablePPO":
                model = MaskablePPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    ent_coef=ent_coef,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )
            elif model_name == "RecurrentPPO":
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    ent_coef=ent_coef,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )
            elif model_name == "ARS":
                model = ARS(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=learning_rate,
                    zero_policy=False,
                    tensorboard_log=log_base_dir,
                    verbose=1,
                    seed=seed,
                )

        for i in range(iterations):
            model.learn(
                total_timesteps=train_timesteps,
                progress_bar=False,
                log_interval=log_frec,
                tb_log_name=f"RL_{i}_run",
                reset_num_timesteps=(i == 0),
                callback=callback
            )
        return model

    # SQIL Model
    elif model_name == "SQIL":
        if vec_env is None or transitions is None:
            raise ValueError("SQIL requires `vec_env` and `transitions`.")
        model = sqil.SQIL(
            venv=vec_env,
            demonstrations=transitions,
            policy="MlpPolicy",
        )
        vec_env.reset(seed)
        mean_reward, std_reward = evaluate_policy(model.policy, vec_env, eval_episodes)
        print(f"SQIL rewards before training: {mean_reward:.2f} ± {std_reward:.2f}")
        model.train(total_timesteps=train_timesteps)
        mean_reward, std_reward = evaluate_policy(model.policy, vec_env, eval_episodes)
        print(f"SQIL rewards after training: {mean_reward:.2f} ± {std_reward:.2f}")
        return model

    # Behavioral Cloning (BC)
    elif model_name == "BC":
        if env is None or transitions is None:
            raise ValueError("BC requires `env` and `transitions`.")
        model = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=np.random.default_rng(seed),
            batch_size=batch_size,
        )
        env.reset(seed)
        mean_reward, std_reward = evaluate_policy(model.policy, env, eval_episodes)
        print(f"BC rewards before training: {mean_reward:.2f} ± {std_reward:.2f}")
        model.train(n_batches=bc_batches, progress_bar=False)
        mean_reward, std_reward = evaluate_policy(model.policy, env, eval_episodes)
        print(f"BC rewards after training: {mean_reward:.2f} ± {std_reward:.2f}")
        return model

    # GAIL Model
    elif model_name == "GAIL":
        if vec_env is None or transitions is None:
            raise ValueError("GAIL requires `vec_env` and `transitions`.")
        gen_algo = PPO(
            policy="MlpPolicy",
            env=vec_env,
            batch_size=batch_size,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            verbose=1,
            seed=seed,
        )
        reward_net = BasicRewardNet(
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            normalize_input_layer=RunningNorm,
        )
        model = GAIL(
            demonstrations=transitions,
            venv=vec_env,
            gen_algo=gen_algo,
            reward_net=reward_net,
            allow_variable_horizon=True,
        )
        vec_env.reset(seed)
        mean_reward, std_reward = evaluate_policy(gen_algo, vec_env, eval_episodes)
        print(f"GAIL rewards before training: {mean_reward:.2f} ± {std_reward:.2f}")
        model.train(total_timesteps=train_timesteps)
        mean_reward, std_reward = evaluate_policy(gen_algo, vec_env, eval_episodes)
        print(f"GAIL rewards after training: {mean_reward:.2f} ± {std_reward:.2f}")
        return model

    # AIRL Model
    elif model_name == "AIRL":
        if vec_env is None or transitions is None:
            raise ValueError("AIRL requires `vec_env` and `transitions`.")
        gen_algo = PPO(
            policy="MlpPolicy",
            env=vec_env,
            batch_size=batch_size,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            verbose=1,
            seed=seed,
        )
        reward_net = BasicRewardNet(
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            normalize_input_layer=RunningNorm,
        )
        model = AIRL(
            demonstrations=transitions,
            venv=vec_env,
            gen_algo=gen_algo,
            reward_net=reward_net,
            allow_variable_horizon=True,
        )
        vec_env.reset(seed)
        mean_reward, std_reward = evaluate_policy(gen_algo, vec_env, eval_episodes)
        print(f"AIRL rewards before training: {mean_reward:.2f} ± {std_reward:.2f}")
        model.train(total_timesteps=train_timesteps)
        mean_reward, std_reward = evaluate_policy(gen_algo, vec_env, eval_episodes)
        print(f"AIRL rewards after training: {mean_reward:.2f} ± {std_reward:.2f}")
        return model

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


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
