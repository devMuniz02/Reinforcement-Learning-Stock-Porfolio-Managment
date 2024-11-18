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
    df = pd.concat(df, axis=1)
    df_unscaled = pd.concat(df_unscaled, axis=1)

    # Return the environment setup
    def env_fn():
        return #TradingEnvUnique(df_unscaled, history_length, reward_type)

    #env = TradingEnvUnique(df_unscaled, history_length, reward_type)
    #check_env(env)

    return env, env_fn, df.index, scalers, df, df_unscaled
def create_env(history_length, reward_type, start_date, end_date, stocks, scaler, scalers):
    df = []
    df_unscaled = []
    interval = '1d'
    num_stocks = len(stocks)
    add_date = False

    while True:
        try:
            date_interval = yf.download(stocks[0], start=start_date, end=end_date, interval=interval, progress=False).shape[0]
            if date_interval < 27:
                start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                add_date = True 
            break
        except Exception as e:
            print("An error occurred:", e)
            print("Retrying...")
            time.sleep(5)
      
    if scalers is None:
        scalers = []
    """while True:
        try:
            # Fetch FRED data
            CPI = fred.get_series('MEDCPIM158SFRBCLE', observation_start=start_date)
            M30US = fred.get_series('MORTGAGE30US', observation_start=start_date)
            UNRA = fred.get_series('UNRATE', observation_start=start_date)
            SP500 = fred.get_series('SP500', observation_start=start_date)
            NEWH = fred.get_series('HOUST', observation_start=start_date)
            NASDAQ = fred.get_series('NASDAQCOM', observation_start=start_date)
            CARSA = fred.get_series('TOTALSA', observation_start=start_date)

            fred_df = pd.concat([SP500, CPI, M30US, UNRA, NEWH, NASDAQ, CARSA], axis=1)
            fred_df.columns = ['SP500', 'CPI', 'M30US', 'UNRA', 'NEWH', 'NASDAQ', 'CARSA']

            fred_df = fred_df.bfill()
            fred_df = fred_df.ffill()
        except Exception as e:
            print("An error occurred:", e)
            print("Retrying...")
            time.sleep(15)"""

    data = yf.download(stocks, start=start_date, end=end_date, progress=False)
    
    # Processing each ticker individually
    for ticker in stocks:
        data[('week_day_number', ticker)] = data['Close'][ticker].index.weekday + 1
        data[('5MA', ticker)] = data['Close'][ticker].rolling(window=5).mean()
        data[('10MA', ticker)] = data['Close'][ticker].rolling(window=10).mean()
        data[('15MA', ticker)] = data['Close'][ticker].rolling(window=15).mean()
        data[('20MA', ticker)] = data['Close'][ticker].rolling(window=20).mean()
        data[('Daily_Return', ticker)] = data['Adj Close'][ticker].pct_change() * 100

        # Adding technical indicators
        bollinger = ta.volatility.BollingerBands(close=data['Close'][ticker])
        data[('Bollinger_High', ticker)] = bollinger.bollinger_hband()
        data[('Bollinger_Low', ticker)] = bollinger.bollinger_lband()
        data[('RSI', ticker)] = ta.momentum.RSIIndicator(close=data['Close'][ticker]).rsi()
        macd = ta.trend.MACD(close=data['Close'][ticker])
        data[('MACD', ticker)] = macd.macd()
        data[('MACD_Signal', ticker)] = macd.macd_signal()
        data[('12EMA', ticker)] = ta.trend.EMAIndicator(close=data['Close'][ticker], window=12).ema_indicator()
        data[('26EMA', ticker)] = ta.trend.EMAIndicator(close=data['Close'][ticker], window=26).ema_indicator()
        data[('OBV', ticker)] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'][ticker], volume=data['Volume'][ticker]).on_balance_volume()

    # Backward fill the NaN values after all calculations
    data = data.bfill().ffill() 
    df = data
    df_unscaled = data
    def env_fn():
        return TradingEnvUniqueMultiple(df_unscaled, history_length, reward_type)
    
    env = TradingEnvUniqueMultiple(df_unscaled, history_length, reward_type)
    #check_env(env)
    
    return env, env_fn, date_interval, scalers, df, df_unscaled

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