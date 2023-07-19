from typing import Tuple
from enum import Enum

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from dataset import BKG, SIG


def _relative_percentage(background: pd.DataFrame, signal: pd.DataFrame) -> float:
    """Get the relative percentage of signal events in the dataset"""
    total = signal.shape[0] + background.shape[0]
    return signal.shape[0] / total


def _sample_data(data: pd.DataFrame, background: pd.DataFrame, signal: pd.DataFrame, sample_size: int = 100_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample the data to get a balanced dataset"""
    sig_size = int(sample_size * _relative_percentage(background, signal))
    bkg_size = sample_size - sig_size

    bkg = data[data['is_signal'] == BKG].sample(bkg_size, random_state=42)
    sig = data[data['is_signal'] == SIG].sample(sig_size, random_state=42)

    return bkg, sig


def _split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into train, val and test datasets"""
    train, val, test = np.split(data, [int((1/3) * len(data)), int((2/3) * len(data))])
    return train, val, test


def train_val_test_split(background: pd.DataFrame, signal: pd.DataFrame, sample_size: int = 100_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.concat([background, signal])

    # Rename label and weight columns to more intuitive names
    data.rename(columns={
        'gen_label': 'is_signal',
        'gen_xsec': 'weight',
    }, inplace=True)

    # Map the is_signal column to binary values
    data['is_signal'] = data['is_signal'].map({'bkg': BKG, 'signal': SIG})

    # Drop useless columns and rows with NaNs
    data = data[[col for col in data.columns if 'gen' not in col]]
    data.dropna(axis=1, inplace=True)

    # Sample the data to get a balanced dataset
    sampled_bkg, sampled_sig = _sample_data(data, background, signal, sample_size)

    # Split the data into train, val and test datasets
    sig_train, sig_val, sig_test = _split_data(sampled_sig)
    bkg_train, bkg_val, bkg_test = _split_data(sampled_bkg)

    # Concatenate the signal and background datasets
    train = pd.concat([sig_train, bkg_train])
    val = pd.concat([sig_val, bkg_val])
    test = pd.concat([sig_test, bkg_test])

    return train, val, test


def normalize(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize the data to have zero mean and unit variance. Also normalize the weights."""

    # Separate the data into features, labels and weights
    x = data.drop(columns=['is_signal', 'weight'])
    y = data['is_signal']
    w = data['weight']

    # Move to numpy arrays
    y = y.to_numpy()
    w = w.to_numpy()

    # Normalize the weights
    w[y == BKG] = w[y == BKG] / np.sum(w[y == BKG]) * len(w) / 2
    w[y == SIG] = w[y == SIG] / np.sum(w[y == SIG]) * len(w) / 2

    # Normalize the features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    return x, y, w
