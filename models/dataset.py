"""Time-ordered train/val/test split.

Never use sklearn's `train_test_split(..., shuffle=True)` on this data - the
whole point is that the model must not see the future during training. All
splits here preserve chronological order: PAST -> TRAIN -> VALIDATION ->
TEST -> FUTURE.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_by_date(
    dataset: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> DatasetSplit:
    """Split by explicit cutoff dates, e.g.:
        split_by_date(df, train_end="2023-01-01", val_end="2024-01-01")
    -> train = [..., 2023-01-01), val = [2023-01-01, 2024-01-01), test = [2024-01-01, ...]
    """
    train = dataset[dataset.index < train_end]
    val = dataset[(dataset.index >= train_end) & (dataset.index < val_end)]
    test = dataset[dataset.index >= val_end]
    return DatasetSplit(train=train, val=val, test=test)


def split_by_ratio(
    dataset: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> DatasetSplit:
    """Split by proportion while keeping chronological order (no shuffling)."""
    n = len(dataset)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return DatasetSplit(
        train=dataset.iloc[:train_end],
        val=dataset.iloc[train_end:val_end],
        test=dataset.iloc[val_end:],
    )


def walk_forward_windows(
    dataset: pd.DataFrame,
    train_years: float,
    val_months: float,
    test_months: float,
    step_months: float,
):
    """Yield successive (train, val, test) DatasetSplit windows that slide
    forward in time - see backtest/walk_forward.py for how this is used to
    validate that a strategy stays robust out-of-sample, not just on one
    lucky backtest window."""
    start = dataset.index.min()
    end = dataset.index.max()

    train_delta = pd.DateOffset(years=train_years) if train_years == int(train_years) else pd.DateOffset(
        months=int(train_years * 12)
    )
    val_delta = pd.DateOffset(months=val_months)
    test_delta = pd.DateOffset(months=test_months)
    step_delta = pd.DateOffset(months=step_months)

    window_start = start
    while True:
        train_end = window_start + train_delta
        val_end = train_end + val_delta
        test_end = val_end + test_delta
        if test_end > end:
            break

        yield DatasetSplit(
            train=dataset[(dataset.index >= window_start) & (dataset.index < train_end)],
            val=dataset[(dataset.index >= train_end) & (dataset.index < val_end)],
            test=dataset[(dataset.index >= val_end) & (dataset.index < test_end)],
        )
        window_start = window_start + step_delta
