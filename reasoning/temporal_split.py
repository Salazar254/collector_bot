"""Temporal-split utilities.

Rules enforced:
- Splits are time-ordered: train < val < test (no timestamp overlap).
- Month indices are 1-based relative to the dataset's earliest timestamp.
- verify_no_leakage() raises AssertionError on any overlap — call it after
  every split and after any data augmentation.
- Random splits are NEVER produced by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar("T")

_SECONDS_PER_MONTH = 30 * 24 * 3600  # 2 592 000 s; fixed calendar approximation


@dataclass
class TemporalSplit:
    train: list[Any]
    val: list[Any]
    test: list[Any]


def temporal_split(
    records: Sequence[T],
    get_timestamp: Callable[[T], float],
    train_range: tuple[int, int] = (1, 18),
    val_range: tuple[int, int] = (19, 21),
    test_range: tuple[int, int] = (22, 24),
) -> TemporalSplit:
    """Assign records to train/val/test by 1-based month index.

    Month 1 starts at min(timestamps). Anything outside all three ranges is
    silently discarded (e.g. a gap month).
    """
    if not records:
        return TemporalSplit([], [], [])

    t_min = min(get_timestamp(r) for r in records)

    train: list[T] = []
    val: list[T] = []
    test: list[T] = []

    for r in records:
        month = int((get_timestamp(r) - t_min) / _SECONDS_PER_MONTH) + 1
        if train_range[0] <= month <= train_range[1]:
            train.append(r)
        elif val_range[0] <= month <= val_range[1]:
            val.append(r)
        elif test_range[0] <= month <= test_range[1]:
            test.append(r)

    return TemporalSplit(train=train, val=val, test=test)


def verify_no_leakage(
    split: TemporalSplit,
    get_timestamp: Callable[[Any], float],
) -> None:
    """Raise AssertionError if any split boundary leaks future data into past.

    Checks: max(train) < min(val)  and  max(val) < min(test).
    Safe to call on empty splits (skips the check).
    """
    if not (split.train and split.val):
        return
    train_max = max(get_timestamp(r) for r in split.train)
    val_min = min(get_timestamp(r) for r in split.val)
    assert train_max < val_min, (
        f"Leakage: train max ts {train_max:.3f} >= val min ts {val_min:.3f}"
    )
    if not split.test:
        return
    val_max = max(get_timestamp(r) for r in split.val)
    test_min = min(get_timestamp(r) for r in split.test)
    assert val_max < test_min, (
        f"Leakage: val max ts {val_max:.3f} >= test min ts {test_min:.3f}"
    )


def walk_forward_splits(
    records: Sequence[T],
    get_timestamp: Callable[[T], float],
    initial_train_months: int = 18,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
    total_months: int = 24,
) -> list[TemporalSplit]:
    """Generate walk-forward cross-validation folds.

    Each fold shifts the train/val/test window forward by step_months.
    All folds are verified to have no leakage before returning.

    Example with defaults (months):
        Fold 0: train 1-18, val 19-21, test 22-24
        Fold 1: train 1-21, val 22-24  (test would need 25-27, skipped)
    """
    splits: list[TemporalSplit] = []
    train_end = initial_train_months

    while True:
        val_start = train_end + 1
        val_end = train_end + val_months
        test_start = val_end + 1
        test_end = val_end + test_months

        if test_end > total_months:
            break

        s = temporal_split(
            records,
            get_timestamp,
            train_range=(1, train_end),
            val_range=(val_start, val_end),
            test_range=(test_start, test_end),
        )
        verify_no_leakage(s, get_timestamp)
        splits.append(s)
        train_end += step_months

    return splits
