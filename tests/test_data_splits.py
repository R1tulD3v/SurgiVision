"""Tests for the deterministic dataset split."""
import pytest

from data_splits import train_val_test_indices


def test_partition_is_complete_and_disjoint():
    n = 41
    train, val, test = train_val_test_indices(n, seed=42)
    combined = sorted(train + val + test)
    assert combined == list(range(n))           # exhaustive
    assert len(set(train) & set(val)) == 0       # disjoint
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0


def test_deterministic_for_same_seed():
    assert train_val_test_indices(50, seed=7) == train_val_test_indices(50, seed=7)


def test_different_seed_changes_split():
    assert train_val_test_indices(50, seed=1) != train_val_test_indices(50, seed=2)


def test_fractions_respected_approximately():
    n = 100
    train, val, test = train_val_test_indices(n, seed=0, val_frac=0.2, test_frac=0.2)
    assert len(val) == 20
    assert len(test) == 20
    assert len(train) == 60


def test_empty_dataset():
    assert train_val_test_indices(0) == ([], [], [])


def test_tiny_dataset_keeps_train_nonempty():
    train, val, test = train_val_test_indices(1, seed=0)
    assert len(train) == 1
    assert val == [] and test == []


def test_sorted_output():
    train, val, test = train_val_test_indices(30, seed=3)
    for part in (train, val, test):
        assert part == sorted(part)


@pytest.mark.parametrize("val_frac,test_frac", [(0.6, 0.5), (1.0, 0.0), (-0.1, 0.2)])
def test_invalid_fractions_raise(val_frac, test_frac):
    with pytest.raises(ValueError):
        train_val_test_indices(100, val_frac=val_frac, test_frac=test_frac)
