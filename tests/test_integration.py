"""Minimal integration test: synthetic ConfigSpace → full pipeline."""

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Float, Categorical

from hpi_for_mo.Data_preparation import RunGroup, Objective, encode_and_normalize
from hpi_for_mo.fanova import do_weighted_fanova_mo
from hpi_for_mo.ablation import do_weighted_ablation_mo
from hpi_for_mo.weighting import get_weightings


def _build_synthetic_group(n_configs=20, seed=42):
    """Create a RunGroup from a synthetic ConfigSpace with random data."""
    rng = np.random.RandomState(seed)

    cs = ConfigurationSpace(seed=seed)
    cs.add(Float("lr", bounds=(1e-4, 1.0), default=0.1, log=True))
    cs.add(Float("dropout", bounds=(0.0, 0.5), default=0.1))
    cs.add(Float("weight_decay", bounds=(1e-6, 1e-2), default=1e-4, log=True))
    cs.add(Categorical("optimizer", items=["sgd", "adam", "rmsprop"], default="adam"))

    configs = {}
    data = []
    for i in range(1, n_configs + 1):
        cfg = cs.sample_configuration()
        configs[str(i)] = dict(cfg)
        # Two objectives: error (lower is better) and time (lower is better)
        error = rng.uniform(0.05, 0.95)
        time_cost = rng.uniform(1.0, 100.0)
        data.append({
            "config_id": i,
            "cost": [error, time_cost],
            "status": 1,
        })

    return RunGroup(
        configspace=cs,
        configs=configs,
        data=data,
        objective_names=["error", "time"],
    )


def test_encode_config():
    group = _build_synthetic_group()
    cfg = group.get_config(1)
    encoded = group.encode_config(cfg, specific=True)
    assert len(encoded) == len(group.configspace.get_hyperparameter_names())
    assert all(isinstance(v, float) for v in encoded)


def test_encode_and_normalize():
    group = _build_synthetic_group()
    objectives = group.get_objectives()
    df = encode_and_normalize(group, objectives)

    hp_names = group.configspace.get_hyperparameter_names()
    for hp in hp_names:
        assert hp in df.columns
    for o in objectives:
        assert o.name in df.columns
        assert o.name + "_normed" in df.columns
    assert "config_id" in df.columns
    assert len(df) == 20


def test_full_pipeline():
    group = _build_synthetic_group()
    objectives = group.get_objectives()
    objectives_normed = [o.name + "_normed" for o in objectives]
    df = encode_and_normalize(group, objectives)

    # Weightings from Pareto front
    weightings = get_weightings(objectives_normed, df)
    assert weightings.shape[1] == 2
    assert len(weightings) >= 1
    # Weights should sum to ~1 per row
    for row in weightings:
        assert abs(sum(row) - 1.0) < 1e-6

    # fANOVA
    df_fanova = do_weighted_fanova_mo(group, df, objectives_normed, weightings)
    assert "hp_name" in df_fanova.columns
    assert "fanova" in df_fanova.columns
    assert "variance" in df_fanova.columns
    assert any(col.startswith("weight_for_") for col in df_fanova.columns)

    # Ablation
    df_ablated, default = do_weighted_ablation_mo(group, df, objectives_normed, weightings)
    assert "hp_name" in df_ablated.columns
    assert "ablation" in df_ablated.columns
    assert "variance" in df_ablated.columns
    assert "new_performance" in df_ablated.columns
    assert any(col.startswith("weight_for_") for col in df_ablated.columns)
    assert len(default) == 2  # two objectives

    print("Full pipeline passed!")


if __name__ == "__main__":
    test_encode_config()
    test_encode_and_normalize()
    test_full_pipeline()
    print("All tests passed!")
