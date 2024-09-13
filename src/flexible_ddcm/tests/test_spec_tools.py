"""Test inputs for rewards transitions and intermediate state creation."""
import math

import numpy as np
import pandas as pd
import scipy
import yaml

from src.flexible_ddcm.example.base.input_functions import (
    map_transition_to_state_choice_entries_nonstandard,
)
from src.flexible_ddcm.example.base.input_functions import reward_function_nonstandard
from src.flexible_ddcm.example.base.input_functions import transition_function_nonstandard
from src.flexible_ddcm.model_spec_utils import _poisson_length
from src.flexible_ddcm.rewards import calculate_rewards_state_choice_space
from src.flexible_ddcm.shared import pandas_dot
from src.flexible_ddcm.state_space import create_state_space


def test_lifetime_wage_rewards():

    params = pd.read_csv("flexible_ddcm/tests/resources/params.csv").set_index(
        ["category", "name"]
    )["value"]
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml")
    )

    state_space = create_state_space(model_options)
    rewards_calculated = calculate_rewards_state_choice_space(
        state_space.state_choice_space, params, reward_function_nonstandard
    )

    # Iterate foreward
    periods = range(20, 40)
    df = pd.DataFrame()
    df["age"] = periods
    df["exp"] = df["age"] - 20
    df["parental_income_1"] = 0
    df["grade_1"] = 0
    df["constant"] = 1
    df["uni_dropout"] = 0

    log_wages = pandas_dot(df, params.loc["wage_vocational"])

    log_wages[log_wages > 4] = 4
    work_utility = pandas_dot(df, params.loc["nonpec_vocational"])

    std = params.loc[("wage_shock_vocational", "std")]
    discount = params.loc[("discount", "discount")]

    calculated = (np.exp(log_wages + (std) ** 2 / 2) * (1500) + work_utility) * (
        df.exp.map(lambda x: discount**x)
    )

    sc_point = state_space.state_choice_space_indexer[
        (20, "mbo4", "mbo4", 0, 0, "vocational_work")
    ]
    np.isclose(calculated.sum(), rewards_calculated.loc[sc_point])


def test_nonpec_rewards():
    pass


def test_poisson_length():
    params = pd.read_csv("flexible_ddcm/tests/resources/params.csv").set_index(
        ["category", "name"]
    )["value"]
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml")
    )

    state_space = create_state_space(model_options)
    states = state_space.state_space[state_space.state_space.variable_key == 0]

    actual = _poisson_length(
        params.loc[f"transition_length_mbo4"],
        states,
        int(params.loc[("transition_max", f"mbo4")]),
        int(params.loc[("transition_min", f"mbo4")]),
    )
    min_ = params.loc[(f"transition_min", "mbo4")]
    max_ = params.loc[(f"transition_max", "mbo4")]

    poisson_vars = params.loc[f"transition_length_mbo4"].index
    lambda_ = sum(
        states.loc[0, col] * params.loc[(f"transition_length_mbo4", col)]
        for col in poisson_vars
    )

    def poisson(val, lambda_, min):
        val = val - min
        return ((lambda_ ** (val)) * (math.e ** (lambda_))) / (math.factorial(int(val)))

    dist = {val: poisson(val, lambda_, min_) for val in range(int(min_), int(max_) + 1)}
    norm = sum(dist.values())

    dist = {key: value / norm for key, value in dist.items()}

    assert np.isclose(pd.Series(dist).values, actual.loc[0].values).all()


def test_logit():
    pass
