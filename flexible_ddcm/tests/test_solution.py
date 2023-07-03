"""Test all state space components."""
import yaml

import pandas as pd
import numpy as np

from src.model.state_space import create_state_space
from src.model.state_space import solve
from src.model.shared import pandas_dot
from src.model.rewards import calculate_rewards_state_choice_space
from src.model.example.input_functions import reward_function, transition_function


def test_continuation_values():
    params = pd.read_csv("src/model/example/params.csv").set_index(["category", "name"])
    model_options = yaml.safe_load(open("src/model/example/specification.yaml"))
    external_probabilities = pd.read_csv(
        "src/model/example/external_probabilities.csv"
    ).drop(columns=["Unnamed: 0"])

    continuation, choice_value_funcs, transitions = solve(
        params, model_options, transition_funcs, rewards
    )


def test_wage_rewards():
    """Here rewards are tested."""
    params = pd.read_csv("src/model/example/params.csv").set_index(["category", "name"])
    model_options = yaml.safe_load(open("src/model/example/specification.yaml"))
    external_probabilities = pd.read_csv(
        "src/model/example/external_probabilities.csv"
    ).drop(columns=["Unnamed: 0"])
    state_space = create_state_space(model_options)
    rewards_calculated = calculate_rewards_state_choice_space(
        state_space.state_choice_space, params, reward_function
    )

    # Iterate foreward
    periods = range(20, 55)
    df = pd.DataFrame()
    df["age"] = periods
    df["exp"] = df["age"] - 20
    df["parental_income"] = 2
    df["ability"] = 2
    df["constant"] = 1
    df["uni_dropout"] = 0

    rewards = pandas_dot(df, params.loc["nonpec_vocational"]) + np.exp(
        pandas_dot(df, params.loc["wage_vocational"])
    )
    full_utility = (
        rewards.values.reshape(35)
        * df["exp"].map(lambda x: params.loc[("discount", "discount")].iloc[0] ** x)
    ).sum()

    sc_point = state_space.state_choice_space_indexer[
        (20, "mbo4", "mbo4", 2, 2, "vocational_work")
    ]
    np.testing.assert_almost_equal(
        full_utility, rewards_calculated.loc[sc_point].iloc[0]
    )


def test_nonpec_rewards():
    params = pd.read_csv("src/model/example/params.csv").set_index(["category", "name"])

    model_options = yaml.safe_load(open("src/model/example/specification.yaml"))
