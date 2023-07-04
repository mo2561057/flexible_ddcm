"""Test inputs for rewards transitions and intermediate state creation."""
import numpy as np
import pandas as pd
import yaml

from flexible_ddcm.example.input_functions import reward_function_nonstandard
from flexible_ddcm.example.input_functions import transition_function_nonstandard
from flexible_ddcm.example.input_functions import \
    map_transition_to_state_choice_entries_nonstandard
from flexible_ddcm.state_space import create_state_space
from flexible_ddcm.shared import pandas_dot
from flexible_ddcm.rewards import calculate_rewards_state_choice_space


def test_lifetime_wage_rewards():
    """Here rewards are tested."""
    params = pd.read_csv("flexible_ddcm/tests/resources/params.csv").set_index(
        ["category", "name"]
    )
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml"))
    
    state_space = create_state_space(model_options)
    rewards_calculated = calculate_rewards_state_choice_space(
        state_space.state_choice_space, params, reward_function_nonstandard
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
    np.isclose(
        full_utility, rewards_calculated.loc[sc_point].iloc[0]
    )

def test_nonpec_rewards():
    pass

def test_poisson_length():
    pass

def test_poisson_length_with_dropout_risk():
    pass
