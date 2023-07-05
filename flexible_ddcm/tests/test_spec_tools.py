"""Test inputs for rewards transitions and intermediate state creation."""
import numpy as np
import pandas as pd
import yaml
import scipy
import math

from flexible_ddcm.example.input_functions import reward_function_nonstandard
from flexible_ddcm.example.input_functions import transition_function_nonstandard
from flexible_ddcm.example.input_functions import \
    map_transition_to_state_choice_entries_nonstandard
from flexible_ddcm.state_space import create_state_space
from flexible_ddcm.shared import pandas_dot
from flexible_ddcm.rewards import calculate_rewards_state_choice_space
from flexible_ddcm.model_spec_utils import combined_logit_length
from flexible_ddcm.model_spec_utils import _poisson_length

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
    params = pd.read_csv(
        "flexible_ddcm/tests/resources/params.csv").set_index(
        ["category", "name"]
    )
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml"))
    
    state_space = create_state_space(
        model_options)
    states = state_space.state_space[
        state_space.state_space.variable_key==0] 

    actual = _poisson_length(
        params.loc[f"transition_length_dropout_mbo4", "value"],
        states,
        int(params.loc[("transition_max", f"mbo4"), "value"]),
        int(params.loc[("transition_min", f"mbo4"), "value"]),
    )
    
    min_ = params.loc[(f"transition_min","mbo4"), "value"]
    max_ = params.loc[(f"transition_max","mbo4"), "value"]
    
    poisson_vars = params.loc[
        f"transition_length_dropout_mbo4"].index
    lambda_ = sum([states.loc[0,col]*params.loc[
        (f"transition_length_dropout_mbo4",col), "value"] for col in poisson_vars])
    
    def poisson(val,lambda_,min):
        val = val - min
        return (
            (lambda_**(val))*(math.e**(lambda_)))/(math.factorial(int(val)))
    
    dist = {
        val: poisson(val,lambda_, min_) for val in range(int(min_),int(max_)+1)}
    norm = sum(dist.values())

    dist = {key: value/norm for key, value in dist.items()}

    assert np.isclose(
        pd.Series(dist).values,actual.loc[0].values).all()

    


def test_logit():
    pass
