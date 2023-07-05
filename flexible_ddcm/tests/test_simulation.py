import numpy as np
import pandas as pd
import yaml

from flexible_ddcm.example.input_functions import reward_function_nonstandard
from flexible_ddcm.example.input_functions import transition_function_nonstandard
from flexible_ddcm.example.input_functions import \
    map_transition_to_state_choice_entries_nonstandard
from flexible_ddcm.simulate import get_simulate_func
from flexible_ddcm.transitions import build_transition_func_from_params
from flexible_ddcm.state_space import create_state_space


def test_transition_shocks():
    """Test transition probabilities."""
    params = pd.read_csv("flexible_ddcm/tests/resources/params.csv").set_index(
        ["category", "name"]
    )
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml"))
    
    external_probabilities = pd.read_csv(
        "flexible_ddcm/tests/resources/external_probabilities.csv"
    ).drop(columns=["Unnamed: 0"])
    
    state_space = create_state_space(model_options)

    simulate = get_simulate_func(
        model_options,
        transition_function_nonstandard,
        reward_function_nonstandard,
        external_probabilities,
        map_transition_to_state_choice_entries_nonstandard
    )

    simulate_dict = simulate(params)

    transitions = build_transition_func_from_params(
        params,
        state_space,
        transition_function_nonstandard
    )
    
    # Groupby initial key and chec next key.
    simulated_transitions = pd.concat(
        [simulate_dict[0][["variable_key","choice","state_key"]],
         simulate_dict[1][["variable_key"]].rename(
        columns={"variable_key":"variable_key_next"})],axis=1)
    
    transitions_by_keys = simulated_transitions.groupby(
        ["variable_key","state_key","choice"]).variable_key_next.value_counts(
        normalize=True)
    
    transitions_predicted = transitions_by_keys.index.map(
        lambda ix: transitions[(ix[2],ix[0])].loc[ix[1],ix[3]])
    
    np.testing.assert_array_almost_equal(
        transitions_predicted.values,
        transitions_by_keys.values,
        decimal=1)
    
    


def test_simulate_func():
    params = pd.read_csv("flexible_ddcm/tests/resources/params.csv").set_index(
        ["category", "name"]
    )
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml"))
    
    external_probabilities = pd.read_csv(
        "flexible_ddcm/tests/resources/external_probabilities.csv"
    ).drop(columns=["Unnamed: 0"])
    
    # Set particular returns
    params.loc[("nonpec_mbo3", "constant"), "value"] = 1e10

    simulate = get_simulate_func(
        model_options,
        transition_function_nonstandard,
        reward_function_nonstandard,
        external_probabilities,
        map_transition_to_state_choice_entries_nonstandard
    )

    simulate_dict = simulate(params)

    assert all(simulate_dict[0].choice == "mbo3")
