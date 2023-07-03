import numpy as np
import pandas as pd
import yaml

from src.model.example.input_functions import reward_function
from src.model.example.input_functions import transition_function
from src.model.simulate import get_simulate_func
from src.model.transitions import build_transition_func_from_params
from src.model.state_space import create_state_space

def test_transition_shocks():
    """Test whether transition probabilities work."""
    params = pd.read_csv("src/tests/resources/params.csv").set_index(
        ["category", "name"]
    )
    model_options = yaml.safe_load(
        open("src/tests/resources/specification.yaml"))
    external_probabilities = pd.read_csv(
        "src/tests/resources/external_probabilities.csv"
    ).drop(columns=["Unnamed: 0"])
    state_space = create_state_space(model_options)
    

    simulate = get_simulate_func(
        model_options,
        transition_function,
        reward_function,
        external_probabilities,
    )

    simulate_dict = simulate(params)

    transitions = build_transition_func_from_params(
        params, state_space, transition_function
    )

    simulate_dict = simulate(params)

    # Groupby initial key and chec next key.

    simulated_transitions = pd.concat(
        [simulate_dict[0][["state_key", "choice"]], simulate_dict[1]][["state_key"]],
        axis=0,
    )


def test_simulate_func():
    params = pd.read_csv("src/model/example/params.csv").set_index(["category", "name"])
    # Set particular returns
    params.loc[("nonpec_mbo3", "constant"), "value"] = 1e10

    model_options = yaml.safe_load(open("src/model/example/specification.yaml"))

    external_probabilities = pd.read_csv(
        "src/model/example/external_probabilities.csv"
    ).drop(columns=["Unnamed: 0"])

    simulate_dict = simulate(
        model_options,
        params,
        transition_function,
        reward_function,
        external_probabilities,
    )

    assert all(simulate_dict[0].choice == "mbo3")
