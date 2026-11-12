"""Script to test model extensions."""
import pandas as pd
import yaml

from flexible_ddcm.model_spec_utils import extreme_value_shocks
from flexible_ddcm.simulate import get_simulate_func
from flexible_ddcm.state_space import create_state_space
from flexible_ddcm.example.belief_updating.input_functions import (
    auxiliary_function_macedonia,
    create_next_period_df_macedonia,
    initial_states_macedonia,
    map_transition_to_state_choice_entries_macedonia,
    reward_function_macedonia,
    transition_function_macedonia,
)

params = pd.read_csv("src/flexible_ddcm/example/belief_updating/params_base.csv").set_index(
    ["category", "name"]
)["value"]

model_options = yaml.safe_load(
    open("src/flexible_ddcm/example/belief_updating/specification.yaml")
)


if __name__ == "__main__":
    simulate = get_simulate_func(
        params,
        model_options,
        transition_function_macedonia,
        reward_function_macedonia,
        extreme_value_shocks,
        map_transition_to_state_choice_entries_macedonia,
        initial_states_macedonia,
        auxiliary_function_macedonia,
        create_next_period_df=create_next_period_df_macedonia,
    )
    simulate_dict = simulate(params)
