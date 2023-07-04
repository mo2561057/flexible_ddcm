import numpy as np
import pandas as pd
import yaml

from flexible_ddcm.example.input_functions import reward_function_nonstandard
from flexible_ddcm.example.input_functions import transition_function_nonstandard
from flexible_ddcm.example.input_functions import \
    map_transition_to_state_choice_entries_nonstandard
from flexible_ddcm.solve  import solve
from flexible_ddcm.state_space import create_state_space
from flexible_ddcm.shared import pandas_dot
from flexible_ddcm.rewards import calculate_rewards_state_choice_space


def test_continuation_values_transition():
    params = pd.read_csv("src/model/example/params.csv").set_index(["category", "name"])
    model_options = yaml.safe_load(open("src/model/example/specification.yaml"))
    state_space = create_state_space(model_options)

    continuation, choice_value_funcs, transitions = solve(
        params,
        model_options,
        transition_function_nonstandard,
        reward_function_nonstandard,
        map_transition_to_state_choice_entries_nonstandard)
    
    choice_value_funcs[7].loc[0,"havo"]

    # Manually calculate
    transition_check = transitions[("havo",0)]

    # Get raw continuation: 
    next_keys = {
        col:state_space.state_and_next_variable_key_to_next_state[(0,col)]\
            for col in transition_check.columns}
    cont_predicted = {
        col:continuation.loc[value]*params.loc[("discount","discount"),"value"]**(
        state_space.state_space.loc[value,"age"]-16) for col,value in next_keys.items()}
    
    continuation_predicted_weighted = sum(
        [cont_predicted[col]*transition_check.loc[0,col] for col in cont_predicted]).iloc[0]

    assert np.isclose(
        continuation_predicted_weighted,
        choice_value_funcs[7].loc[0,"havo"])
