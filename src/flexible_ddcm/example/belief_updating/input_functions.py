"""Build all the objects we need for the model.
This initializes all the required model objects.
Things partialed out in here are dependent on the particular struture of
the model currently estimated. Make sure to change all objects sccrodingly when
updating the model.
"""
import functools

import numpy as np
import pandas as pd
import yaml
from flexible_ddcm.model_spec_utils import reward_function

from flexible_ddcm.example.belief_updating.function_utils import (
    auxiliary_function,
    create_next_period_df,
    initial_expectation,
    perceived_transition_attendance_vec,
    reward_school_attendance,
    transition_function_actual_macedonia,
    transition_graduation,
    work_rewards,
    work_rewards_log_utility,
    work_transition_vectorized,
)


external_probabilities = pd.read_csv(
    "src/flexible_ddcm/example/belief_updating/external_probabilities.csv"
)


model_options = yaml.safe_load(
    open("src/flexible_ddcm/example/belief_updating/specification.yaml")
)

CHOICES = [
    "high_attendance",
    "medium_attendance",
    "low_attendance",
    "very_low_attendance",
    "work",
]

choice_reward_functions = {
    "very_low_attendance": functools.partial(
        reward_school_attendance, choice="very_low_attendance"
    ),
    "low_attendance": functools.partial(
        reward_school_attendance, choice="low_attendance"
    ),
    "medium_attendance": functools.partial(
        reward_school_attendance, choice="medium_attendance"
    ),
    "high_attendance": functools.partial(
        reward_school_attendance, choice="high_attendance"
    ),
    "work": functools.partial(work_rewards, discount_key=("discount", "discount")),
}


choice_reward_functions_variance = {
    "very_low_attendance": functools.partial(
        reward_school_attendance, choice="very_low_attendance"
    ),
    "low_attendance": functools.partial(
        reward_school_attendance, choice="low_attendance"
    ),
    "medium_attendance": functools.partial(
        reward_school_attendance, choice="medium_attendance"
    ),
    "high_attendance": functools.partial(
        reward_school_attendance, choice="high_attendance"
    ),
    "work": functools.partial(
        work_rewards_log_utility, discount_key=("discount", "discount")
    ),
}

choice_transition_functions_macedonia = {
    "work": work_transition_vectorized,
    "high_attendance": functools.partial(
        transition_function_actual_macedonia, choice="high_attendance"
    ),
    "medium_attendance": functools.partial(
        transition_function_actual_macedonia, choice="medium_attendance"
    ),
    "low_attendance": functools.partial(
        transition_function_actual_macedonia, choice="low_attendance"
    ),
    "very_low_attendance": functools.partial(
        transition_function_actual_macedonia, choice="very_low_attendance"
    ),
}

choice_perceived_transition_functions_macedonia = {
    "high_attendance": functools.partial(
        transition_function_actual_macedonia, choice="high_attendance"
    ),
    "medium_attendance": functools.partial(
        transition_function_actual_macedonia, choice="medium_attendance"
    ),
    "low_attendance": functools.partial(
        transition_function_actual_macedonia, choice="low_attendance"
    ),
    "very_low_attendance": functools.partial(
        transition_function_actual_macedonia, choice="very_low_attendance"
    ),
    "work": work_transition_vectorized,
}

transition_function_macedonia = {
    "subjective": choice_perceived_transition_functions_macedonia,
    "objective": choice_transition_functions_macedonia,
}

reward_function_macedonia = functools.partial(
    reward_function, choice_reward_functions=choice_reward_functions
)

unemp = model_options[
    "state_space"]["unemployment_rate"]["list"]
primary = model_options[
    "state_space"]["expected_wage_primary"]["list"]
secondary = model_options[
    "state_space"]["expected_wage_secondary"]["list"]


ix = pd.MultiIndex.from_product([primary, secondary, unemp], names=["expected_wage_primary", "expected_wage_secondary", "unemployment_rate"])
cols = pd.MultiIndex.from_product([primary, secondary], names=["expected_wage_primary", "expected_wage_secondary"])

PROBABILITY_DF = np.random.chisquare(1, size=27*9).reshape(27, 9)
PROBABILITY_DF = PROBABILITY_DF / PROBABILITY_DF.sum(axis=1)[:, np.newaxis]
PROBABILITY_DF = pd.DataFrame(PROBABILITY_DF, index=ix, columns=cols)


initial_states_macedonia = functools.partial(
   initial_expectation,
   external_probabilities=external_probabilities,
   probability_df=PROBABILITY_DF,
)


create_next_period_df_macedonia = functools.partial(
    create_next_period_df, probability_df=PROBABILITY_DF
)

info_transition = {
    "expectations": (
        ["expected_wage_primary", "expected_wage_secondary", "unemployment_rate"],
        ["expected_wage_primary", "expected_wage_secondary", "unemployment_rate"],
    ),
    "age": (["age"], ["age"]),
    "degree": (["degree"], ["degree"]),
    "lagged_choice": (
        ["lagged_choice"],
        ["lagged_choice"],
    ),
}

auxiliary_function_macedonia = functools.partial(
    auxiliary_function,
    transition_graduation=transition_graduation,
    perceived_transition_attendance_vec=perceived_transition_attendance_vec,
)


def map_transition_to_state_choice_entries_macedonia(
    initial, choice, arrival, state_space
):
    position_stop = len(state_space.states)
    # Add n states attribute to state space.
    initial_state = tuple(state_space.state_space.loc[initial])[:position_stop]
    return [state_space.state_choice_space_indexer[(*initial_state, choice)]]
