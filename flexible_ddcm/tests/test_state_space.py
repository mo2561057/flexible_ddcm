"""Test all state space components."""
import numpy as np
import yaml

from flexible_ddcm.state_space import create_state_space


def test_state_space_mapper():
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml")
    )
    state_space = create_state_space(model_options)
    for (
        _,
        variable_key,
    ), next_state in state_space.state_and_next_variable_key_to_next_state.items():
        assert (
            state_space.state_space.loc[next_state, "variable_key"] == variable_key
        ).all()


def test_segment_keys():
    model_options = yaml.safe_load(
        open("flexible_ddcm/tests/resources/specification.yaml")
    )
    state_space = create_state_space(model_options)

    predicted = state_space.variable_state_space.loc[
        state_space.state_space.variable_key
    ]

    actual = state_space.state_space[state_space.variable_state_space.columns]

    assert (actual.values == predicted.values).all(axis=1).all()
