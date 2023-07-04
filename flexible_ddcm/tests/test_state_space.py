"""Test all state space components."""
import yaml

from flexible_ddcm.state_space import create_state_space


def create_test_state_space():
    model_options = yaml.safe_load(open("src/model/example/specification.yaml"))
    return create_state_space(model_options)


def test_state_space_mapper():
    model_options = yaml.safe_load(
        open("src/model/example/specification.yaml"))
    state_space = create_state_space(model_options)
    for (
        _,
        variable_key,
    ), next_state in state_space.state_and_next_variable_key_to_next_state.items():
        assert (
            state_space.state_space.loc[next_state, "variable_key"] == variable_key
        ).all()


def test_state_choice_space_creation():
    pass
