"""solve_fast must reproduce solve, and simulations with fast_solve=True must match.

Uses a small self-contained model: age 0-3, a variable state x that moves
stochastically while in school, a fixed type, and a terminal work choice.
"""
import numpy as np
import pandas as pd
import pytest
from flexible_ddcm.model_spec_utils import extreme_value_shocks
from flexible_ddcm.simulate import get_simulate_func
from flexible_ddcm.solve import solve
from flexible_ddcm.solve_fast import _build_plan
from flexible_ddcm.solve_fast import solve_fast
from flexible_ddcm.state_space import create_state_space

X_GRID = [0, 1, 2]
X_TRANSITION = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])


def model_options():
    return {
        "state_space": {
            "age": {"type": "integer_grid", "lowest": 0, "highest": 4, "fixed": False},
            "x": {"type": "list", "list": X_GRID, "start": "random", "fixed": False},
            "type": {"type": "list", "list": [0, 1], "start": "random", "fixed": True},
        },
        "choices": {
            "terminal_choices": ["work"],
            "choice_sets": {"school": ["age<3"], "work": ["age<4"]},
        },
        "covariates": {"constant": "1"},
        "n_simulation_agents": 2_000,
        "seed": 123,
    }


def params():
    index = pd.MultiIndex.from_tuples(
        [("discount", "discount"), ("ev_shocks", "scale")], names=["category", "name"]
    )
    return pd.Series([0.9, 0.7], index=index)


def transition_function(state_space, choice, params):
    out = {}
    for vk, locs in state_space.state_space.groupby("variable_key").groups.items():
        if choice not in state_space.variable_key_to_choice_set[vk]:
            continue
        if choice == "work":
            out[(choice, vk)] = pd.DataFrame(1.0, index=locs, columns=["terminal"])
            continue
        age, x = state_space.variable_state_space.loc[vk, ["age", "x"]]
        arrivals = [
            state_space.variable_state_space_indexer[(age + 1, x_next)]
            for x_next in X_GRID
        ]
        probs = np.tile(X_TRANSITION[X_GRID.index(x)], (len(locs), 1))
        out[(choice, vk)] = pd.DataFrame(probs, index=locs, columns=arrivals)
    return out


def reward_function(state_choice_space, params):
    scs = state_choice_space
    value = np.where(
        scs["choice"] == "school",
        -1 + 0.5 * scs["x"] + scs["type"],
        2 * scs["x"] + 0.3 * scs["age"] - 0.5 * scs["type"],
    )
    return pd.DataFrame({"value": value}, index=scs.index)


def initial_states(model_options, params):
    rng = np.random.default_rng(0)
    n = model_options["n_simulation_agents"]
    return pd.DataFrame(
        {"age": 0, "x": rng.choice(X_GRID, n), "type": rng.integers(0, 2, n)}
    )


@pytest.fixture(scope="module")
def state_space():
    return create_state_space(model_options())


def test_solve_fast_equals_solve(state_space):
    args = (
        params(),
        model_options(),
        state_space,
        transition_function,
        reward_function,
    )
    cv_old, csv_old, _ = solve(*args, None)
    cv_new, csv_new, _ = solve_fast(*args)

    pd.testing.assert_index_equal(cv_old.index, cv_new.index)
    np.testing.assert_allclose(cv_new.values, cv_old.values, rtol=1e-12)
    assert csv_old.keys() == csv_new.keys()
    for key in csv_old:
        old = csv_old[key].astype(float).sort_index()
        new = csv_new[key].sort_index()
        assert list(old.columns) == list(new.columns)
        pd.testing.assert_index_equal(old.index, new.index)
        np.testing.assert_allclose(new.values, old.values, rtol=1e-12)


def test_solve_fast_terminal_age_is_logsum_of_work(state_space):
    """At the last age only work is available: value = reward of work."""
    cv, csv, _ = solve_fast(
        params(), model_options(), state_space, transition_function, reward_function
    )
    last = state_space.state_space[state_space.state_space["age"] == 3]
    expected = 2 * last["x"] + 0.3 * 3 - 0.5 * last["type"]
    np.testing.assert_allclose(cv.loc[last.index, "continuation_value"], expected)


def test_simulation_with_fast_solve_is_identical():
    def simulate(fast):
        return get_simulate_func(
            params(),
            model_options(),
            transition_function,
            reward_function,
            extreme_value_shocks,
            None,
            initial_states,
            # The default auxiliary_function has the wrong number of arguments.
            auxiliary_function=lambda ss, mo, p, transition, reward: (
                transition,
                reward,
            ),
            fast_solve=fast,
        )(params())

    old, new = simulate(False), simulate(True)
    assert old.keys() == new.keys()
    for period in old:
        pd.testing.assert_frame_equal(old[period], new[period], check_dtype=False)


def test_plan_rejects_probabilities_not_summing_to_one(state_space):
    transitions = {}
    for choice in ["school", "work"]:
        transitions.update(transition_function(state_space, choice, params()))
    key = next(k for k in transitions if k[0] == "school")
    transitions[key] = transitions[key] * 0.5
    with pytest.raises(ValueError, match="do not sum to one"):
        _build_plan(transitions, state_space)
