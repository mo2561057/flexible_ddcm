"""Tests for flexible_ddcm.fast on a small yearly toy model.

Students choose each year between school and (terminal) work. A school year is
passed with a probability that depends on the unobserved `skill`; agents
believe the pass probability is the population average (subjective), the
simulation uses the true one (objective).
"""
import functools

import numpy as np
import pandas as pd
import pytest
from flexible_ddcm.fast import build_transitions_fast
from flexible_ddcm.fast import simulate_fast
from flexible_ddcm.fast import solution_to_legacy
from flexible_ddcm.fast import solve_fast
from flexible_ddcm.model_spec_utils import extreme_value_shocks
from flexible_ddcm.shared import build_covariates
from flexible_ddcm.solve import solve
from flexible_ddcm.state_space import create_state_space

MODEL_OPTIONS = {
    "state_space": {
        "age": {"type": "integer_grid", "start": 0, "lowest": 0, "highest": 5,
                "fixed": False},
        "years": {"type": "list", "start": 0, "list": [0, 1, 2, 3, 4],
                  "fixed": False},
        "lagged_choice": {"type": "list", "start": "school",
                          "list": ["school", "work"], "fixed": False},
        "income": {"type": "list", "start": "random_external", "list": [0, 1],
                   "fixed": True},
        "skill": {"type": "list", "start": "random_external", "list": [0, 1],
                  "fixed": True},
    },
    "choices": {
        "terminal_choices": ["work"],
        "choice_sets": {"school": ["age<4"], "work": ["age<5"]},
    },
    "covariates": {"constant": 1, "high_income": "income == 1"},
    "state_space_filter": ["years > age", "lagged_choice == 'work'"],
    "n_simulation_agents": 20_000,
    "seed": 123,
}

PARAMS = pd.Series(
    {
        ("discount", "discount"): 0.95,
        ("ev_shocks", "scale"): 2.0,
        ("pass", "constant"): 0.2,
        ("pass", "skill"): 1.5,
        ("school", "constant"): -1.0,
        ("school", "high_income"): 0.5,
        ("work", "constant"): 2.0,
        ("work", "years"): 1.2,
    }
)


def _pass_prob(states, params, subjective):
    skill = 0.5 if subjective else states["skill"]
    return 1 / (1 + np.exp(-(params[("pass", "constant")] + params[("pass", "skill")] * skill)))


def _next_keys(state_space, variable_key):
    age, years, _ = tuple(state_space.variable_state_space.loc[variable_key])
    index = state_space.variable_state_space_indexer
    return index[(age + 1, years + 1, "school")], index[(age + 1, years, "school")]


def transition_legacy(state_space, choice, params, subjective):
    out = {}
    ss = state_space.state_space
    for variable_key, locs in ss.groupby("variable_key").groups.items():
        if choice not in state_space.variable_key_to_choice_set[variable_key]:
            continue
        if choice == "work":
            out[(choice, variable_key)] = pd.DataFrame(1.0, index=locs, columns=["terminal"])
            continue
        p = _pass_prob(ss.loc[locs], params, subjective)
        passed, failed = _next_keys(state_space, variable_key)
        out[(choice, variable_key)] = pd.DataFrame({passed: p, failed: 1 - p}, index=locs)
    return out


def transition_long(state_space, choice, params, subjective):
    legacy = transition_legacy(state_space, choice, params, subjective)
    rows = []
    for trans in legacy.values():
        for col in trans.columns:
            rows.append(pd.DataFrame({
                "state": trans.index,
                "next_variable_key": -1 if col == "terminal" else col,
                "prob": trans[col].to_numpy(),
            }))
    return pd.concat(rows, ignore_index=True)


def reward_function(state_choice_space, params):
    school = state_choice_space.choice == "school"
    value = np.where(
        school,
        params[("school", "constant")]
        + params[("school", "high_income")] * state_choice_space.high_income,
        params[("work", "constant")] + params[("work", "years")] * state_choice_space.years,
    )
    return pd.DataFrame({"value": value}, index=state_choice_space.index)


def initial_states(model_options, params):
    rng = np.random.default_rng(0)
    n = model_options["n_simulation_agents"]
    out = pd.DataFrame({"age": 0, "years": 0, "lagged_choice": "school",
                        "income": rng.integers(0, 2, n), "skill": rng.integers(0, 2, n),
                        "region": rng.integers(0, 3, n)})
    return build_covariates(out, model_options["covariates"])


@pytest.fixture(scope="module")
def state_space():
    return create_state_space(MODEL_OPTIONS)


@pytest.mark.parametrize("transition", [transition_legacy, transition_long])
def test_fast_solution_equals_legacy_solve(state_space, transition):
    legacy_trans = functools.partial(transition_legacy, subjective=True)
    _, expected, _ = solve(PARAMS, MODEL_OPTIONS, state_space, legacy_trans,
                           reward_function, None)
    solution = solve_fast(PARAMS, MODEL_OPTIONS, state_space,
                          functools.partial(transition, subjective=True), reward_function)
    _, got = solution_to_legacy(solution, state_space)
    for key in expected:
        np.testing.assert_allclose(
            got[key].values, expected[key].loc[got[key].index, got[key].columns].values.astype(float)
        )


def test_invariant_states_give_same_solution(state_space):
    trans = functools.partial(transition_long, subjective=True)
    full = solve_fast(PARAMS, MODEL_OPTIONS, state_space, trans, reward_function)
    reduced = solve_fast(PARAMS, MODEL_OPTIONS, state_space, trans, reward_function,
                         invariant_states=["skill"])
    np.testing.assert_allclose(full.values, reduced.values)
    np.testing.assert_allclose(full.continuation_values, reduced.continuation_values)


def test_simulated_transitions_follow_objective_probabilities(state_space):
    transitions = {
        "subjective": functools.partial(transition_long, subjective=True),
        "objective": functools.partial(transition_long, subjective=False),
    }
    sim = simulate_fast(PARAMS, state_space, MODEL_OPTIONS, transitions, reward_function,
                        extreme_value_shocks, initial_states, invariant_states=["skill"])
    first = sim[0]
    in_school = first[first.choice == "school"].index
    passed = sim[1].loc[in_school, "years"] == 1
    for skill in [0, 1]:
        share = passed[first.loc[in_school, "skill"] == skill].mean()
        expected = _pass_prob(pd.DataFrame({"skill": [skill]}), PARAMS, False).iloc[0]
        assert abs(share - expected) < 0.02
    # Non-state columns of the initial states are carried along.
    assert (sim[max(sim)]["region"] == first["region"]).all()
    # Every agent ends in a terminal choice.
    assert sim[max(sim)]["choice"].eq("work").all()


def test_transitions_must_sum_to_one(state_space):
    def broken(state_space, choice, params):
        out = transition_long(state_space, choice, params, subjective=True)
        out["prob"] *= 0.9
        return out

    with pytest.raises(ValueError, match="sum"):
        build_transitions_fast(PARAMS, MODEL_OPTIONS, state_space, broken)
