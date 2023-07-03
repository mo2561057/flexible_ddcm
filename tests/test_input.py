"""Tests for own input files."""
import os
import yaml

import pandas as pd
import numpy as np

from src.model.example.input_functions import combined_logit_length
from src.model.state_space import create_state_space


def test_logit_length():
    """Check exact changes."""
    params = {
        ("risk_1", "constant"): 1,
        ("risk_1", "age"): 1,
        ("risk_1", "grade"): 1,
        ("length", "constant"): 1,
        ("length", "age"): 1,
        ("length", "grade"): 1,
        ("length_dropout", "constant"): 1,
        ("length_dropout", "age"): 1,
        ("length_dropout", "grade"): 1,
    }


def test_choice_sets():
    """Unit test for choice sets."""
    model_options = yaml.safe_load(
        "src/model/examples/specification.yaml")
    state_space = create_state_space(model_options)
