"""Tests for the test-scaler-refit case."""

import numpy as np
import pytest

from ml_failure_lab.cases import test_scaler_refit as case


def test_bad_approach_refits_scaler_and_is_rejected():
    result = case.report("bad")
    assert result["raw_test_values"] == [10.0, 12.0]
    assert result["applied_mean"] == pytest.approx(11.0)
    assert result["training_mean"] == pytest.approx(1.0)
    assert result["transformed_values"] == pytest.approx([-1.0, 1.0])
    assert result["expected_transformed_values"] == pytest.approx([9.0, 11.0])
    assert not result["reused_training_parameters"]
    assert result["predictions"] == [0, 0]
    assert result["expected_predictions"] == [1, 1]

    with pytest.raises(ValueError, match="parameters were mean=1.0 and scale=1.0"):
        case.verify("bad")


def test_fixed_approach_reuses_training_scaler_and_meets_requirement():
    result = case.report("fixed")
    assert result["raw_test_values"] == [10.0, 12.0]
    assert result["applied_mean"] == pytest.approx(1.0)
    assert result["training_mean"] == pytest.approx(1.0)
    assert result["transformed_values"] == pytest.approx([9.0, 11.0])
    assert result["expected_transformed_values"] == pytest.approx([9.0, 11.0])
    assert result["reused_training_parameters"]
    assert result["predictions"] == [1, 1]
    assert result["expected_predictions"] == [1, 1]

    case.verify("fixed")


def test_shifted_fixture_preserves_training_coordinate_system():
    # Second shifted test fixture [4.0, 6.0]
    result = case.report("fixed", test_data=[[4.0], [6.0]])
    assert result["raw_test_values"] == [4.0, 6.0]
    assert result["transformed_values"] == pytest.approx([3.0, 5.0])
    assert result["expected_transformed_values"] == pytest.approx([3.0, 5.0])
    assert result["reused_training_parameters"]
    case.require_training_scaler_parameters(result)


def test_invalid_input_dimensions_are_rejected():
    with pytest.raises(ValueError, match="two-dimensional array with one column"):
        case.report("fixed", test_data=[])
    with pytest.raises(ValueError, match="two-dimensional array with one column"):
        case.report("fixed", test_data=np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_transformed_values_mismatch_raises_in_require():
    invalid_result = {
        "reused_training_parameters": True,
        "transformed_values": [1.0, 2.0],
        "expected_transformed_values": [9.0, 11.0],
    }
    with pytest.raises(ValueError, match="do not match expected values"):
        case.require_training_scaler_parameters(invalid_result)
