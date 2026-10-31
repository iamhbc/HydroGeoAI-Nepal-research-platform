import pytest

from hydrogeoai.evaluation.metrics import reliability_curve


@pytest.mark.parametrize("probability", [0.0, 0.25, 1.0])
def test_constant_forecasts_produce_one_populated_bin(probability):
    result = reliability_curve([0, 1, 0, 0], [probability] * 4)
    assert result == {
        "mean_predicted": [probability],
        "observed_frequency": [0.25],
        "count": [4],
    }


def test_tied_forecasts_preserve_all_observations():
    result = reliability_curve([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9])
    assert sum(result["count"]) == 4
    assert result["mean_predicted"] == pytest.approx([0.1, 0.9])
    assert result["observed_frequency"] == [0.0, 1.0]
