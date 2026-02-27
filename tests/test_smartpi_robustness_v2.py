import logging
import pytest
import time
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from custom_components.versatile_thermostat.prop_algo_smartpi import SmartPI
from custom_components.versatile_thermostat.smartpi.learning import ABEstimator

def test_ols_slope_accuracy():
    """Test OLS slope estimation accuracy on regular data."""
    # Samples: (0, 0), (60, 0.05), ...
    # Expected slope: 0.05 deg/min = 0.000833 deg/s
    samples = []
    for i in range(6):
        samples.append((i * 60.0, float(i * 0.05)))

    slope_min, method, n = ABEstimator.robust_dTdt_per_min(samples)
    assert method == "ols"
    assert n == 6
    assert abs(slope_min - 0.05) < 0.02
