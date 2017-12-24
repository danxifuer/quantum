from __future__ import absolute_import
from indicator import catch_errors
from indicator.function_helper import fill_for_noncomputable_vals
from six.moves import range


def momentum(data, period):
    """
    Momentum.

    Formula:
    DATA[i] - DATA[i - period]
    """
    catch_errors.check_for_period_error(data, period)

    momentum = [data[idx] - data[idx+1-period] for idx in range(period-1, len(data))]
    momentum = fill_for_noncomputable_vals(data, momentum)
    return momentum
