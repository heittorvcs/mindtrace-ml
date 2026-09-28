import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.agreement import agreement  # noqa: E402

MANUAL = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])


def test_close_measures_are_equivalent():
    automatic = MANUAL + np.array([0.5, -0.5, 0.3, -0.3, 0.4, -0.4])

    result = agreement(MANUAL, automatic, margin_fraction=0.15)

    assert result["tost_p"] < 0.05
    assert result["ccc"] > 0.99


def test_a_constant_bias_keeps_pearson_but_not_lin():
    # Correlação perfeita não é concordância: medir sempre 10 a mais correlaciona 1,0.
    result = agreement(MANUAL, MANUAL + 10)

    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["ccc"] < 0.9
    assert result["bias"] == pytest.approx(10.0)


def test_no_significant_difference_is_not_equivalence():
    # Diferenças grandes para os dois lados: o teste t não acusa nada, mas a
    # equivalência também não se sustenta.
    automatic = MANUAL + np.array([15.0, -15.0, 12.0, -12.0, 14.0, -14.0])

    result = agreement(MANUAL, automatic, margin_fraction=0.15)

    assert result["paired_t_p"] > 0.5
    assert result["tost_p"] > 0.05
