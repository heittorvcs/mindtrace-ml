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


def test_total_error_splits_mse_into_bias_and_spread():
    from mindtrace_ml.agreement import total_error
    manual = np.array([10.0, 20.0, 30.0, 40.0])
    automatic = manual + np.array([2.0, 2.0, 2.0, 2.0])     # erra sempre +2: só viés

    result = total_error(manual, automatic)

    assert result["rmse"] == pytest.approx(2.0)
    assert result["bias_share"] == pytest.approx(1.0)
    assert result["mse"] == pytest.approx(result["bias"] ** 2 + result["spread"])


def test_total_error_spread_without_bias_and_the_session_that_weighs_most():
    from mindtrace_ml.agreement import total_error
    manual = np.array([10.0, 10.0, 0.0])
    automatic = np.array([12.0, 8.0, 0.0])                   # +2 e -2: média certa, espalhamento

    result = total_error(manual, automatic, sessions=["a", "b", "c"])

    assert result["bias"] == pytest.approx(0.0)
    assert result["bias_share"] == pytest.approx(0.0)
    # Sessão sem episódios marcados não divide por zero no erro relativo.
    assert result["relative_median"] == pytest.approx(0.2)
    assert result["worst_session"] in {"a", "b"} and result["worst_share"] == pytest.approx(0.5)
