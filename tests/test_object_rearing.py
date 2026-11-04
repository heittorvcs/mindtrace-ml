import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.behavior_model import compare, ignore_bouts, object_bouts  # noqa: E402

OBJECTS = pd.DataFrame([{"session_id": "TT_1_cam1", "object_id": 1, "center_x": 100.0, "center_y": 100.0,
                         "radius": 10.0}])


def nose(points):
    x, y = zip(*points)
    return pd.DataFrame({"nose_x": x, "nose_y": y})


def mask(n, *intervals):
    out = np.zeros(n, bool)
    for a, b in intervals:
        out[a:b + 1] = True
    return out


class TestObjectBouts:
    def test_nose_near_the_object_in_any_frame(self):
        # Focinho a 20 px do centro (10 da borda) num quadro do segundo episódio.
        points = [(200, 200)] * 10
        points[6] = (120, 100)
        marked = mask(10, (0, 2), (5, 8))

        assert object_bouts(marked, nose(points), OBJECTS) == [(5, 8)]

    def test_missing_nose_is_not_near(self):
        # Ponto ausente (-1, -1) não pode cair "perto" de um objeto no canto.
        corner = pd.DataFrame([{"center_x": 5.0, "center_y": 5.0, "radius": 10.0}])
        marked = mask(4, (0, 3))

        assert object_bouts(marked, nose([(-1, -1)] * 4), corner) == []

    def test_no_objects(self):
        assert object_bouts(mask(4, (0, 3)), nose([(100, 100)] * 4), OBJECTS.iloc[:0]) == []


class TestIgnoreBouts:
    def test_ignored_rearing_is_neither_missed_nor_false(self):
        marked = mask(30, (0, 5), (10, 15))
        predicted = mask(30, (1, 4), (12, 18), (25, 27))

        kept, shown = ignore_bouts(marked, predicted, [(10, 15)])
        result = compare(kept, shown, 30.0)

        assert result["manual_bouts"] == 1
        assert result["detected_bouts"] == 2      # a de 25–27 continua falsa
        assert result["matched_bouts"] == 1

    def test_inputs_are_not_changed(self):
        marked, predicted = mask(10, (0, 5)), mask(10, (2, 3))

        ignore_bouts(marked, predicted, [(0, 5)])

        assert marked.sum() == 6 and predicted.sum() == 2
