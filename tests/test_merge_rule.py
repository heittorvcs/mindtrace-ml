import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.behavior_model import bouts, close_gaps, gate, label_mask  # noqa: E402
from mindtrace_ml.labels import bouts_for  # noqa: E402

FPS = 30.0


def rearings(*intervals):
    return pd.DataFrame([{"behavior": "rearing", "start_frame": a, "end_frame": b, "field": 0}
                         for a, b in intervals])


class TestMergeRule:
    def test_brief_touch_down_does_not_end_the_rearing(self):
        # Desce por 0,4 s e sobe de novo: para o laboratório, um rearing só.
        labels = rearings((0, 59), (72, 120))

        mask = label_mask(labels, "rearing", np.arange(200), FPS)

        assert bouts(mask) == [(0, 120)]

    def test_a_real_pause_starts_a_new_rearing(self):
        # 0,7 s no chão: são dois rearings.
        labels = rearings((0, 59), (81, 120))

        mask = label_mask(labels, "rearing", np.arange(200), FPS)

        assert bouts(mask) == [(0, 59), (81, 120)]

    def test_detection_follows_the_same_rule_as_the_labels(self):
        detected = np.zeros(200, bool)
        detected[0:60] = True
        detected[72:121] = True
        detected[150:180] = True                    # 1 s depois: outro rearing

        merged = gate("rearing", detected, np.zeros(200, bool), FPS)

        assert bouts(merged) == [(0, 120), (150, 179)]

    def test_other_behaviors_are_not_merged(self):
        labels = pd.DataFrame([{"behavior": "grooming", "start_frame": a, "end_frame": b, "field": 0}
                               for a, b in ((0, 59), (72, 120))])

        assert len(bouts(label_mask(labels, "grooming", np.arange(200), FPS))) == 2

    def test_gap_zero_keeps_the_old_behavior(self):
        labels = rearings((0, 10), (11, 20), (30, 40))

        assert bouts_for(labels, "rearing") == [(0, 20), (30, 40)]
        assert close_gaps(np.array([1, 0, 1], bool), 0).tolist() == [True, False, True]
