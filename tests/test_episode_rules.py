import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.behavior_model import bouts, gate, join_episodes, label_mask  # noqa: E402

FPS = 30.0


def labels(*intervals):
    return pd.DataFrame([{"session_id": "HD2_F7_cam1", "behavior": "rearing", "start_frame": a, "end_frame": b}
                         for a, b in intervals])


def mask(n, *intervals):
    out = np.zeros(n, bool)
    for a, b in intervals:
        out[a:b + 1] = True
    return out


class TestPairRule:
    def test_a_short_dip_joins_at_most_two_rearings(self):
        # Três subidas de 2 s com descidas de 0,43 s: as duas primeiras juntam, a terceira abre outro.
        marked = mask(300, (0, 59), (72, 131), (144, 203))

        assert bouts(join_episodes(marked, "rearing", FPS, "par")) == [(0, 131), (144, 203)]
        assert bouts(join_episodes(marked, "rearing", FPS, "atual")) == [(0, 203)]

    def test_dip_of_0_6_s_does_not_join(self):
        marked = mask(200, (0, 59), (78, 137))

        assert len(bouts(join_episodes(marked, "rearing", FPS, "par"))) == 2

    def test_short_rearing_counts_only_in_the_current_rule(self):
        own = labels((0, 29), (100, 159))           # 1 s e 2 s
        frames = np.arange(300)

        assert bouts(label_mask(own, "rearing", frames, FPS, "par")) == [(100, 159)]
        assert bouts(label_mask(own, "rearing", frames, FPS, "par", count=False)) == [(0, 29), (100, 159)]
        assert len(bouts(label_mask(own, "rearing", frames, FPS))) == 2

    def test_detection_uses_the_same_rule(self):
        detected = mask(300, (0, 29), (100, 159))
        still = np.zeros(300, bool)

        assert bouts(gate("rearing", detected, still, FPS, "par")) == [(100, 159)]
        assert len(bouts(gate("rearing", detected, still, FPS))) == 2

    def test_grooming_ignores_the_rule(self):
        detected = mask(300, (0, 29), (100, 159))
        still = np.zeros(300, bool)

        assert bouts(gate("grooming", detected, still, FPS, "par")) == bouts(gate("grooming", detected, still, FPS))
