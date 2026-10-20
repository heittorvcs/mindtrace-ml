import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.behavior_model import dwell_signals, matched_episodes, run_seconds  # noqa: E402

FPS = 30.0


class TestDwellSignals:
    def test_run_counts_seconds_and_resets(self):
        flag = np.array([True] * 60 + [False] + [True] * 30)

        out = run_seconds(flag, FPS)

        assert out[59] == 2.0 and out[60] == 0.0 and out[90] == 1.0

    def test_run_is_capped(self):
        assert run_seconds(np.ones(int(100 * FPS), bool), FPS, cap=60).max() == 60

    def test_still_rat_at_the_wall_accumulates_both(self):
        # Parado (2 px/s) e encostado (5 px da parede) por 10 s.
        signals = pd.DataFrame({"body_speed": np.full(300, 2.0), "wall_distance": np.full(300, 5.0)})

        out = dwell_signals(signals, FPS)

        assert out.still_sec.iloc[-1] == 10.0 and out.wall_sec.iloc[-1] == 10.0

    def test_walking_resets_still_but_not_wall(self):
        speed = np.r_[np.full(150, 2.0), np.full(30, 40.0), np.full(120, 2.0)]
        signals = pd.DataFrame({"body_speed": speed, "wall_distance": np.full(300, 5.0)})

        out = dwell_signals(signals, FPS)

        assert out.still_sec.iloc[-1] < 5.0 and out.wall_sec.iloc[-1] == 10.0

    def test_missing_pose_does_not_interrupt(self):
        speed = np.r_[np.full(100, 2.0), np.full(20, np.nan), np.full(100, 2.0)]
        signals = pd.DataFrame({"body_speed": speed, "wall_distance": np.full(220, 5.0)})

        assert dwell_signals(signals, FPS).still_sec.iloc[-1] > 7.0


class TestMatchedEpisodes:
    def test_one_detection_per_rearing(self):
        assert matched_episodes([(10, 20), (40, 50)], [(12, 18), (41, 49)]) == 2

    def test_merging_two_rearings_counts_once(self):
        # Uma janela longa sobre dois rearings: por sobreposição seriam 2 achados.
        assert matched_episodes([(10, 20), (30, 40)], [(5, 45)]) == 1

    def test_splitting_one_rearing_counts_once(self):
        assert matched_episodes([(10, 40)], [(10, 20), (25, 40)]) == 1

    def test_no_overlap_no_pair(self):
        assert matched_episodes([(10, 20)], [(21, 30)]) == 0
