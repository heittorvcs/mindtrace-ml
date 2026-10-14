import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.behavior_model import STEP, triage_episodes  # noqa: E402
from mindtrace_ml.labels import mask_to_ranges, ranges_to_mask, seen_frames  # noqa: E402

FPS = 30.0


def scores_with(*bumps, n_frames=900, floor=0.02):
    """Notas por janela a partir de trechos (início, fim, nota) em quadros."""
    per_frame = np.full(n_frames, floor)
    for start, end, value in bumps:
        per_frame[start:end + 1] = value
    centers = np.arange(0, n_frames, STEP)
    return per_frame[centers], centers, n_frames


def classify(*bumps, cut=0.4, **kwargs):
    window, centers, n = scores_with(*bumps, **kwargs)
    found = triage_episodes(window, centers, n, np.zeros(n, bool), FPS, low=0.1, high=0.7, cut=cut)
    return [(round(peak, 2), accepted) for _, _, peak, accepted in found]


class TestTriageEpisodes:
    def test_each_score_band_gets_its_class(self):
        found = classify((60, 120, 0.9), (240, 300, 0.5), (420, 480, 0.2), (600, 660, 0.05))

        # Seguro entra aceito; acima ou abaixo do limiar de operação, duvidoso vai
        # para revisão; abaixo de 0,1 não aparece.
        assert found == [(0.9, True), (0.5, False), (0.2, False)]

    def test_operating_threshold_keeps_close_rearings_apart(self):
        # Dois rearings com uma descida de 1 s em que a nota fica em 0,2: acima do
        # limiar baixo, então recortar por ele juntaria os dois num episódio só.
        bumps = ((60, 120, 0.9), (121, 150, 0.2), (151, 210, 0.9))

        assert classify(*bumps) == [(0.9, True), (0.9, True)]
        assert classify(*bumps, cut=None) == [(0.9, True)]

    def test_weak_shoulder_of_a_found_episode_is_not_reviewed_again(self):
        # A rampa de nota em volta de um episódio achado não vira um duvidoso à parte.
        found = classify((40, 59, 0.2), (60, 120, 0.9), (121, 140, 0.2))

        assert found == [(0.9, True)]


class TestSeenFrames:
    def test_ranges_round_trip(self):
        mask = np.zeros(20, bool)
        mask[2:5] = mask[9] = mask[15:20] = True

        assert mask_to_ranges(mask) == "2-4;9-9;15-19"
        assert (ranges_to_mask("2-4;9-9;15-19", 20) == mask).all()

    def test_continuous_labeling_counts_everything_up_to_watched(self):
        progress = pd.DataFrame([{"watched_until": 9}])

        assert seen_frames(progress, 20).tolist() == [True] * 10 + [False] * 10

    def test_triage_counts_only_what_was_on_screen(self):
        progress = pd.DataFrame([{"watched_until": 18, "seen": "2-4;15-18"}])

        assert np.flatnonzero(seen_frames(progress, 20)).tolist() == [2, 3, 4, 15, 16, 17, 18]

    def test_rows_of_the_same_session_add_up(self):
        progress = pd.DataFrame([{"watched_until": 3, "seen": np.nan},
                                 {"watched_until": 18, "seen": "15-18"}])

        assert np.flatnonzero(seen_frames(progress, 20)).tolist() == [0, 1, 2, 3, 15, 16, 17, 18]
