import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.detectors import triage
from mindtrace_ml.kinematics import frame_kinematics
from mindtrace_ml.triage_model import (
    KEYPOINTS,
    SPEC,
    animal_of,
    clip_features,
    frame_scores,
    frame_signals,
    label_sets,
    sliding_features,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_detectors import FPS, make_pose  # noqa: E402

OBJECTS = pd.DataFrame({"session_id": ["s"], "object_id": [1],
                        "center_x": [300.0], "center_y": [200.0], "radius": [15.0]})


def signals_for(pose):
    kinematics = frame_kinematics(pose, KEYPOINTS, FPS)
    triaged = triage(pose, kinematics, OBJECTS, FPS)
    return frame_signals(pose, kinematics, triaged, OBJECTS, FPS)


def wandering_pose(n=240, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 2, size=(n, 2)).cumsum(axis=0) + 150
    return make_pose(n, body_xy=[tuple(p) for p in steps])


class TestSignals:
    def test_translation_does_not_move_the_head(self):
        # Andar em linha reta desloca todos os pontos juntos: a cabeça relativa ao
        # dorso fica parada, que é o que separa locomoção de grooming.
        pose = make_pose(60, body_xy=[(100 + i * 3, 100) for i in range(60)])

        signals = signals_for(pose)

        assert np.nanmax(signals["head_speed"]) < 1e-6
        assert np.nanmin(signals["body_speed"].iloc[1:]) > 80

    def test_lengths_are_relative_to_the_session(self):
        signals = signals_for(make_pose(60))

        assert np.allclose(signals["body_length"], 1.0)
        assert np.allclose(signals["ear_span"], 1.0)

    def test_missing_nose_is_nan_not_zero(self):
        pose = make_pose(60)
        pose.loc[20:40, "nose_p"] = 0.0

        signals = signals_for(pose)

        assert signals["object_distance"].iloc[20:41].isna().all()
        assert signals["full_length"].iloc[20:41].isna().all()


class TestWindows:
    def test_training_and_application_see_the_same_features(self):
        # O modelo treina em clipes e roda em janelas deslizantes: se as duas
        # vias calculassem diferente, a avaliação mediria outra coisa.
        signals = signals_for(wandering_pose())
        window, step = 60, 15

        sliding, centers = sliding_features(signals, window, step)
        clips = clip_features(signals, [(k * step, k * step + window - 1) for k in range(3)])

        pd.testing.assert_frame_equal(sliding.iloc[:3].reset_index(drop=True), clips)
        assert centers[0] == (window - 1) / 2

    def test_one_column_per_statistic(self):
        sliding, _ = sliding_features(signals_for(wandering_pose()), 60, 15)

        assert sliding.shape[1] == sum(len(stats) for stats in SPEC.values())

    def test_frame_takes_the_nearest_window(self):
        scores = frame_scores(np.array([0.1, 0.9, 0.2]), np.array([29.5, 44.5, 59.5]), 90)

        assert scores[0] == 0.1           # antes do primeiro centro
        assert scores[44] == 0.9
        assert scores[60] == 0.2
        assert scores[89] == 0.2          # depois do último


class TestLabels:
    def test_every_round_format(self):
        labels = pd.DataFrame({
            "label": ["grooming", "rearing", "walking"],
            "also": [None, "walking", None],
            "labels": [None, None, "walking;sniffing"],
        })

        names = label_sets(labels).tolist()

        assert names == [{"grooming"}, {"rearing", "walking"}, {"walking", "sniffing"}]

    def test_train_and_test_days_are_the_same_animal(self):
        assert animal_of("TR_22_cam1") == animal_of("TT_22_cam1") == 22
