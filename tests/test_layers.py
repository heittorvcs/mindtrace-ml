import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mindtrace_ml.behavior_model import bouts, centered_features, compare, frame_mask, gate  # noqa: E402
from mindtrace_ml.kinematics import frame_kinematics  # noqa: E402
from mindtrace_ml.movement import movement_layer, session_summary  # noqa: E402
from mindtrace_ml.triage_model import SPEC, describe  # noqa: E402
from test_detectors import FPS, KEYPOINTS, make_pose  # noqa: E402
from test_triage_model import signals_for, wandering_pose  # noqa: E402

OBJECTS = pd.DataFrame({"session_id": ["s", "s"], "object_id": [1, 2],
                        "center_x": [100.0, 300.0], "center_y": [40.0, 40.0], "radius": [15.0, 15.0]})


def layer_for(pose):
    return movement_layer(pose, frame_kinematics(pose, KEYPOINTS, FPS), OBJECTS, FPS)


class TestMovement:
    def test_standing_still_travels_nothing(self):
        layer = layer_for(make_pose(90))

        assert not layer.moving.any()
        # O primeiro quadro não tem velocidade (não há anterior): conta como sem pose.
        assert layer.still.iloc[1:].all()
        assert layer.distance.iloc[-1] == 0

    def test_walking_counts_the_path(self):
        # 3 px por quadro durante 3 s: ~90 px/s, bem acima do corte de movimento.
        pose = make_pose(90, body_xy=[(150 + i * 3, 150) for i in range(90)])

        layer = layer_for(pose)

        assert layer.moving.mean() > 0.8
        assert 200 < layer.distance.iloc[-1] <= 270

    def test_detection_jumps_are_not_distance(self):
        positions = [(150 + i * 3, 150) for i in range(45)] + [(400 + i * 3, 150) for i in range(45)]

        layer = layer_for(make_pose(90, body_xy=positions))

        assert layer.distance.iloc[-1] < 300

    def test_exploration_is_split_by_object(self):
        # Focinho encostado no objeto 1 o tempo todo.
        pose = make_pose(60, body_xy=[(100.0, 100.0)] * 60, nose_xy=[(100.0, 62.0)] * 60)

        layer = layer_for(pose)
        summary = session_summary(layer, FPS)

        assert layer.exploring_1.all() and not layer.exploring_2.any()
        assert summary["exploration_1_s"] == round(60 / FPS, 1)
        assert summary["exploration_1_bouts"] == 1
        assert summary["exploration_total_s"] == summary["exploration_1_s"]


class TestBehaviorWindows:
    def test_windows_are_centered_on_the_frame(self):
        signals = signals_for(wandering_pose())
        centers = np.array([40, 100])

        features = centered_features(signals, centers, scales=(15,))

        matrix = signals[list(SPEC)].to_numpy(float)
        expected = describe(np.stack([matrix[33:48], matrix[93:108]]))
        np.testing.assert_allclose(features.to_numpy(), expected.to_numpy(), equal_nan=True)

    def test_windows_near_the_edge_do_not_fail(self):
        signals = signals_for(wandering_pose(n=120))

        features = centered_features(signals, np.array([0, 119]))

        assert len(features) == 2

    def test_short_detections_are_dropped(self):
        scores = np.zeros(20)
        scores[5] = 0.9                 # uma janela isolada, ~8 quadros
        scores[10:14] = 0.9             # quatro seguidas, ~32 quadros
        centers = np.arange(0, 160, 8)

        mask = frame_mask(scores, centers, 160, 0.5, FPS)

        assert bouts(mask) == [(76, 107)]

    def test_compare_counts_frames_and_bouts(self):
        marked = np.zeros(100, bool)
        marked[10:30] = True
        marked[60:70] = True
        detected = np.zeros(100, bool)
        detected[20:40] = True

        result = compare(marked, detected, 10.0)

        assert result["manual_bouts"] == 2 and result["detected_bouts"] == 1
        assert result["frames_found"] == 10 / 30
        assert result["frames_right"] == 10 / 20
        assert result["bouts_found"] == 0.5
        assert result["bouts_right"] == 1.0

    def test_freezing_vetoes_grooming_but_not_rearing(self):
        detected = np.zeros(100, bool)
        detected[10:60] = True
        freezing = np.zeros(100, bool)
        freezing[30:100] = True

        assert bouts(gate("grooming", detected, freezing, FPS)) == [(10, 29)]
        assert bouts(gate("rearing", detected, freezing, FPS)) == [(10, 59)]
