import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.detectors import (
    Thresholds,
    detect_freezing,
    detect_object_interaction,
    detect_walking,
    head_direction,
    reduction_rate,
    review_segments,
    triage,
)
from mindtrace_ml.kinematics import frame_kinematics

KEYPOINTS = ("nose", "ear_left", "ear_right", "neck", "body", "tail_base")
FPS = 30.0


def make_pose(n, body_xy=None, nose_xy=None, ears=None):
    body_xy = body_xy or [(100.0, 100.0)] * n
    nose_xy = nose_xy or [(x, y - 20) for x, y in body_xy]
    if ears is None:
        ears = [((x, y - 10), (x + 8, y - 10)) for x, y in body_xy]

    data = {
        "frame": np.arange(n),
        "time_ms": np.arange(n) * (1000.0 / FPS),
        "nose_x": [p[0] for p in nose_xy],
        "nose_y": [p[1] for p in nose_xy],
        "ear_left_x": [e[0][0] for e in ears],
        "ear_left_y": [e[0][1] for e in ears],
        "ear_right_x": [e[1][0] for e in ears],
        "ear_right_y": [e[1][1] for e in ears],
        "neck_x": [x for x, _ in body_xy],
        "neck_y": [y - 8 for _, y in body_xy],
        "body_x": [p[0] for p in body_xy],
        "body_y": [p[1] for p in body_xy],
        "tail_base_x": [x for x, _ in body_xy],
        "tail_base_y": [y + 15 for _, y in body_xy],
    }
    for name in KEYPOINTS:
        data[f"{name}_p"] = [1.0] * n
    return pd.DataFrame(data)


def kin(pose):
    return frame_kinematics(pose, KEYPOINTS, FPS)


class TestWalking:
    def test_sustained_motion_is_walking(self):
        pose = make_pose(60, body_xy=[(100 + i * 3, 100) for i in range(60)])

        mask = detect_walking(kin(pose), FPS, Thresholds())

        assert mask.sum() > 40

    def test_stationary_is_not_walking(self):
        mask = detect_walking(kin(make_pose(60)), FPS, Thresholds())

        assert not mask.any()

    def test_brief_burst_is_discarded(self):
        # 3 quadros de movimento = 0,1 s, abaixo do mínimo de 0,5 s.
        positions = [(100.0, 100.0)] * 20 + [(130.0, 100.0), (160.0, 100.0), (190.0, 100.0)]
        positions += [(190.0, 100.0)] * 20
        pose = make_pose(len(positions), body_xy=positions)

        mask = detect_walking(kin(pose), FPS, Thresholds())

        assert not mask.any()


class TestFreezing:
    def test_immobility_is_freezing(self):
        mask = detect_freezing(kin(make_pose(60)), FPS, Thresholds())

        assert mask.sum() > 40

    def test_motion_is_not_freezing(self):
        pose = make_pose(60, body_xy=[(100 + i * 3, 100) for i in range(60)])

        assert not detect_freezing(kin(pose), FPS, Thresholds()).any()

    def test_missing_pose_is_not_freezing(self):
        # Ausência de pose não é imobilidade — é ausência de informação.
        pose = make_pose(60)
        for name in KEYPOINTS:
            pose.loc[10:40, f"{name}_x"] = np.nan
            pose.loc[10:40, f"{name}_y"] = np.nan

        mask = detect_freezing(kin(pose), FPS, Thresholds())

        assert not mask[15:38].any()


class TestObjectInteraction:
    def setup_method(self):
        self.objects = pd.DataFrame([
            {"center_x": 100.0, "center_y": 40.0, "radius": 15.0},
        ])

    def test_near_and_facing_counts(self):
        # Focinho logo abaixo do objeto, cabeça apontando para cima (para ele).
        n = 30
        pose = make_pose(n, body_xy=[(100.0, 100.0)] * n,
                         nose_xy=[(100.0, 62.0)] * n,
                         ears=[((96.0, 75.0), (104.0, 75.0))] * n)

        mask = detect_object_interaction(pose, self.objects, FPS, Thresholds())

        assert mask.all()

    def test_near_but_facing_away_does_not_count(self):
        # Mesma posição, cabeça apontando para baixo — passar perto não é explorar.
        n = 30
        pose = make_pose(n, body_xy=[(100.0, 100.0)] * n,
                         nose_xy=[(100.0, 62.0)] * n,
                         ears=[((96.0, 50.0), (104.0, 50.0))] * n)

        mask = detect_object_interaction(pose, self.objects, FPS, Thresholds())

        assert not mask.any()

    def test_facing_but_far_does_not_count(self):
        n = 30
        pose = make_pose(n, body_xy=[(100.0, 220.0)] * n,
                         nose_xy=[(100.0, 200.0)] * n,
                         ears=[((96.0, 210.0), (104.0, 210.0))] * n)

        assert not detect_object_interaction(pose, self.objects, FPS, Thresholds()).any()

    def test_head_direction_points_from_ears_to_nose(self):
        pose = make_pose(1, body_xy=[(100.0, 100.0)],
                         nose_xy=[(100.0, 60.0)],
                         ears=[((96.0, 80.0), (104.0, 80.0))])

        dx, dy = head_direction(pose)

        assert dx[0] == pytest.approx(0.0)
        assert dy[0] == pytest.approx(-20.0)


class TestTriage:
    def test_review_is_what_no_detector_claimed(self):
        pose = make_pose(60)
        objects = pd.DataFrame([{"center_x": 300.0, "center_y": 20.0, "radius": 10.0}])

        result = triage(pose, kin(pose), objects, FPS)

        # Parado longe do objeto: freezing cobre, então não vai para revisão.
        assert result["freezing"].sum() > 40
        assert result["review"].sum() < 20
        assert not (result["review"] & result[list(("walking", "freezing", "object_interaction"))].any(axis=1)).any()

    def test_reduction_rate_counts_what_is_skipped(self):
        triaged = pd.DataFrame({
            "frame": np.arange(10), "time_ms": np.arange(10) * 33.3,
            "walking": [True] * 8 + [False] * 2,
            "freezing": [False] * 10,
            "object_interaction": [False] * 10,
        })
        triaged["review"] = ~triaged[["walking", "freezing", "object_interaction"]].any(axis=1)

        assert reduction_rate(triaged) == pytest.approx(0.8)

    def test_segments_come_longest_first(self):
        review = np.zeros(100, dtype=bool)
        review[10:15] = True
        review[40:70] = True
        triaged = pd.DataFrame({
            "frame": np.arange(100), "time_ms": np.arange(100) * 33.3,
            "walking": ~review, "freezing": False, "object_interaction": False,
            "review": review,
        })

        segments = review_segments(triaged)

        assert len(segments) == 2
        assert segments.iloc[0]["n_frames"] == 30
        assert segments.iloc[1]["n_frames"] == 5
