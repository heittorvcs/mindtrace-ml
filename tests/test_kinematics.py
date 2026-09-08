import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.kinematics import (
    build_features,
    coverage_by_keypoint,
    frame_kinematics,
    window_features,
)

KEYPOINTS = ("nose", "body")


def make_pose(nose_xy, body_xy=None, probability=1.0):
    n = len(nose_xy)
    body_xy = body_xy or [(x, y + 10) for x, y in nose_xy]
    return pd.DataFrame({
        "frame": np.arange(n),
        "time_ms": np.arange(n) * 40.0,
        "nose_x": [p[0] for p in nose_xy],
        "nose_y": [p[1] for p in nose_xy],
        "nose_p": [probability] * n,
        "body_x": [p[0] for p in body_xy],
        "body_y": [p[1] for p in body_xy],
        "body_p": [probability] * n,
    })


class TestSpeedUnits:
    def test_speed_is_pixels_per_second_not_per_frame(self):
        # 10 px por quadro a 25 fps = 250 px/s.
        pose = make_pose([(0, 0), (10, 0), (20, 0)])

        base = frame_kinematics(pose, KEYPOINTS, fps=25.0)

        assert base["speed_nose"].iloc[1] == pytest.approx(250.0)

    def test_same_motion_at_different_fps_gives_different_speed(self):
        pose = make_pose([(0, 0), (10, 0)])

        at_25 = frame_kinematics(pose, KEYPOINTS, fps=25.0)["speed_nose"].iloc[1]
        at_30 = frame_kinematics(pose, KEYPOINTS, fps=30.0)["speed_nose"].iloc[1]

        assert at_30 > at_25
        assert at_30 / at_25 == pytest.approx(30 / 25)

    def test_first_frame_speed_is_undefined(self):
        pose = make_pose([(0, 0), (5, 0)])

        base = frame_kinematics(pose, KEYPOINTS, fps=25.0)

        assert np.isnan(base["speed_nose"].iloc[0])

    def test_rejects_invalid_fps(self):
        pose = make_pose([(0, 0), (1, 1)])

        with pytest.raises(ValueError):
            frame_kinematics(pose, KEYPOINTS, fps=0)


class TestMissingPose:
    def test_nan_position_propagates_instead_of_reading_as_stationary(self):
        pose = make_pose([(0, 0), (np.nan, np.nan), (0, 0)])

        base = frame_kinematics(pose, KEYPOINTS, fps=25.0)

        assert np.isnan(base["speed_nose"].iloc[1])
        assert np.isnan(base["speed_nose"].iloc[2])
        # Um zero aqui seria lido pelo modelo como "animal parado", que é falso.
        assert not (base["speed_nose"].fillna(-1) == 0).any()


class TestPairwiseDistance:
    def test_distance_between_keypoints(self):
        pose = make_pose([(0, 0)], body_xy=[(3, 4)])

        base = frame_kinematics(pose, KEYPOINTS, fps=25.0)

        assert base["dist_nose_body"].iloc[0] == pytest.approx(5.0)

    def test_one_column_per_pair(self):
        pose = make_pose([(0, 0), (1, 1)])

        base = frame_kinematics(pose, KEYPOINTS, fps=25.0)
        distance_columns = [c for c in base.columns if c.startswith("dist_")]

        assert distance_columns == ["dist_nose_body"]


class TestWindows:
    def test_window_size_derives_from_fps(self):
        # Sinal com aceleração, para que o tamanho da janela mude a média.
        pose = make_pose([(i * i, 0) for i in range(50)])
        base = frame_kinematics(pose, KEYPOINTS, fps=25.0)

        narrow = window_features(base, fps=25.0, windows_sec=(0.4,), stats=("mean",))
        wide = window_features(base, fps=50.0, windows_sec=(0.4,), stats=("mean",))

        assert not narrow["speed_nose_w0_4_mean"].equals(wide["speed_nose_w0_4_mean"])

    def test_spatial_spread_separates_grooming_from_walking(self):
        # As duas trajetórias têm velocidade idêntica em módulo (8 px/quadro).
        # O que difere é a área percorrida, e é isso que precisa aparecer.
        oscillating = [(0, 0), (8, 0), (0, 0), (8, 0), (0, 0), (8, 0), (0, 0), (8, 0)]
        advancing = [(i * 8, 0) for i in range(8)]

        groom = build_features(make_pose(oscillating), KEYPOINTS, 25.0)
        walk = build_features(make_pose(advancing), KEYPOINTS, 25.0)

        assert groom["speed_nose_w0_4_mean"].max() == pytest.approx(
            walk["speed_nose_w0_4_mean"].max()
        )
        assert walk["posx_nose_w0_4_std"].max() > groom["posx_nose_w0_4_std"].max()

    def test_window_is_centered_so_it_uses_future_context(self):
        # Um pico no quadro 5 deve aparecer na janela do quadro 4, o que só é
        # possível com janela centrada.
        positions = [(0, 0)] * 5 + [(100, 0)] + [(100, 0)] * 4
        base = frame_kinematics(make_pose(positions), KEYPOINTS, fps=25.0)

        centered = window_features(base, fps=25.0, windows_sec=(0.2,), stats=("max",))

        assert centered["speed_nose_w0_2_max"].iloc[4] > 0

    def test_column_count_matches_windows_times_stats(self):
        pose = make_pose([(i, 0) for i in range(20)])

        feats = build_features(pose, KEYPOINTS, 25.0,
                               windows_sec=(0.2, 1.0), stats=("mean", "std"))

        # Por ponto: velocidade + posição x + posição y. Mais um par de distância.
        n = len(KEYPOINTS)
        expected_base = 3 * n + n * (n - 1) // 2
        value_columns = [c for c in feats.columns if c not in ("frame", "time_ms")]

        assert len(value_columns) == expected_base * 2 * 2


class TestCoverage:
    def test_reports_fraction_above_threshold(self):
        pose = make_pose([(0, 0)] * 10)
        pose.loc[:4, "nose_p"] = 0.2

        report = coverage_by_keypoint(pose, KEYPOINTS).set_index("keypoint")

        assert report.loc["nose", "coverage"] == pytest.approx(0.5)
        assert report.loc["nose", "frames_lost"] == 5
        assert report.loc["body", "coverage"] == 1.0
