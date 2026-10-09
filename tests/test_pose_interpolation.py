import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.pose import Keypoint, extract_video, interpolate_skipped  # noqa: E402


def pose(*points, field=0):
    """Um ponto (`nose`) por quadro; None marca quadro sem inferência."""
    rows = []
    for frame, point in enumerate(points):
        x, y, p = (np.nan, np.nan, np.nan) if point is None else point
        rows.append({"frame": frame, "time_ms": frame * 33.37, "field": field,
                     "nose_x": x, "nose_y": y, "nose_p": p})
    return pd.DataFrame(rows)


def nose(table):
    return [tuple(round(float(v), 3) for v in row) for row in table[["nose_x", "nose_y", "nose_p"]].to_numpy()]


class TestInterpolateSkipped:
    def test_skipped_frame_lies_on_the_line_between_its_neighbours(self):
        table = pose((10, 100, 0.6), None, (20, 120, 0.8), None, (40, 120, 0.8))

        filled = nose(interpolate_skipped(table, ["nose"]))

        assert filled[1] == (15, 110, 0.7)
        assert filled[3] == (30, 120, 0.8)

    def test_works_for_any_step(self):
        table = pose((0, 0, 0.6), None, None, (30, 60, 0.6))

        filled = nose(interpolate_skipped(table, ["nose"]))

        assert filled[1:3] == [(10, 20, 0.6), (20, 40, 0.6)]

    def test_undetected_neighbour_does_not_invent_a_position(self):
        # A reta entre (100, 100) e o -1 de um ponto perdido passaria por (49,5; 49,5):
        # uma posição que o rato nunca ocupou.
        lost = pose((100, 100, 0.8), None, (-1, -1, 0.0))
        doubtful = pose((100, 100, 0.8), None, (120, 100, 0.1))

        assert nose(interpolate_skipped(lost, ["nose"]))[1] == (-1, -1, 0)
        assert nose(interpolate_skipped(doubtful, ["nose"]))[1] == (-1, -1, 0)

    def test_inferred_frames_are_untouched(self):
        # Inclusive os de confiança baixa: o limiar é decisão de análise, não de extração.
        table = pose((10, 10, 0.8), None, (50, 50, 0.1))

        filled = nose(interpolate_skipped(table, ["nose"]))

        assert filled[0] == (10, 10, 0.8)
        assert filled[2] == (50, 50, 0.1)

    def test_after_the_last_inferred_frame_repeats_it(self):
        table = pose((10, 10, 0.8), None, (20, 20, 0.8), None)

        assert nose(interpolate_skipped(table, ["nose"]))[3] == (20, 20, 0.8)

    def test_fields_are_interpolated_independently(self):
        # No mosaico, as linhas dos campos se intercalam quadro a quadro.
        first = pose((0, 0, 0.8), None, (20, 20, 0.8), field=0)
        second = pose((100, 100, 0.8), None, (300, 300, 0.8), field=1)
        table = pd.concat([first, second]).sort_values(["frame", "field"]).reset_index(drop=True)

        filled = interpolate_skipped(table, ["nose"])

        assert nose(filled[filled.field == 0])[1] == (10, 10, 0.8)
        assert nose(filled[filled.field == 1])[1] == (200, 200, 0.8)

    def test_nothing_to_fill_returns_the_same_values(self):
        table = pose((10, 10, 0.8), (11, 10, 0.8))

        assert nose(interpolate_skipped(table, ["nose"])) == nose(table)


class CountingModel:
    """Modelo de pose falso: um ponto que anda 10 px a cada inferência."""

    n_keypoints = 1

    def __init__(self):
        self.calls = 0

    def infer(self, _crop):
        point = Keypoint(x=10.0 * self.calls, y=5.0, p=0.9)
        self.calls += 1
        return [point]


def write_video(path, n_frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (360, 240))
    for index in range(n_frames):
        writer.write(np.full((240, 360, 3), index * 10, np.uint8))
    writer.release()


class TestExtractVideoEvery:
    def test_every_two_infers_half_and_keeps_one_row_per_frame(self, tmp_path):
        video = tmp_path / "sessao.avi"
        write_video(video, 6)
        model = CountingModel()

        rows, _ = extract_video(video, model, layout="full", every=2, keypoints=("nose",))
        table = pd.DataFrame(rows)

        assert model.calls == 3
        assert table.frame.tolist() == [0, 1, 2, 3, 4, 5]
        assert table.nose_x.tolist() == [0, 5, 10, 15, 20, 20]

    def test_sparse_keeps_only_the_inferred_frames(self, tmp_path):
        video = tmp_path / "sessao.avi"
        write_video(video, 6)

        rows, _ = extract_video(video, CountingModel(), layout="full", every=2,
                                keypoints=("nose",), interpolate=False)

        assert [row["frame"] for row in rows] == [0, 2, 4]
