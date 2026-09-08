import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml import (
    BEHAVIORS,
    bouts_for,
    check_leakage,
    cooccurrence,
    frame_metrics,
    label_matrix,
    mask_to_bouts,
    multilabel_rate,
    rule_predictions,
    split_by_animal,
    unreachable_recall,
)
from mindtrace_ml.metrics import bout_metrics, interval_iou, match_bouts
from mindtrace_ml.schema import FEATURE_COLUMNS, FRAME_COLUMN, RULE_LABEL_COLUMN


def make_labels(rows):
    return pd.DataFrame(
        [
            {
                "session_id": "S001",
                "animal_id": "R01",
                "field": 0,
                "behavior": behavior,
                "start_frame": start,
                "end_frame": end,
                "annotator": "ana",
            }
            for behavior, start, end in rows
        ]
    )


def make_features(frames, rule_labels):
    data = {FRAME_COLUMN: frames}
    for column in FEATURE_COLUMNS:
        data[column] = np.zeros(len(frames))
    data[RULE_LABEL_COLUMN] = rule_labels
    return pd.DataFrame(data)


class TestLabelExpansion:
    def test_projects_onto_recorded_frames_only(self):
        # Quadros 5-7 não foram registrados pelo MindTrace (confiança baixa).
        frames = np.array([0, 1, 2, 3, 4, 8, 9, 10])
        labels = make_labels([("walking", 2, 9)])

        y, scorable = label_matrix(labels, frames)

        walking = y[:, BEHAVIORS.index("walking")]
        assert walking.tolist() == [False, False, True, True, True, True, True, False]
        assert scorable.all()

    def test_unscorable_masks_frames_without_labelling_behaviour(self):
        frames = np.arange(10)
        labels = make_labels([("resting", 0, 9), ("unscorable", 4, 6)])

        y, scorable = label_matrix(labels, frames)

        assert y[:, BEHAVIORS.index("resting")].all()
        assert scorable.tolist() == [True] * 4 + [False] * 3 + [True] * 3

    def test_cooccurrence_is_preserved(self):
        frames = np.arange(10)
        labels = make_labels([("sniffing", 0, 5), ("rearing", 3, 8)])

        y, _ = label_matrix(labels, frames)

        both = y[:, BEHAVIORS.index("sniffing")] & y[:, BEHAVIORS.index("rearing")]
        assert both.sum() == 3

    def test_adjacent_bouts_of_same_behaviour_merge(self):
        labels = make_labels([("walking", 0, 4), ("walking", 5, 9)])
        assert bouts_for(labels, "walking") == [(0, 9)]

    def test_disjoint_bouts_stay_separate(self):
        labels = make_labels([("walking", 0, 4), ("walking", 20, 24)])
        assert bouts_for(labels, "walking") == [(0, 4), (20, 24)]


class TestBoutSegmentation:
    def test_gap_in_recorded_frames_splits_bout(self):
        frames = np.array([0, 1, 2, 30, 31, 32])
        mask = np.ones(6, dtype=bool)

        assert mask_to_bouts(mask, frames, max_gap=2) == [(0, 2), (30, 32)]

    def test_small_gap_is_tolerated(self):
        frames = np.array([0, 1, 3, 4])
        mask = np.ones(4, dtype=bool)

        assert mask_to_bouts(mask, frames, max_gap=2) == [(0, 4)]

    def test_empty_mask_yields_no_bouts(self):
        assert mask_to_bouts(np.zeros(5, dtype=bool), np.arange(5)) == []


class TestBoutMatching:
    def test_iou_of_identical_intervals_is_one(self):
        assert interval_iou((0, 9), (0, 9)) == 1.0

    def test_disjoint_intervals_have_zero_iou(self):
        assert interval_iou((0, 4), (10, 14)) == 0.0

    def test_half_overlap(self):
        assert interval_iou((0, 9), (5, 14)) == pytest.approx(5 / 15)

    def test_each_bout_matches_at_most_once(self):
        true_bouts = [(0, 9)]
        pred_bouts = [(0, 9), (0, 8)]

        matches = match_bouts(true_bouts, pred_bouts, iou_threshold=0.5)

        assert len(matches) == 1
        assert matches[0][1] == 0


class TestRuleBaseline:
    def test_rule_labels_become_one_hot(self):
        features = make_features(np.arange(3), [0, 1, 4])

        predictions = rule_predictions(features)

        assert predictions.sum(axis=1).tolist() == [1, 1, 1]
        assert predictions[0, BEHAVIORS.index("walking")]
        assert predictions[1, BEHAVIORS.index("sniffing")]
        assert predictions[2, BEHAVIORS.index("rearing")]

    def test_cooccurring_truth_is_unreachable_for_exclusive_rules(self):
        frames = np.arange(10)
        labels = make_labels([("sniffing", 0, 9), ("rearing", 0, 9)])
        y_true, scorable = label_matrix(labels, frames)
        # As regras dão prioridade a sniffing, então rearing nunca é emitido.
        features = make_features(frames, [1] * 10)
        y_pred = rule_predictions(features)

        report = frame_metrics(y_true, y_pred, scorable).set_index("behavior")

        assert report.loc["sniffing", "recall"] == 1.0
        assert report.loc["rearing", "recall"] == 0.0

        unreachable = unreachable_recall(y_true, scorable).set_index("behavior")
        assert unreachable.loc["rearing", "unreachable_pct"] == 100.0
        assert multilabel_rate(y_true, scorable) == 1.0

    def test_cooccurrence_diagonal_is_prevalence(self):
        frames = np.arange(10)
        labels = make_labels([("walking", 0, 4)])
        y_true, scorable = label_matrix(labels, frames)

        matrix = cooccurrence(y_true, scorable)

        assert matrix.loc["walking", "walking"] == pytest.approx(0.5)


class TestFrameMetrics:
    def test_perfect_prediction(self):
        frames = np.arange(10)
        labels = make_labels([("walking", 0, 9)])
        y_true, scorable = label_matrix(labels, frames)
        features = make_features(frames, [0] * 10)

        report = frame_metrics(y_true, rule_predictions(features), scorable).set_index("behavior")

        assert report.loc["walking", "f1"] == 1.0
        assert report.loc["resting", "f1"] == 0.0

    def test_unscorable_frames_are_excluded(self):
        frames = np.arange(10)
        # Regras erram nos quadros 0-4, mas eles estão marcados como não pontuáveis.
        labels = make_labels([("walking", 0, 9), ("unscorable", 0, 4)])
        y_true, scorable = label_matrix(labels, frames)
        features = make_features(frames, [3] * 5 + [0] * 5)

        report = frame_metrics(y_true, rule_predictions(features), scorable).set_index("behavior")

        assert report.loc["walking", "recall"] == 1.0


class TestBoutMetrics:
    def test_fragmented_prediction_hurts_count_not_frames(self):
        frames = np.arange(20)
        labels = make_labels([("walking", 0, 19)])
        y_true, _ = label_matrix(labels, frames)
        # Uma predição partida ao meio: quase todos os quadros certos, 2 bouts em vez de 1.
        y_pred = np.zeros_like(y_true)
        walking = BEHAVIORS.index("walking")
        y_pred[0:9, walking] = True
        y_pred[12:20, walking] = True

        report = bout_metrics(y_true, y_pred, frames).set_index("behavior")

        assert report.loc["walking", "n_true"] == 1
        assert report.loc["walking", "n_pred"] == 2
        assert report.loc["walking", "count_error"] == 1

    def test_total_time_error_is_reported(self):
        frames = np.arange(20)
        labels = make_labels([("resting", 0, 9)])
        y_true, _ = label_matrix(labels, frames)
        y_pred = np.zeros_like(y_true)
        y_pred[0:5, BEHAVIORS.index("resting")] = True

        report = bout_metrics(y_true, y_pred, frames).set_index("behavior")

        assert report.loc["resting", "time_true"] == 10
        assert report.loc["resting", "time_pred"] == 5
        assert report.loc["resting", "time_error_pct"] == pytest.approx(-50.0)


class TestSplits:
    def test_animals_never_cross_partitions(self):
        manifest = pd.DataFrame({
            "session_id": [f"S{i:03d}" for i in range(20)],
            "animal_id": [f"R{i % 10:02d}" for i in range(20)],
        })

        assignment = split_by_animal(manifest, seed=7)

        assert check_leakage(assignment) == []
        assert set(assignment["split"]) <= {"train", "validation", "test"}

    def test_split_is_reproducible(self):
        manifest = pd.DataFrame({
            "session_id": [f"S{i:03d}" for i in range(12)],
            "animal_id": [f"R{i:02d}" for i in range(12)],
        })

        first = split_by_animal(manifest, seed=1)
        second = split_by_animal(manifest, seed=1)

        assert first["split"].tolist() == second["split"].tolist()

    def test_fractions_must_sum_to_one(self):
        manifest = pd.DataFrame({"session_id": ["S1"], "animal_id": ["R1"]})

        with pytest.raises(ValueError):
            split_by_animal(manifest, fractions=(0.5, 0.2, 0.2))
