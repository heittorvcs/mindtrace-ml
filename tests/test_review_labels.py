import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_labels import (  # noqa: E402
    Consensus,
    compare_episodes,
    decision_groups,
    disagreements,
    frame_status,
    group_at,
    initial_consensus,
)

FPS = 30.0


def marks(*intervals, behavior="rearing"):
    return pd.DataFrame([{"behavior": behavior, "start_frame": a, "end_frame": b, "field": 0}
                         for a, b in intervals], columns=["behavior", "start_frame", "end_frame", "field"])


def status_by_lane(episodes):
    return {lane: [(e["start"], e["end"], e["status"]) for e in episodes if e["lane"] == lane] for lane in "ab"}


class TestCompareEpisodes:
    def test_overlapping_episodes_are_shared_even_with_different_borders(self):
        episodes, _ = compare_episodes(marks((100, 160)), marks((110, 170)), "rearing", 400, FPS)

        assert status_by_lane(episodes) == {"a": [(100, 160, "both")], "b": [(110, 170, "both")]}

    def test_episode_marked_in_only_one_is_a_disagreement(self):
        episodes, _ = compare_episodes(marks((100, 160), (300, 330)), marks((105, 150)), "rearing", 400, FPS)

        assert status_by_lane(episodes)["a"][1] == (300, 330, "only_a")
        assert disagreements(episodes) == [(300, 330)]

    def test_brief_touch_down_is_merged_like_in_the_analysis(self):
        # Duas marcações separadas por 0,3 s são um rearing só, como em label_mask.
        episodes, _ = compare_episodes(marks((100, 130), (140, 170)), marks((100, 170)), "rearing", 400, FPS)

        assert status_by_lane(episodes)["a"] == [(100, 170, "both")]

    def test_unscorable_in_either_marking_is_left_out(self):
        a = pd.concat([marks((100, 160)), marks((90, 170), behavior="unscorable")])

        episodes, masks = compare_episodes(a, marks((300, 330)), "rearing", 400, FPS)

        assert status_by_lane(episodes) == {"a": [], "b": [(300, 330, "only_b")]}
        assert frame_status(masks, 120) == "unscorable"

    def test_other_behaviors_are_ignored(self):
        episodes, _ = compare_episodes(marks((100, 160), behavior="grooming"), marks(), "rearing", 400, FPS)

        assert episodes == []


class TestFrameStatus:
    def test_reports_what_each_marking_says_at_the_frame(self):
        _, masks = compare_episodes(marks((100, 160)), marks((130, 200)), "rearing", 400, FPS)

        assert frame_status(masks, 110) == "only_a"
        assert frame_status(masks, 140) == "both"
        assert frame_status(masks, 180) == "only_b"
        assert frame_status(masks, 300) is None


class TestDecisionGroups:
    def test_a_chain_of_overlaps_is_one_decision(self):
        # A antiga marcou dois rearings; a nova, um só que cruza os dois.
        groups = decision_groups(marks((100, 140), (170, 200)), marks((120, 190)), "rearing", 400, FPS)

        assert [(g["start"], g["end"], g["status"]) for g in groups] == [(100, 200, "both")]
        assert groups[0]["a"] == [(100, 140), (170, 200)]
        assert groups[0]["b"] == [(120, 190)]

    def test_each_version_keeps_the_marks_as_they_were_made(self):
        # Duas marcações fundidas pela descida breve continuam duas ao manter a versão.
        groups = decision_groups(marks((100, 130), (140, 170)), marks(), "rearing", 400, FPS)

        assert groups[0]["status"] == "only_a"
        assert groups[0]["a"] == [(100, 130), (140, 170)]
        assert groups[0]["b"] == []

    def test_disagreements_start_pending_and_agreements_prefilled(self):
        groups = decision_groups(marks((100, 160), (300, 330)), marks((110, 170)), "rearing", 400, FPS)

        intervals, decided = initial_consensus(groups, "b", "rearing")

        assert intervals == [("rearing", 110, 170)]
        assert decided == {0}
        assert initial_consensus(groups, "none", "rearing") == ([], set())


class TestGroupAt:
    def test_finds_the_group_ahead_after_a_jump(self):
        # n para 1 s antes do começo: o grupo ainda está à frente do cursor.
        groups = decision_groups(marks((100, 160)), marks(), "rearing", 400, FPS)

        assert group_at(groups, 70, lead=30) == 0
        assert group_at(groups, 60, lead=30) is None


class TestConsensus:
    def test_choosing_a_version_replaces_the_stretch_and_undoes_at_once(self):
        consensus = Consensus([("rearing", 110, 170), ("rearing", 300, 330)])

        consensus.replace(100, 200, [(100, 140), (170, 200)], "rearing")
        assert sorted(consensus.intervals) == [("rearing", 100, 140), ("rearing", 170, 200), ("rearing", 300, 330)]

        assert consensus.undo() == "replace"
        assert sorted(consensus.intervals) == [("rearing", 110, 170), ("rearing", 300, 330)]

    def test_keeping_neither_empties_the_stretch(self):
        consensus = Consensus([("rearing", 110, 170), ("unscorable", 120, 130)])

        consensus.replace(100, 200, [], "rearing")

        assert consensus.intervals == [("unscorable", 120, 130)]
