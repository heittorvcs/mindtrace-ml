import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_labels import compare_episodes, disagreements, frame_status  # noqa: E402

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
