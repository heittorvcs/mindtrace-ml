import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from label_session import Annotation, replace_rows  # noqa: E402
from mindtrace_ml.labels import load_labels  # noqa: E402


class TestAnnotation:
    def test_key_opens_then_closes(self):
        annotation = Annotation()

        annotation.toggle("grooming", 100)
        annotation.toggle("grooming", 160)

        assert annotation.intervals == [("grooming", 100, 160)]
        assert not annotation.open

    def test_behaviors_overlap(self):
        annotation = Annotation()

        annotation.toggle("sniffing", 10)
        annotation.toggle("rearing", 20)
        annotation.toggle("rearing", 40)
        annotation.toggle("sniffing", 50)

        assert sorted(annotation.intervals) == [("rearing", 20, 40), ("sniffing", 10, 50)]

    def test_closing_before_the_start_swaps(self):
        # Voltou o vídeo antes de fechar: o intervalo continua válido.
        annotation = Annotation()

        annotation.toggle("rearing", 80)
        annotation.toggle("rearing", 50)

        assert annotation.intervals == [("rearing", 50, 80)]

    def test_undo_close_reopens_at_original_start(self):
        annotation = Annotation()
        annotation.toggle("grooming", 100)
        annotation.toggle("grooming", 130)

        annotation.undo()
        annotation.toggle("grooming", 200)

        assert annotation.intervals == [("grooming", 100, 200)]

    def test_undo_open_cancels(self):
        annotation = Annotation()
        annotation.toggle("other", 5)

        annotation.undo()

        assert not annotation.open and not annotation.intervals

    def test_remove_and_undo_restores(self):
        annotation = Annotation([("rearing", 10, 20), ("grooming", 30, 90)])

        removed = annotation.remove_at(50)
        assert removed == ("grooming", 30, 90)
        assert annotation.intervals == [("rearing", 10, 20)]

        annotation.undo()
        assert annotation.intervals == [("rearing", 10, 20), ("grooming", 30, 90)]

    def test_quitting_closes_what_is_open(self):
        annotation = Annotation()
        annotation.toggle("sniffing", 10)

        closed = annotation.close_all(40)

        assert closed == ["sniffing"]
        assert annotation.intervals == [("sniffing", 10, 40)]


class TestPersistence:
    def test_saved_file_is_a_valid_label_file(self, tmp_path):
        annotation = Annotation([("rearing", 10, 20), ("object_interaction", 30, 45)])
        path = tmp_path / "labels.csv"

        replace_rows(path, annotation.rows("TT_22_cam1", "a"), "TT_22_cam1", "a")

        labels = load_labels(path)
        assert list(labels.behavior) == ["rearing", "object_interaction"]
        assert (labels.animal_id == 22).all()

    def test_saving_one_session_keeps_the_others(self, tmp_path):
        path = tmp_path / "labels.csv"
        replace_rows(path, Annotation([("rearing", 1, 2)]).rows("TR_19_cam1", "a"), "TR_19_cam1", "a")
        replace_rows(path, Annotation([("grooming", 5, 9)]).rows("TT_22_cam1", "a"), "TT_22_cam1", "a")

        replace_rows(path, Annotation([("grooming", 5, 12)]).rows("TT_22_cam1", "a"), "TT_22_cam1", "a")

        saved = pd.read_csv(path, encoding="utf-8-sig")
        assert len(saved) == 2
        assert saved.set_index("session_id").loc["TT_22_cam1", "end_frame"] == 12
