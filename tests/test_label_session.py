import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from label_session import Annotation, Stepper, replace_rows  # noqa: E402
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


class TestSuggestions:
    def test_accepted_suggestion_can_be_undone(self):
        annotation = Annotation()

        annotation.add("rearing", 100, 140)
        assert annotation.intervals == [("rearing", 100, 140)]

        annotation.undo()
        assert annotation.intervals == []

    def test_delete_only_touches_the_behavior_being_labeled(self):
        # Sessão já rotulada por completo, reaberta só para rearing: apagar sob o
        # cursor não pode levar junto a exploração marcada antes.
        annotation = Annotation([("object_interaction", 0, 100), ("rearing", 40, 60)])

        assert annotation.remove_at(50, allowed={"rearing"}) == ("rearing", 40, 60)
        assert annotation.remove_at(50, allowed={"rearing"}) is None
        assert annotation.intervals == [("object_interaction", 0, 100)]


class FakeRoot:
    """Relógio de mentira com o after/after_cancel do Tk. Cada passo custa 30 ms de
    desenho, como no notebook; o que vence enquanto isso espera, como na fila do Tk."""

    def __init__(self):
        self.now, self.queue, self.count = 0, [], 0

    def after(self, ms, callback):
        self.count += 1
        self.queue.append((self.now + ms, self.count, callback))
        return self.count

    def after_cancel(self, ident):
        self.queue = [item for item in self.queue if item[1] != ident]

    def run_until(self, until):
        while True:
            due = sorted(item for item in self.queue if item[0] <= until)
            if not due:
                break
            self.queue.remove(due[0])
            self.now = max(self.now, due[0][0])
            due[0][2]()
        self.now = max(self.now, until)


def held_arrow(release_at, big=False, autorepeat_from=500):
    """Aperta a seta em 0 e solta em `release_at`, com a repetição do X11 no meio."""
    root, steps = FakeRoot(), []

    def step(key, large):
        steps.append(root.now)
        root.now += 30

    stepper = Stepper(root, step)
    root.after(0, lambda: stepper.press("Right", big))
    for t in range(autorepeat_from, release_at, 33):
        root.after(t, lambda: stepper.release("Right"))
        root.after(t, lambda: stepper.press("Right", big))
    root.after(release_at, lambda: stepper.release("Right"))
    return root, stepper, steps


class TestStepper:
    def test_a_tap_moves_one_step(self):
        root, _, steps = held_arrow(release_at=100)

        root.run_until(3000)

        assert len(steps) == 1

    def test_holding_keeps_moving_and_releasing_stops_at_once(self):
        root, _, steps = held_arrow(release_at=1500)

        root.run_until(5000)

        assert len(steps) > 20
        # Depois de soltar, no máximo o passo que já estava sendo desenhado.
        assert max(steps) <= 1500 + Stepper.RELEASE_MS + 30

    def test_keyboard_repeat_adds_no_steps_of_its_own(self):
        noisy, _, with_repeat = held_arrow(release_at=1500)
        quiet, _, without = held_arrow(release_at=1500, autorepeat_from=10_000)

        noisy.run_until(5000)
        quiet.run_until(5000)

        assert abs(len(with_repeat) - len(without)) <= 2

    def test_big_steps_held_are_paced(self):
        root, _, steps = held_arrow(release_at=1300, big=True)

        root.run_until(5000)

        assert 4 <= len(steps) <= 7

    def test_losing_focus_stops_the_walk(self):
        # Sem foco, a janela não recebe mais nem a repetição nem o soltar da tecla.
        root, stepper, steps = held_arrow(release_at=10_000, autorepeat_from=10_000)

        root.run_until(800)
        stepper.stop()
        root.run_until(3000)

        assert max(steps) <= 800 + 30
