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


class FakeTk:
    """Laço de eventos com o comportamento do registro real (sessão de 2026-10-28):
    com teclas na fila, o Tk trata uma tecla e depois um temporizador vencido, e o
    que é ocioso (after_idle) só roda com a fila de teclas vazia. Cada passo custa
    STEP_MS de desenho, e cada tecla, KEY_MS."""
    STEP_MS = 30
    KEY_MS = 1     # tratar uma tecla também custa: é Python rodando

    def __init__(self):
        self.now, self.timers, self.idle, self.keys, self.count = 0.0, [], [], [], 0

    def after(self, ms, callback):
        self.count += 1
        self.timers.append((self.now + ms, self.count, callback))
        return self.count

    def after_idle(self, callback):
        self.count += 1
        self.idle.append((self.count, callback))
        return self.count

    def after_cancel(self, ident):
        self.timers = [t for t in self.timers if t[1] != ident]
        self.idle = [i for i in self.idle if i[0] != ident]

    def key(self, at, callback):
        self.keys.append((at, self.count, callback)); self.count += 1

    def run_until(self, until):
        while self.now <= until:
            arrived = sorted(k for k in self.keys if k[0] <= self.now)
            due = sorted(t for t in self.timers if t[0] <= self.now)
            if arrived:
                self.keys.remove(arrived[0]); arrived[0][2]()
                self.now += self.KEY_MS
                due = sorted(t for t in self.timers if t[0] <= self.now)
                if due:
                    self.timers.remove(due[0]); due[0][2]()
                continue
            if due:
                self.timers.remove(due[0]); due[0][2](); continue
            if self.idle:
                batch, self.idle = self.idle, []
                for _, callback in batch:
                    callback()
                continue
            upcoming = [k[0] for k in self.keys] + [t[0] for t in self.timers]
            if not upcoming:
                break
            self.now = max(self.now, min(upcoming))
        self.now = max(self.now, until)


def hold(stepper_cls, release_at, big=False, repeat_from=500, every=30):
    """Seta apertada em 0 e solta em `release_at`; a repetição do teclado chega
    como soltar e apertar no mesmo instante, a cada `every` ms."""
    root, steps = FakeTk(), []
    def step(key, large):
        steps.append(root.now); root.now += FakeTk.STEP_MS
    stepper = stepper_cls(root, step)
    root.key(0, lambda: stepper.press("Right", big))
    for t in range(repeat_from, release_at, every):
        root.key(t, lambda: stepper.release("Right"))
        root.key(t, lambda: stepper.press("Right", big))
    root.key(release_at, lambda: stepper.release("Right"))
    return root, stepper, steps


class TestStepper:
    def test_a_tap_moves_one_step(self):
        root, _, steps = hold(Stepper, release_at=100)

        root.run_until(3000)

        assert len(steps) == 1

    def test_holding_moves_and_releasing_stops_at_once(self):
        # O caso medido: 3,6 s segurando, e o vídeo seguia 2,9 s depois de soltar.
        root, _, steps = hold(Stepper, release_at=3600)

        root.run_until(30_000)

        assert len(steps) > 80
        assert max(steps) <= 3600 + FakeTk.STEP_MS

    def test_the_longer_the_hold_the_same_stop(self):
        # Com a fila crescendo, o atraso aumentava com o tempo segurando.
        root, _, steps = hold(Stepper, release_at=8000)

        root.run_until(60_000)

        assert max(steps) <= 8000 + FakeTk.STEP_MS

    def test_big_steps_held_are_paced(self):
        root, _, steps = hold(Stepper, release_at=1300, big=True)

        root.run_until(10_000)

        assert 4 <= len(steps) <= 7

    def test_losing_focus_stops_the_walk(self):
        # Sem foco, a janela não recebe mais nem a repetição nem o soltar da tecla.
        root, stepper, steps = hold(Stepper, release_at=10_000, repeat_from=10_000)

        root.run_until(800)
        stepper.stop()
        root.run_until(3000)

        assert max(steps) <= 800 + FakeTk.STEP_MS
