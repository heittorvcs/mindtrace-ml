"""Revisão de duas marcações da mesma sessão, assistindo ao vídeo.

Mostra, quadro a quadro, onde duas marcações de um comportamento concordam e onde
divergem: a antiga e a re-rotulagem às cegas, ou dois observadores. A faixa sob o
vídeo muda de cor conforme o quadro atual, e `n`/`p` pulam de uma discordância
para outra. Sem `--output`, só leitura: nada é gravado.

Os episódios seguem a mesma definição da análise: descidas breves fundidas
(MERGE_GAP_SEC), e os trechos "não dá para ver" de qualquer das duas ficam fora
da comparação. Um episódio está "nas duas" se cruza algum episódio da outra.

Uso:
    python scripts/review_labels.py --session TT_9_cam2 --videos ../videos_lab/arenas
    python scripts/review_labels.py --session TT_15_cam1 --videos ../videos_lab/arenas \\
        --a data/relabel/session_labels_r1_original.csv --names antiga,nova

Teclas:
    espaço = tocar/pausar   ← → = um quadro   shift+← → = 2 s   ↑ ↓ = velocidade
    n / p = próxima / anterior discordância (começa 1 s antes e toca)
    clique na linha do tempo = pula para lá   q = sair

**Arbitragem** (`--output consenso.csv`): uma terceira linha, o consenso, que é
gravado a cada decisão. Cada trecho em que alguma das marcações tem um episódio
é um grupo a decidir. Os grupos em que as duas concordam já entram preenchidos
com a versão de `--prefill`; as discordâncias ficam pendentes, e `n`/`p` andam
só por elas. Reabrir continua de onde parou.

    python scripts/review_labels.py --session TT_15_cam1 --videos ../videos_lab/arenas \\
        --a data/relabel/session_labels_r1_original.csv --names antiga,nova \\
        --output data/relabel/session_labels_arbitragem.csv

    1 / 2 = manter a versão da primeira / segunda marcação no grupo do cursor
    0 = nenhuma das duas   r = abre/fecha um episódio novo no consenso
    x = abre/fecha "não dá para ver"   delete = apaga o intervalo do consenso sob o cursor
    volta = desfaz   shift+n / shift+p = próximo / anterior grupo, pendente ou não
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from label_session import SCALE, SPEEDS, ZOOM_SEC, Annotation, Stepper, Video, replace_rows  # noqa: E402
from mindtrace_ml.behavior_model import bouts, label_mask  # noqa: E402

# As mesmas cores do gráfico de comparação: azul nas duas, laranja só na
# primeira, verde-água só na segunda.
STATUS_COLORS = {"both": "#2a78d6", "only_a": "#eb6834", "only_b": "#1baf7a"}
UNSCORABLE_COLOR = "#7f8c8d"
IDLE_COLOR = "#e1e0d9"
CONSENSUS_COLOR = "#6250d6"
PENDING_COLOR = "#f1c40f"


def compare_episodes(a: pd.DataFrame, b: pd.DataFrame, behavior: str, n_frames: int, fps: float):
    """Episódios das duas marcações com o status de concordância, mais as máscaras por quadro.

    Devolve (episodes, masks). Cada episódio é {lane, start, end, status}, com
    lane "a" ou "b" e status "both", "only_a" ou "only_b". As máscaras são
    "a", "b" e "unscorable", uma posição por quadro do vídeo.
    """
    frames = np.arange(n_frames)
    unscorable = label_mask(a, "unscorable", frames, fps) | label_mask(b, "unscorable", frames, fps)
    masks = {lane: label_mask(labels, behavior, frames, fps) & ~unscorable
             for lane, labels in (("a", a), ("b", b))}
    episodes = []
    for lane, other in (("a", "b"), ("b", "a")):
        for start, end in bouts(masks[lane]):
            shared = masks[other][start:end + 1].any()
            episodes.append({"lane": lane, "start": int(start), "end": int(end),
                             "status": "both" if shared else f"only_{lane}"})
    masks["unscorable"] = unscorable
    return sorted(episodes, key=lambda e: (e["start"], e["lane"])), masks


def disagreements(episodes) -> list[tuple[int, int]]:
    """Os episódios que só uma das marcações tem, em ordem de início."""
    return [(e["start"], e["end"]) for e in episodes if e["status"] != "both"]


def frame_status(masks, frame: int) -> str | None:
    """O que cada marcação diz neste quadro. Pode divergir do status do episódio:
    nas bordas de um episódio "nas duas", uma marcação começa antes da outra."""
    if masks["unscorable"][frame]:
        return "unscorable"
    a, b = bool(masks["a"][frame]), bool(masks["b"][frame])
    if a and b:
        return "both"
    if a or b:
        return "only_a" if a else "only_b"
    return None



def decision_groups(a: pd.DataFrame, b: pd.DataFrame, behavior: str, n_frames: int, fps: float):
    """Trechos a decidir na arbitragem: episódios das duas marcações que se cruzam viram um grupo.

    Cada grupo é {start, end, status, a, b}: o trecho coberto, o status de
    concordância e as marcações originais de cada fonte que caem nele, como
    foram feitas (antes de fundir as descidas breves). Manter uma versão é
    copiar essas marcações.
    """
    episodes, _ = compare_episodes(a, b, behavior, n_frames, fps)
    groups = []
    for episode in sorted(episodes, key=lambda e: e["start"]):
        if groups and episode["start"] <= groups[-1]["end"]:
            groups[-1]["end"] = max(groups[-1]["end"], episode["end"])
            groups[-1]["lanes"].add(episode["lane"])
        else:
            groups.append({"start": episode["start"], "end": episode["end"], "lanes": {episode["lane"]}})
    for group in groups:
        lanes = group.pop("lanes")
        group["status"] = "both" if len(lanes) == 2 else f"only_{lanes.pop()}"
        for lane, labels in (("a", a), ("b", b)):
            mine = labels[(labels.behavior == behavior) & (labels.start_frame <= group["end"])
                          & (labels.end_frame >= group["start"])]
            group[lane] = sorted(zip(mine.start_frame.astype(int), mine.end_frame.astype(int)))
    return groups


def initial_consensus(groups, prefill: str, behavior: str):
    """Consenso de partida: os grupos em que as duas concordam, na versão de `prefill`.

    Devolve (intervalos, decididos). Com prefill "none", nada entra e tudo fica
    pendente.
    """
    intervals, decided = [], set()
    if prefill != "none":
        for index, group in enumerate(groups):
            if group["status"] == "both":
                intervals += [(behavior, start, end) for start, end in group[prefill]]
                decided.add(index)
    return intervals, decided


def group_at(groups, frame: int, lead: int) -> int | None:
    """O grupo do cursor: o que contém o quadro, com `lead` de folga dos dois lados.

    A folga existe porque `n` para 1 s antes do começo: logo depois do pulo, o
    grupo que se quer decidir ainda está à frente do cursor.
    """
    near = [(abs(frame - (g["start"] + g["end"]) / 2), index) for index, g in enumerate(groups)
            if g["start"] - lead <= frame <= g["end"] + lead]
    return min(near)[1] if near else None


def span_key(group) -> str:
    return f"{group['start']}-{group['end']}"


class Consensus(Annotation):
    """Annotation com a troca de um trecho inteiro, desfeita de uma vez."""

    def replace(self, start: int, end: int, intervals, behavior: str) -> None:
        removed = [i for i in self.intervals if i[0] == behavior and i[1] <= end and i[2] >= start]
        self.intervals = [i for i in self.intervals if i not in removed]
        added = [(behavior, int(a), int(b)) for a, b in intervals]
        self.intervals += added
        self.history.append(("replace", removed, added))

    def undo(self) -> str | None:
        if self.history and self.history[-1][0] == "replace":
            _, removed, added = self.history.pop()
            for item in added:
                self.intervals.remove(item)
            self.intervals += removed
            return "replace"
        return super().undo()


class App:
    def __init__(self, root, video: Video, episodes, masks, session_id: str, behavior: str, names):
        import tkinter as tk
        import tkinter.font as tkfont

        self.root, self.video = root, video
        self.episodes, self.masks = episodes, masks
        self.todo = disagreements(episodes)
        self.session_id, self.behavior, self.names = session_id, behavior, names
        self.playing = False
        self.after_id = None
        self.speed = 1.0
        # Espaço do nome de cada linha, à esquerda das linhas do tempo.
        self.margin = 16 + max(tkfont.Font(family="Consolas", size=9).measure(name) for name in names)

        width, height = (side * SCALE for side in video.size)
        self.width = width
        self.screen = tk.Canvas(root, width=width, height=height, bg="black", highlightthickness=0)
        self.screen.pack()
        self.banner = tk.Canvas(root, width=width, height=12, highlightthickness=0, bg=IDLE_COLOR)
        self.banner.pack()
        self.status = tk.Label(root, font=("Consolas", 11), anchor="w", justify="left")
        self.status.pack(fill="x", padx=8, pady=(6, 0))
        self.full = tk.Canvas(root, width=width, height=46, bg="#f4f4f4", highlightthickness=0)
        self.full.pack(pady=(6, 0))
        self.zoom = tk.Canvas(root, width=width, height=46, bg="#ffffff", highlightthickness=0)
        self.zoom.pack(pady=(4, 0))
        self.legend = tk.Label(root, font=("Consolas", 10), justify="left", anchor="w", text=(
            f"azul = nas duas   laranja = só na {names[0]}   verde-água = só na {names[1]}   "
            "cinza = não dá para ver\n"
            "espaco=tocar/pausar  <- ->=1 quadro  shift+<- ->=2 s  cima/baixo=velocidade  "
            "n/p=próxima/anterior discordância  q=sair"))
        self.legend.pack(fill="x", padx=8, pady=6)

        self.image_item = self.screen.create_image(0, 0, anchor="nw")
        self.photo = None
        self.full.bind("<Button-1>", lambda e: self.seek_click(e, 0, self.video.count))
        self.zoom.bind("<Button-1>", lambda e: self.seek_click(e, *self.zoom_range()))
        root.bind("<Key>", self.on_key)
        root.bind("<KeyRelease>", self.on_key_release)
        self.stepper = Stepper(root, self.step)
        root.bind("<FocusOut>", self.stepper.stop)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.title(f"Revisão de marcações — {session_id} ({behavior})")

        self.video.read(0)
        self.render()

    # --- navegação ---------------------------------------------------------
    def step(self, key: str, big: bool) -> None:
        self.play(False)
        jump = int(round(2 * self.video.fps)) if big else 1
        self.seek(self.video.position + (jump if key == "Right" else -jump))

    def on_key_release(self, event) -> None:
        if event.keysym in ("Right", "Left"):
            self.stepper.release(event.keysym)

    def seek(self, frame: int) -> None:
        self.video.read(frame)
        self.render()

    def seek_click(self, event, first: int, last: int) -> None:
        share = (event.x - self.margin) / max(1, self.width - self.margin)
        self.seek(first + int(np.clip(share, 0, 1) * (last - first)))

    def zoom_range(self) -> tuple[int, int]:
        half = int(ZOOM_SEC * self.video.fps / 2)
        first = max(0, self.video.position - half)
        return first, min(self.video.count, first + 2 * half)

    def play(self, playing: bool) -> None:
        self.playing = playing
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if playing:
            self.after_id = self.root.after(0, self.tick)

    def tick(self) -> None:
        self.after_id = None
        if not self.playing:
            return
        if not self.video.advance(max(1, int(round(self.speed)))):
            self.playing = False
        self.render()
        if self.playing:
            self.after_id = self.root.after(int(1000 / (self.video.fps * min(self.speed, 1.0))), self.tick)

    def jump(self, forward: bool) -> None:
        # Começa 1 s antes: é o começo do movimento que decide se houve rearing.
        lead = int(round(self.video.fps))
        frame = self.video.position
        if forward:
            target = next((s for s, _ in self.todo if s - lead > frame), None)
        else:
            target = next((s for s, _ in reversed(self.todo) if s - lead < frame - 1), None)
        if target is None:
            self.play(False)
            self.render("não há mais discordâncias nessa direção")
            return
        self.seek(max(0, target - lead))
        self.play(True)

    def on_key(self, event) -> None:
        key, shift = event.keysym, bool(event.state & 0x1)
        if key == "space":
            self.play(not self.playing)
        elif key in ("Right", "Left"):
            self.stepper.press(key, shift)
            return
        elif key in ("Up", "Down"):
            index = SPEEDS.index(self.speed) + (1 if key == "Up" else -1)
            self.speed = SPEEDS[int(np.clip(index, 0, len(SPEEDS) - 1))]
        elif key.lower() in ("n", "p"):
            self.jump(key.lower() == "n")
            return
        elif key.lower() == "q":
            self.quit()
            return
        self.render()

    # --- desenho -----------------------------------------------------------
    def render(self, message: str = "") -> None:
        from PIL import Image, ImageTk

        position = self.video.position
        rgb = cv2.cvtColor(self.video.image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.screen.itemconfig(self.image_item, image=self.photo)

        state = frame_status(self.masks, position)
        self.banner.configure(bg=STATUS_COLORS.get(state, UNSCORABLE_COLOR if state == "unscorable" else IDLE_COLOR))
        now = {"both": f"as duas marcam {self.behavior}",
               "only_a": f"só a {self.names[0]} marca {self.behavior}",
               "only_b": f"só a {self.names[1]} marca {self.behavior}",
               "unscorable": "trecho marcado como 'não dá para ver'",
               None: "nenhuma das duas marca"}[state]
        seconds, total = position / self.video.fps, self.video.count / self.video.fps
        done = sum(s <= position for s, _ in self.todo)
        self.status.config(text=(
            f"{self.session_id}   {self.clock(seconds)} / {self.clock(total)}   quadro {position}   "
            f"{self.speed:g}x {'>' if self.playing else '||'}   discordância {done} de {len(self.todo)}\n"
            f"agora: {now}   {message}"))

        self.draw_timeline(self.full, 0, self.video.count)
        self.draw_timeline(self.zoom, *self.zoom_range())

    @staticmethod
    def clock(seconds: float) -> str:
        return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"

    def draw_timeline(self, canvas, first: int, last: int) -> None:
        canvas.delete("all")
        span = max(1, last - first)

        def x(frame):
            return self.margin + (frame - first) / span * (self.width - self.margin)

        for row, (lane, name) in enumerate((("a", self.names[0]), ("b", self.names[1]))):
            top = 8 + 16 * row
            canvas.create_text(4, top + 6, text=name, anchor="w", font=("Consolas", 9), fill="#777777")
            canvas.create_rectangle(self.margin, top, self.width, top + 12, fill="#ececea", width=0)
            for episode in self.episodes:
                if episode["lane"] == lane and episode["end"] >= first and episode["start"] <= last:
                    left = x(max(episode["start"], first))
                    right = max(x(min(episode["end"] + 1, last)), left + 2)
                    canvas.create_rectangle(left, top, right, top + 12,
                                            fill=STATUS_COLORS[episode["status"]], width=0)
        for start, end in bouts(self.masks["unscorable"]):
            if end >= first and start <= last:
                canvas.create_rectangle(x(max(start, first)), 6, x(min(end + 1, last)), 38,
                                        fill=UNSCORABLE_COLOR, stipple="gray50", width=0)
        cursor = x(self.video.position)
        canvas.create_line(cursor, 0, cursor, 46, fill="black", width=2)

    def quit(self) -> None:
        self.play(False)
        self.root.destroy()


class Arbitration(App):
    """O revisor com uma terceira linha editável, o consenso, gravado a cada decisão."""

    def __init__(self, root, video: Video, episodes, masks, groups, consensus: Consensus, decided: set,
                 session_id: str, behavior: str, names, output: Path, progress: Path, annotator: str,
                 sources: dict):
        import tkinter.font as tkfont

        self.groups, self.consensus, self.decided = groups, consensus, decided
        self.decisions = []   # (grupo, já estava decidido) de cada troca, para o desfazer
        self.output, self.progress_path, self.annotator, self.sources = output, progress, annotator, sources
        self.lead = int(round(video.fps))
        self.focus = None     # grupo do último n/p: o cursor para antes dele, às vezes ainda no anterior
        super().__init__(root, video, episodes, masks, session_id, behavior, names)
        self.margin = 16 + max(tkfont.Font(family="Consolas", size=9).measure(name)
                               for name in (*names, "consenso"))
        for canvas in (self.full, self.zoom):
            canvas.configure(height=62)
        root.title(f"Arbitragem — {session_id} ({behavior})")
        self.legend.config(text=(
            f"1 = manter a {names[0]}   2 = manter a {names[1]}   0 = nenhuma   "
            f"r = {behavior} novo   x = não dá para ver   delete = apagar do consenso   volta = desfazer\n"
            "n/p = próxima/anterior pendente   shift+n/p = qualquer grupo   espaco=tocar/pausar  "
            "<- ->=1 quadro  shift+<- ->=2 s  cima/baixo=velocidade  q=salvar e sair\n"
            f"azul = nas duas   laranja = só na {names[0]}   verde-água = só na {names[1]}   "
            "roxo = consenso   amarelo = pendente"))
        self.render()

    def current(self) -> int | None:
        """O grupo a decidir: o do último pulo enquanto o cursor estiver perto dele."""
        frame = self.video.position
        if self.focus is not None:
            group = self.groups[self.focus]
            if group["start"] - self.lead <= frame <= group["end"] + self.lead:
                return self.focus
            self.focus = None
        return group_at(self.groups, frame, self.lead)

    def pending(self) -> list[int]:
        return [index for index in range(len(self.groups)) if index not in self.decided]

    # --- decisões ------------------------------------------------------------
    def choose(self, lane: str | None) -> str:
        index = self.current()
        if index is None:
            return "nenhum grupo perto do cursor"
        group = self.groups[index]
        self.consensus.replace(group["start"], group["end"], group[lane] if lane else [], self.behavior)
        self.decisions.append((index, index in self.decided))
        self.decided.add(index)
        self.save()
        kept = f"a {self.names[0 if lane == 'a' else 1]}" if lane else "nenhuma das duas"
        return f"grupo {index + 1}: mantida {kept}"

    def touch(self, start: int, end: int) -> None:
        """Edição feita à mão decide os grupos que ela cruza."""
        for index, group in enumerate(self.groups):
            if group["start"] <= end and group["end"] >= start:
                self.decided.add(index)

    def on_key(self, event) -> None:
        key, frame = event.keysym, self.video.position
        if key in ("1", "2", "0", "KP_1", "KP_2", "KP_0"):
            message = self.choose({"1": "a", "2": "b", "0": None}[key[-1]])
        elif key in ("N", "P"):
            self.jump(key == "N", everything=True)
            return
        elif key.lower() in ("r", "x"):
            behavior = self.behavior if key.lower() == "r" else "unscorable"
            was_open = behavior in self.consensus.open
            self.consensus.toggle(behavior, frame)
            if was_open:
                self.touch(*self.consensus.intervals[-1][1:])
            message = f"{behavior} {'fechado' if was_open else 'aberto'} no quadro {frame}"
            self.save()
        elif key == "Delete":
            removed = self.consensus.remove_at(frame)
            if removed:
                self.touch(removed[1], removed[2])
            message = f"apagado do consenso: {removed[1]}–{removed[2]}" if removed else "nada do consenso sob o cursor"
            self.save()
        elif key == "BackSpace":
            action = self.consensus.undo()
            if action == "replace":
                index, was_decided = self.decisions.pop()
                if not was_decided:
                    self.decided.discard(index)
            message = f"desfeito: {action}" if action else "nada para desfazer"
            self.save()
        else:
            super().on_key(event)
            return
        self.render(message)

    def jump(self, forward: bool, everything: bool = False) -> None:
        frame = self.video.position
        options = [index for index in range(len(self.groups)) if everything or index not in self.decided]
        start = lambda index: self.groups[index]["start"]
        if forward:
            target = next((i for i in options if start(i) - self.lead > frame), None)
        else:
            target = next((i for i in reversed(options) if start(i) - self.lead < frame - 1), None)
        if target is None:
            self.play(False)
            self.render("não há mais grupos pendentes nessa direção" if not everything else "fim dos grupos")
            return
        self.focus = target
        self.seek(max(0, start(target) - self.lead))
        self.play(True)

    # --- desenho -----------------------------------------------------------
    def render(self, message: str = "") -> None:
        super().render(message)
        if not hasattr(self, "legend"):
            return
        frame = self.video.position
        inside = any(b == self.behavior and s <= frame <= e for b, s, e in self.consensus.intervals)
        index = self.current()
        if index is None:
            where = "fora dos grupos"
        else:
            status = {"both": "nas duas", "only_a": f"só na {self.names[0]}",
                      "only_b": f"só na {self.names[1]}"}[self.groups[index]["status"]]
            where = (f"grupo {index + 1} de {len(self.groups)} ({status})"
                     f"{'' if index in self.decided else ' PENDENTE'}")
        opened = ", ".join(self.consensus.open)
        self.status.config(text=self.status.cget("text") + (
            f"\nconsenso: {'marca' if inside else 'não marca'} {self.behavior}"
            f"{'   aberto: ' + opened if opened else ''}   |   {where}   |   pendentes: {len(self.pending())}"))

    def draw_timeline(self, canvas, first: int, last: int) -> None:
        if not hasattr(self, "groups"):
            return
        canvas.delete("all")
        span = max(1, last - first)

        def x(frame):
            return self.margin + (frame - first) / span * (self.width - self.margin)

        def bar(start, end, top, **style):
            left = x(max(start, first))
            canvas.create_rectangle(left, top, max(x(min(end + 1, last)), left + 2), top + 12, width=0, **style)

        for row, name in enumerate((*self.names, "consenso")):
            top = 8 + 16 * row
            canvas.create_text(4, top + 6, text=name, anchor="w", font=("Consolas", 9), fill="#777777")
            canvas.create_rectangle(self.margin, top, self.width, top + 12, fill="#ececea", width=0)
        for start, end in bouts(self.masks["unscorable"]):
            if end >= first and start <= last:
                bar(start, end, 8, fill=UNSCORABLE_COLOR, stipple="gray50")
                bar(start, end, 24, fill=UNSCORABLE_COLOR, stipple="gray50")
        for episode in self.episodes:
            if episode["end"] >= first and episode["start"] <= last:
                bar(episode["start"], episode["end"], 8 if episode["lane"] == "a" else 24,
                    fill=STATUS_COLORS[episode["status"]])
        for index in self.pending():
            group = self.groups[index]
            if group["end"] >= first and group["start"] <= last:
                bar(group["start"], group["end"], 40, fill=PENDING_COLOR, stipple="gray50")
        for behavior, start, end in self.consensus.intervals:
            if end >= first and start <= last:
                bar(start, end, 40, fill=CONSENSUS_COLOR if behavior == self.behavior else UNSCORABLE_COLOR)
        for behavior, start in self.consensus.open.items():
            low, high = sorted((start, self.video.position))
            bar(low, high, 40, fill=CONSENSUS_COLOR if behavior == self.behavior else UNSCORABLE_COLOR,
                stipple="gray50")
        current = self.current()
        if current is not None:
            group = self.groups[current]
            canvas.create_rectangle(x(max(group["start"], first)) - 2, 5,
                                    x(min(group["end"] + 1, last)) + 2, 55, outline="#333333", width=1)
        cursor = x(self.video.position)
        canvas.create_line(cursor, 0, cursor, 62, fill="black", width=2)

    # --- persistência ------------------------------------------------------
    def save(self) -> None:
        replace_rows(self.output, self.consensus.rows(self.session_id, self.annotator),
                     self.session_id, self.annotator)
        record = pd.DataFrame([{"session_id": self.session_id, "annotator": self.annotator, **self.sources,
                                "groups": len(self.groups), "pending": len(self.pending()),
                                "decided": ";".join(span_key(self.groups[i]) for i in sorted(self.decided))}])
        replace_rows(self.progress_path, record, self.session_id, self.annotator)

    def quit(self) -> None:
        self.play(False)
        # Ao contrário da rotulagem, um episódio aberto não é fechado no quadro da saída:
        # quase sempre é tecla apertada sem querer, e fechá-lo inventaria um episódio.
        for behavior, start in list(self.consensus.open.items()):
            print(f"descartado, estava aberto ao sair: {behavior} desde o quadro {start}")
        self.consensus.open.clear()
        self.save()
        self.root.destroy()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--a", type=Path, default=ROOT / "data" / "session_labels.csv",
                        help="primeira marcação (padrão: os rótulos oficiais)")
    parser.add_argument("--b", type=Path, default=ROOT / "data" / "relabel" / "session_labels_r2.csv",
                        help="segunda marcação (padrão: a re-rotulagem às cegas)")
    parser.add_argument("--names", default="oficial,re-rotulagem",
                        help="nomes das duas marcações, separados por vírgula")
    parser.add_argument("--behavior", default="rearing")
    parser.add_argument("--output", type=Path, default=None,
                        help="arquivo do consenso: liga a arbitragem (sem ele, só leitura)")
    parser.add_argument("--progress", type=Path, default=ROOT / "data" / "relabel" / "arbitragem_progresso.csv",
                        help="com --output: grupos já decididos e de onde veio cada marcação")
    parser.add_argument("--prefill", choices=["a", "b", "none"], default="b",
                        help="com --output: versão que preenche os grupos em que as duas concordam "
                             "(padrão b, a marcação mais recente)")
    parser.add_argument("--annotator", default="heittor", help="com --output: anotador gravado no consenso")
    args = parser.parse_args()

    path = args.videos / f"{args.session}.mp4"
    if not path.exists():
        print(f"vídeo não encontrado: {path}")
        return 1
    names = tuple(args.names.split(","))
    labels = []
    for source in (args.a, args.b):
        table = pd.read_csv(source, encoding="utf-8-sig")
        labels.append(table[table.session_id == args.session])
    empty = [name for name, table in zip(names, labels) if table.empty]
    if len(empty) == 2:
        print(f"{args.session} sem marcações nas duas fontes")
        return 1
    if empty:
        # Uma sessão assistida sem nenhum episódio é legítima (rato imóvel), e é aí que
        # os alarmes falsos da outra fonte ficam mais à vista.
        print(f"{args.session}: nenhuma marcação em '{empty[0]}' — tudo o que aparecer é só da outra")

    video = Video(path)
    episodes, masks = compare_episodes(*labels, args.behavior, video.count, video.fps)
    for lane, name in zip("ab", names):
        mine = [e for e in episodes if e["lane"] == lane]
        alone = sum(e["status"] != "both" for e in mine)
        print(f"{name}: {len(mine)} episódios de {args.behavior}, {alone} só nela")

    import tkinter as tk
    root = tk.Tk()
    if not args.output:
        App(root, video, episodes, masks, args.session, args.behavior, names)
        root.mainloop()
        return 0

    groups = decision_groups(*labels, args.behavior, video.count, video.fps)
    saved = pd.read_csv(args.output, encoding="utf-8-sig") if args.output.exists() else pd.DataFrame()
    mine = saved[(saved.session_id == args.session) & (saved.annotator == args.annotator)] if len(saved) else saved
    if len(mine):
        intervals = list(zip(mine.behavior, mine.start_frame.astype(int), mine.end_frame.astype(int)))
        progress = pd.read_csv(args.progress, encoding="utf-8-sig") if args.progress.exists() else pd.DataFrame()
        record = (progress[(progress.session_id == args.session) & (progress.annotator == args.annotator)]
                  if len(progress) else progress)
        done = set(str(record.decided.iloc[0]).split(";")) if len(record) and pd.notna(record.decided.iloc[0]) else set()
        decided = {index for index, group in enumerate(groups) if span_key(group) in done}
        print(f"continuando a arbitragem: {len(intervals)} intervalos no consenso")
    else:
        intervals, decided = initial_consensus(groups, args.prefill, args.behavior)
    pending = len(groups) - len(decided)
    print(f"{len(groups)} grupos a decidir: {sum(g['status'] == 'both' for g in groups)} nas duas, "
          f"{pending} pendentes")

    sources = {"source_a": str(args.a), "source_b": str(args.b), "names": args.names, "prefill": args.prefill}
    app = Arbitration(root, video, episodes, masks, groups, Consensus(intervals), decided, args.session,
                      args.behavior, names, args.output, args.progress, args.annotator, sources)
    root.mainloop()
    episodes_now = bouts(label_mask(app.consensus.rows(args.session, args.annotator), args.behavior,
                                    np.arange(video.count), video.fps))
    print(f"consenso salvo em {args.output}: {len(episodes_now)} episódios de {args.behavior}, "
          f"{len(app.pending())} grupos pendentes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
