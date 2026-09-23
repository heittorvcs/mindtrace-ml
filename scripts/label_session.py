"""Rotulagem de sessões completas: início e fim de cada comportamento.

Os clipes de 2 s respondem "tem ou não tem", e isso bastou para a triagem. A
camada de pose precisa de mais: quanto tempo durou cada grooming, quantos
rearings houve — é o que o laboratório reporta. E uma sessão inteira é
representativa por construção: nenhum sorteio decide o que entra.

Cada tecla de comportamento abre o comportamento no quadro atual; apertar de novo
fecha. Vários podem estar abertos ao mesmo tempo. Movimento e parado não se
marcam: saem da camada de movimento, calculada da posição.

Às cegas como os clipes: a janela não mostra pose, triagem nem modelo.

Salva a cada marcação. Reabrir a mesma sessão continua de onde parou; o quanto
já foi assistido fica em `session_progress.csv`, porque ausência de marcação só
significa ausência de comportamento no trecho que foi de fato visto.

Uso:
    python scripts/label_session.py --session TT_22_cam1 --videos .../arenas

Teclas:
    espaço = tocar/pausar   ← → = um quadro   shift+← → = 2 s   ↑ ↓ = velocidade
    e = explorando objeto   g = grooming   s = sniffing fora do objeto
    r = rearing   o = outro comportamento notável   x = não dá para ver
    volta = desfaz a última ação   delete = apaga o intervalo sob o cursor
    clique na linha do tempo = pula para lá   q = salvar e sair
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.schema import LABEL_COLUMNS  # noqa: E402
from mindtrace_ml.triage_model import animal_of  # noqa: E402

KEYS = {
    "e": "object_interaction",
    "g": "grooming",
    "s": "sniffing",
    "r": "rearing",
    "o": "other",
    "x": "unscorable",
}
SHORT = {"object_interaction": "explorando", "grooming": "grooming", "sniffing": "sniffing",
         "rearing": "rearing", "other": "outro", "unscorable": "nao da p/ ver"}
COLORS = {"object_interaction": "#2e86de", "grooming": "#e67e22", "sniffing": "#16a085",
          "rearing": "#8e44ad", "other": "#c0392b", "unscorable": "#7f8c8d"}
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
SCALE = 2
ZOOM_SEC = 20


class Annotation:
    """Intervalos de uma sessão, com desfazer. Sem interface: testável à parte."""

    def __init__(self, intervals=None):
        self.intervals = [tuple(i) for i in (intervals or [])]   # (comportamento, início, fim)
        self.open = {}                                          # comportamento -> início
        self.history = []

    def toggle(self, behavior: str, frame: int) -> None:
        if behavior in self.open:
            start = self.open.pop(behavior)
            low, high = sorted((start, frame))
            self.intervals.append((behavior, low, high))
            self.history.append(("close", behavior, start))
        else:
            self.open[behavior] = frame
            self.history.append(("open", behavior, frame))

    def remove_at(self, frame: int) -> tuple | None:
        """Apaga o intervalo mais recente que contém o quadro."""
        for index in range(len(self.intervals) - 1, -1, -1):
            behavior, start, end = self.intervals[index]
            if start <= frame <= end:
                removed = self.intervals.pop(index)
                self.history.append(("remove", index, removed))
                return removed
        return None

    def undo(self) -> str | None:
        if not self.history:
            return None
        action, first, second = self.history.pop()
        if action == "open":
            self.open.pop(first, None)
        elif action == "close":
            # Reabre no início original: o fim estava errado, não o começo.
            self.intervals.pop()
            self.open[first] = second
        elif action == "remove":
            self.intervals.insert(first, second)
        return action

    def close_all(self, frame: int) -> list[str]:
        closing = list(self.open)
        for behavior in closing:
            self.toggle(behavior, frame)
        return closing

    def rows(self, session_id: str, annotator: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "session_id": session_id, "animal_id": animal_of(session_id), "field": 0,
            "behavior": behavior, "start_frame": int(start), "end_frame": int(end),
            "annotator": annotator,
        } for behavior, start, end in sorted(self.intervals, key=lambda i: (i[1], i[0]))],
            columns=list(LABEL_COLUMNS))


def replace_rows(path: Path, fresh: pd.DataFrame, session_id: str, annotator: str) -> None:
    """Troca as linhas desta sessão e anotador, preservando as das outras."""
    if path.exists():
        previous = pd.read_csv(path, encoding="utf-8-sig")
        keep = ~((previous.session_id == session_id) & (previous.annotator == annotator))
        fresh = pd.concat([previous[keep], fresh], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path, index=False, encoding="utf-8-sig")


class Video:
    def __init__(self, path: Path):
        self.capture = cv2.VideoCapture(str(path))
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 29.97
        self.position = -1
        self.image = None

    def read(self, index: int) -> bool:
        index = int(np.clip(index, 0, self.count - 1))
        if index != self.position + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.capture.read()
        if ok:
            self.position, self.image = index, image
        return ok

    def advance(self, frames: int) -> bool:
        # Em velocidade alta, os quadros pulados são só decodificados, sem conversão.
        for _ in range(frames - 1):
            if not self.capture.grab():
                return False
            self.position += 1
        return self.read(self.position + 1)


class App:
    def __init__(self, root, video: Video, annotation: Annotation, session_id: str,
                 annotator: str, output: Path, progress: Path, watched: int):
        import tkinter as tk

        self.root, self.video, self.annotation = root, video, annotation
        self.session_id, self.annotator = session_id, annotator
        self.output, self.progress = output, progress
        self.watched = watched
        self.playing = False
        self.after_id = None
        self.speed = 1.0
        self.message = ""

        width, height = (side * SCALE for side in video.size)
        self.screen = tk.Canvas(root, width=width, height=height, bg="black", highlightthickness=0)
        self.screen.pack()
        self.status = tk.Label(root, font=("Consolas", 11), anchor="w", justify="left")
        self.status.pack(fill="x", padx=8, pady=(6, 0))
        self.timeline_width = width
        rows = len(KEYS)
        self.full = tk.Canvas(root, width=width, height=14 * rows + 16, bg="#f4f4f4", highlightthickness=0)
        self.full.pack(pady=(6, 0))
        self.zoom = tk.Canvas(root, width=width, height=14 * rows + 16, bg="#ffffff", highlightthickness=0)
        self.zoom.pack(pady=(4, 0))
        tk.Label(root, font=("Consolas", 10), justify="left", anchor="w", text=(
            "e=explorando  g=grooming  s=sniffing  r=rearing  o=outro  x=nao da p/ ver"
            "   (aperte para abrir, de novo para fechar)\n"
            "espaco=tocar/pausar  <- ->=1 quadro  shift+<- ->=2 s  cima/baixo=velocidade  "
            "volta=desfazer  delete=apagar intervalo  q=sair")).pack(fill="x", padx=8, pady=6)

        self.photo = None
        self.image_item = self.screen.create_image(0, 0, anchor="nw")
        self.full.bind("<Button-1>", lambda e: self.seek_click(e, 0, self.video.count))
        self.zoom.bind("<Button-1>", lambda e: self.seek_click(e, *self.zoom_range()))
        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.title(f"Rotulagem de sessão — {session_id}")

        self.video.read(0)
        self.render()

    # --- navegação ---------------------------------------------------------
    def seek(self, frame: int) -> None:
        self.video.read(frame)
        self.render()

    def seek_click(self, event, first: int, last: int) -> None:
        self.seek(first + int(event.x / self.timeline_width * (last - first)))

    def zoom_range(self) -> tuple[int, int]:
        half = int(ZOOM_SEC * self.video.fps / 2)
        first = max(0, self.video.position - half)
        return first, min(self.video.count, first + 2 * half)

    def play(self, playing: bool) -> None:
        # Cancela o passo agendado: sem isso, pausar e tocar rápido deixaria dois
        # laços rodando, e o vídeo andaria no dobro da velocidade.
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
        step = max(1, int(round(self.speed)))
        if not self.video.advance(step):
            self.playing = False
            self.message = "fim do vídeo"
        self.render()
        if self.playing:
            delay = int(1000 / (self.video.fps * min(self.speed, 1.0)))
            self.after_id = self.root.after(delay, self.tick)

    # --- teclado -----------------------------------------------------------
    def on_key(self, event) -> None:
        key = event.keysym
        shift = bool(event.state & 0x1)
        frame = self.video.position
        self.message = ""

        if key == "space":
            self.play(not self.playing)
        elif key in ("Right", "Left"):
            self.play(False)
            jump = int(round(2 * self.video.fps)) if shift else 1
            self.seek(frame + (jump if key == "Right" else -jump))
            return
        elif key in ("Up", "Down"):
            index = SPEEDS.index(self.speed) + (1 if key == "Up" else -1)
            self.speed = SPEEDS[int(np.clip(index, 0, len(SPEEDS) - 1))]
        elif key.lower() in KEYS:
            behavior = KEYS[key.lower()]
            was_open = behavior in self.annotation.open
            self.annotation.toggle(behavior, frame)
            self.message = f"{SHORT[behavior]} {'fechado' if was_open else 'aberto'} no quadro {frame}"
            self.save()
        elif key == "BackSpace":
            action = self.annotation.undo()
            self.message = f"desfeito: {action}" if action else "nada para desfazer"
            self.save()
        elif key == "Delete":
            removed = self.annotation.remove_at(frame)
            self.message = (f"apagado: {SHORT[removed[0]]} {removed[1]}–{removed[2]}"
                            if removed else "nenhum intervalo sob o cursor")
            self.save()
        elif key.lower() == "q":
            self.quit()
            return
        self.render()

    # --- desenho -----------------------------------------------------------
    def render(self) -> None:
        from PIL import Image, ImageTk

        position = self.video.position
        self.watched = max(self.watched, position)
        rgb = cv2.cvtColor(self.video.image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.screen.itemconfig(self.image_item, image=self.photo)

        seconds = position / self.video.fps
        total = self.video.count / self.video.fps
        opened = ", ".join(SHORT[b] for b in self.annotation.open) or "—"
        self.status.config(text=(
            f"{self.session_id}   {int(seconds // 60):02d}:{seconds % 60:04.1f} / "
            f"{int(total // 60):02d}:{total % 60:04.1f}   quadro {position}   "
            f"{self.speed:g}x {'>' if self.playing else '||'}   "
            f"assistido {self.watched / max(1, self.video.count - 1):.0%}\n"
            f"abertos: {opened}   {self.message}"))

        self.draw_timeline(self.full, 0, self.video.count)
        self.draw_timeline(self.zoom, *self.zoom_range())

    def draw_timeline(self, canvas, first: int, last: int) -> None:
        canvas.delete("all")
        span = max(1, last - first)
        width = self.timeline_width

        def x(frame):
            return (frame - first) / span * width

        # Faixa do que já foi assistido: fora dela, "sem marcação" não quer dizer nada.
        canvas.create_rectangle(0, 0, x(min(self.watched, last)), 4, fill="#95a5a6", width=0)
        for row, behavior in enumerate(KEYS.values()):
            top = 8 + 14 * row
            canvas.create_text(4, top + 6, text=SHORT[behavior], anchor="w",
                               font=("Consolas", 8), fill="#999999")
            for name, start, end in self.annotation.intervals:
                if name == behavior and end >= first and start <= last:
                    canvas.create_rectangle(x(start), top, max(x(end), x(start) + 2), top + 11,
                                            fill=COLORS[name], width=0)
            if behavior in self.annotation.open:
                start = self.annotation.open[behavior]
                low, high = sorted((start, self.video.position))
                canvas.create_rectangle(x(low), top, max(x(high), x(low) + 2), top + 11,
                                        fill=COLORS[behavior], stipple="gray50", width=0)
        cursor = x(self.video.position)
        canvas.create_line(cursor, 0, cursor, 14 * len(KEYS) + 16, fill="black", width=2)

    # --- persistência ------------------------------------------------------
    def save(self) -> None:
        replace_rows(self.output, self.annotation.rows(self.session_id, self.annotator),
                     self.session_id, self.annotator)
        progress = pd.DataFrame([{"session_id": self.session_id, "annotator": self.annotator,
                                  "watched_until": int(self.watched),
                                  "n_frames": int(self.video.count)}])
        replace_rows(self.progress, progress, self.session_id, self.annotator)

    def quit(self) -> None:
        self.play(False)
        closed = self.annotation.close_all(self.video.position)
        self.save()
        if closed:
            print(f"fechados no quadro {self.video.position}: {', '.join(closed)}")
        self.root.destroy()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "session_labels.csv")
    parser.add_argument("--progress", type=Path, default=ROOT / "data" / "session_progress.csv")
    parser.add_argument("--annotator", default="heittor")
    args = parser.parse_args()

    path = args.videos / f"{args.session}.mp4"
    if not path.exists():
        print(f"vídeo não encontrado: {path}")
        return 1

    intervals, watched = [], 0
    if args.output.exists():
        previous = pd.read_csv(args.output, encoding="utf-8-sig")
        mine = previous[(previous.session_id == args.session) & (previous.annotator == args.annotator)]
        intervals = list(zip(mine.behavior, mine.start_frame, mine.end_frame))
    if args.progress.exists():
        progress = pd.read_csv(args.progress, encoding="utf-8-sig")
        mine = progress[(progress.session_id == args.session) & (progress.annotator == args.annotator)]
        watched = int(mine.watched_until.max()) if len(mine) else 0
    if intervals or watched:
        print(f"continuando: {len(intervals)} intervalos, assistido até o quadro {watched}")

    import tkinter as tk
    root = tk.Tk()
    app = App(root, Video(path), Annotation(intervals), args.session, args.annotator,
              args.output, args.progress, watched)
    if watched:
        app.seek(watched)
    root.mainloop()

    saved = pd.read_csv(args.output, encoding="utf-8-sig") if args.output.exists() else pd.DataFrame()
    if len(saved):
        mine = saved[(saved.session_id == args.session) & (saved.annotator == args.annotator)]
        print(f"\n{len(mine)} intervalos salvos em {args.output}")
        if len(mine):
            seconds = (mine.end_frame - mine.start_frame + 1) / app.video.fps
            summary = mine.assign(seconds=seconds).groupby("behavior").seconds.agg(["size", "sum"])
            print(summary.rename(columns={"size": "vezes", "sum": "segundos"}).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
