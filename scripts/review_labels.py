"""Revisão de duas marcações da mesma sessão, assistindo ao vídeo.

Mostra, quadro a quadro, onde duas marcações de um comportamento concordam e onde
divergem: a antiga e a re-rotulagem às cegas, ou dois observadores. A faixa sob o
vídeo muda de cor conforme o quadro atual, e `n`/`p` pulam de uma discordância
para outra. Só leitura: nada é gravado.

Os episódios seguem a mesma definição da análise: descidas breves fundidas
(MERGE_GAP_SEC), e os trechos "não dá para ver" de qualquer das duas ficam fora
da comparação. Um episódio está "nas duas" se cruza algum episódio da outra.

Uso:
    python scripts/review_labels.py --session TT_9_cam2 --videos ../videos_lab/arenas
    python scripts/review_labels.py --session TT_15_cam1 --videos ../videos_lab/arenas \\
        --a data/session_labels.csv --b data/relabel/session_labels_r2.csv --names antiga,nova

Teclas:
    espaço = tocar/pausar   ← → = um quadro   shift+← → = 2 s   ↑ ↓ = velocidade
    n / p = próxima / anterior discordância (começa 1 s antes e toca)
    clique na linha do tempo = pula para lá   q = sair
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

from label_session import SCALE, SPEEDS, ZOOM_SEC, Video  # noqa: E402
from mindtrace_ml.behavior_model import bouts, label_mask  # noqa: E402

# As mesmas cores do gráfico de comparação: azul nas duas, laranja só na
# primeira, verde-água só na segunda.
STATUS_COLORS = {"both": "#2a78d6", "only_a": "#eb6834", "only_b": "#1baf7a"}
UNSCORABLE_COLOR = "#7f8c8d"
IDLE_COLOR = "#e1e0d9"
MARGIN = 54   # espaço do nome de cada linha, à esquerda das linhas do tempo


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


class App:
    def __init__(self, root, video: Video, episodes, masks, session_id: str, behavior: str, names):
        import tkinter as tk

        self.root, self.video = root, video
        self.episodes, self.masks = episodes, masks
        self.todo = disagreements(episodes)
        self.session_id, self.behavior, self.names = session_id, behavior, names
        self.playing = False
        self.after_id = None
        self.speed = 1.0

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
        tk.Label(root, font=("Consolas", 10), justify="left", anchor="w", text=(
            f"azul = nas duas   laranja = só na {names[0]}   verde-água = só na {names[1]}   "
            "cinza = não dá para ver\n"
            "espaco=tocar/pausar  <- ->=1 quadro  shift+<- ->=2 s  cima/baixo=velocidade  "
            "n/p=próxima/anterior discordância  q=sair")).pack(fill="x", padx=8, pady=6)

        self.image_item = self.screen.create_image(0, 0, anchor="nw")
        self.photo = None
        self.full.bind("<Button-1>", lambda e: self.seek_click(e, 0, self.video.count))
        self.zoom.bind("<Button-1>", lambda e: self.seek_click(e, *self.zoom_range()))
        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.title(f"Revisão de marcações — {session_id} ({behavior})")

        self.video.read(0)
        self.render()

    # --- navegação ---------------------------------------------------------
    def seek(self, frame: int) -> None:
        self.video.read(frame)
        self.render()

    def seek_click(self, event, first: int, last: int) -> None:
        share = (event.x - MARGIN) / max(1, self.width - MARGIN)
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
            self.play(False)
            step = int(round(2 * self.video.fps)) if shift else 1
            self.seek(self.video.position + (step if key == "Right" else -step))
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
            return MARGIN + (frame - first) / span * (self.width - MARGIN)

        for row, (lane, name) in enumerate((("a", self.names[0]), ("b", self.names[1]))):
            top = 8 + 16 * row
            canvas.create_text(4, top + 6, text=name, anchor="w", font=("Consolas", 9), fill="#777777")
            canvas.create_rectangle(MARGIN, top, self.width, top + 12, fill="#ececea", width=0)
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
    parser.add_argument("--names", default="antiga,nova", help="nomes das duas marcações, separados por vírgula")
    parser.add_argument("--behavior", default="rearing")
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
    if any(table.empty for table in labels):
        empty = [str(s) for s, t in zip((args.a, args.b), labels) if t.empty]
        print(f"{args.session} sem marcações em: {', '.join(empty)}")
        return 1

    video = Video(path)
    episodes, masks = compare_episodes(*labels, args.behavior, video.count, video.fps)
    for lane, name in zip("ab", names):
        mine = [e for e in episodes if e["lane"] == lane]
        alone = sum(e["status"] != "both" for e in mine)
        print(f"{name}: {len(mine)} episódios de {args.behavior}, {alone} só nela")

    import tkinter as tk
    root = tk.Tk()
    App(root, video, episodes, masks, args.session, args.behavior, names)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
