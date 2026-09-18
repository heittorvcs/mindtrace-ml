"""Marca a posição e o tamanho dos objetos de cada arena.

Os objetos não se movem durante a sessão, então são marcados uma vez por vídeo.
Guardar **centro e raio** em vez de só um ponto importa porque o critério de
exploração do NOR é a distância do focinho à *borda* do objeto: com objetos de
diâmetros diferentes, um ponto único produziria zonas de exploração desiguais sem
que isso ficasse visível em lugar nenhum.

O quadro exibido é a **mediana** de várias amostras do vídeo, o que remove o
animal em movimento e deixa só o cenário estático — os objetos aparecem limpos,
sem precisar caçar um quadro em que o rato não esteja em cima deles.

A interface usa matplotlib porque o OpenCV instalado é a variante headless, sem
suporte a janela.

Uso:
    python scripts/mark_objects.py --videos .../arenas --output data/objects.csv

Controles:
    clique 1 = centro do objeto, clique 2 = borda (define o raio)
    n = próximo vídeo      r = refazer este vídeo
    c = copiar do anterior q = salvar e sair
    a lupa e a mão da barra de ferramentas dão zoom e deslocamento
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

SAMPLES = 25
OBJECTS_PER_ARENA = 2
COLORS = ["#ffb300", "#00b0ff"]


def background_plate(video_path: Path) -> np.ndarray:
    """Mediana de amostras ao longo do vídeo: o animal some, o cenário fica."""
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in np.linspace(total * 0.1, total * 0.9, SAMPLES).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"não consegui ler quadros de {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def mark_one(plate: np.ndarray, title: str, previous):
    state = {"marks": [], "pending": None, "action": None}

    figure, axes = plt.subplots(figsize=(13, 9))
    axes.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), interpolation="nearest")
    axes.set_axis_off()

    def refresh():
        for patch in list(axes.patches):
            patch.remove()
        for index, (cx, cy, radius) in enumerate(state["marks"]):
            axes.add_patch(Circle((cx, cy), radius, fill=False, lw=2, color=COLORS[index]))
            axes.add_patch(Circle((cx, cy), 1.5, color=COLORS[index]))
        if state["pending"] is not None:
            axes.add_patch(Circle(state["pending"], 1.5, color="#00e676"))

        faltam = OBJECTS_PER_ARENA - len(state["marks"])
        proximo = "clique a BORDA" if state["pending"] is not None else "clique o CENTRO"
        estado = f"faltam {faltam} — {proximo}" if faltam else "completo — tecle n"
        axes.set_title(f"{title}\n{estado}      n=proximo  r=refazer  c=copiar  q=sair",
                       fontsize=11)
        figure.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not axes or event.xdata is None:
            return
        if figure.canvas.toolbar and figure.canvas.toolbar.mode:
            return  # zoom/pan ativo: não registra clique como marcação
        if len(state["marks"]) >= OBJECTS_PER_ARENA:
            return

        point = (float(event.xdata), float(event.ydata))
        if state["pending"] is None:
            state["pending"] = point
        else:
            cx, cy = state["pending"]
            radius = float(np.hypot(point[0] - cx, point[1] - cy))
            state["marks"].append((cx, cy, radius))
            state["pending"] = None
        refresh()

    def on_key(event):
        if event.key == "r":
            state["marks"], state["pending"] = [], None
            refresh()
        elif event.key == "c" and previous:
            state["marks"] = list(previous)
            state["pending"] = None
            refresh()
        elif event.key == "n" and len(state["marks"]) == OBJECTS_PER_ARENA:
            state["action"] = "next"
            plt.close(figure)
        elif event.key == "q":
            state["action"] = "quit"
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()

    return state["marks"], state["action"]


def save(output: Path, rows):
    frame = pd.DataFrame(rows)
    if output.exists():
        frame = pd.concat([pd.read_csv(output, encoding="utf-8-sig"), frame])
        frame = frame.drop_duplicates(["session_id", "object_id"], keep="last")
    frame.to_csv(output, index=False, encoding="utf-8-sig")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default="*.mp4")
    args = parser.parse_args()

    videos = sorted(args.videos.glob(args.pattern))
    if not videos:
        print(f"nenhum vídeo em {args.videos}")
        return 1

    done = {}
    if args.output.exists():
        existing = pd.read_csv(args.output, encoding="utf-8-sig")
        for session, group in existing.groupby("session_id"):
            done[session] = [(r.center_x, r.center_y, r.radius) for r in group.itertuples()]
        print(f"{len(done)} vídeos já marcados — serão pulados\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = None

    for index, video in enumerate(videos, start=1):
        if video.stem in done:
            previous = done[video.stem]
            continue

        print(f"[{index}/{len(videos)}] {video.name}")
        marks, action = mark_one(background_plate(video), video.stem, previous)

        if marks:
            save(args.output, [
                {"session_id": video.stem, "object_id": object_id,
                 "center_x": round(cx, 2), "center_y": round(cy, 2), "radius": round(radius, 2)}
                for object_id, (cx, cy, radius) in enumerate(marks, start=1)
            ])
            previous = marks

        if action == "quit":
            print("\ninterrompido — o que foi marcado está salvo")
            break

    if args.output.exists():
        final = pd.read_csv(args.output, encoding="utf-8-sig")
        print(f"\n{final['session_id'].nunique()} vídeos marcados em {args.output}")
        print(f"raio mediano: {final['radius'].median():.1f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
