"""Marca o chão de cada arena: os quatro cantos onde o chão encontra as paredes.

Até aqui a parede era estimada pelos percentis da posição do rato, um
retângulo alinhado aos eixos. A câmera é oblíqua, então o chão na imagem é um
quadrilátero qualquer, e num rato que quase não anda o "retângulo" vira o canto
onde ele ficou. As câmeras do DVR são fixas: entre sessões da mesma câmera, o
fundo se desloca no máximo 0,4 px. Então basta marcar uma vez por câmera.

A imagem é a mediana dos fundos de todas as sessões da câmera, na qual somem o
rato em movimento e também os que ficaram parados numa sessão só.

Uso:
    python scripts/mark_arena.py --videos ../videos_lab/arenas --output data/arena_floor.csv

Controles:
    clique = canto do chão (quatro, em qualquer ordem)   volta = desfaz o último
    r = refazer   n = próxima câmera (com os quatro cantos)   q = salvar e sair
    a lupa e a mão da barra de ferramentas dão zoom e deslocamento
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from mark_objects import background_plate  # noqa: E402
from mindtrace_ml.arena import arena_of, order_corners  # noqa: E402

CORNERS = 4


def mark_one(plate: np.ndarray, title: str):
    state = {"points": [], "action": None}
    figure, axes = plt.subplots(figsize=(13, 9))
    axes.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), interpolation="nearest")
    axes.set_axis_off()

    def refresh():
        for artist in list(axes.patches) + list(axes.lines):
            artist.remove()
        points = state["points"]
        if len(points) == CORNERS:
            axes.add_patch(Polygon(order_corners(points), closed=True, fill=False, lw=2, color="#00e676"))
        for x, y in points:
            axes.plot(x, y, "o", color="#ffb300", ms=6)
        missing = CORNERS - len(points)
        status = f"faltam {missing} cantos do chão" if missing else "completo — tecle n"
        axes.set_title(f"{title}\n{status}      volta=desfaz  r=refazer  n=próxima  q=sair", fontsize=11)
        figure.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not axes or event.xdata is None:
            return
        if figure.canvas.toolbar and figure.canvas.toolbar.mode:
            return   # zoom/pan ativo: o clique não é marcação
        if len(state["points"]) < CORNERS:
            state["points"].append((float(event.xdata), float(event.ydata)))
            refresh()

    def on_key(event):
        if event.key == "backspace" and state["points"]:
            state["points"].pop()
            refresh()
        elif event.key == "r":
            state["points"] = []
            refresh()
        elif event.key == "n" and len(state["points"]) == CORNERS:
            state["action"] = "next"
            plt.close(figure)
        elif event.key == "q":
            state["action"] = "quit"
            plt.close(figure)

    # Os atalhos do matplotlib colidem com as teclas da marcação.
    for name in ("keymap.back", "keymap.home", "keymap.quit", "keymap.save", "keymap.grid"):
        plt.rcParams[name] = []
    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()
    return state["points"], state["action"]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "arena_floor.csv")
    parser.add_argument("--redo", action="store_true", help="marca de novo as arenas já marcadas")
    args = parser.parse_args()

    by_arena = defaultdict(list)
    for video in sorted(args.videos.glob("*.mp4")):
        by_arena[arena_of(video.stem)].append(video)
    done = pd.read_csv(args.output, encoding="utf-8-sig") if args.output.exists() else pd.DataFrame(
        columns=["arena", "corner", "x", "y", "sessions"])

    for arena, videos in sorted(by_arena.items()):
        if arena in set(done.arena) and not args.redo:
            print(f"{arena}: já marcada, pulada")
            continue
        print(f"{arena}: mediana dos fundos de {len(videos)} sessões ...", flush=True)
        plate = np.median(np.stack([background_plate(v) for v in videos]), axis=0).astype(np.uint8)
        points, action = mark_one(plate, f"{arena} — {len(videos)} sessões")
        if len(points) == CORNERS:
            rows = pd.DataFrame([{"arena": arena, "corner": i + 1, "x": round(x, 1), "y": round(y, 1),
                                  "sessions": len(videos)} for i, (x, y) in enumerate(order_corners(points))])
            done = pd.concat([done[done.arena != arena], rows], ignore_index=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            done.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"{arena}: salva")
        if action != "next":
            break
    print(f"\narenas marcadas: {sorted(set(done.arena))} em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
