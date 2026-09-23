"""Reproduz os clipes sorteados e registra o rótulo humano, às cegas.

A janela não mostra o que a triagem achou do clipe. Isso é deliberado: ver o
palpite da máquina leva a confirmar em vez de julgar, e a medição de escape
viraria circular.

As classes são as quatro do plano da triagem: caminhando, parado, exploração e
outros. "Outros" é tudo o que não é rotina — grooming, rearing, sniffing fora do
objeto, o inesperado —, e não precisa ser discriminado: a triagem só precisa
saber o que pode ser pulado.

O critério é **presença, não dominância**. Use `c` ou `p` somente quando o
clipe for inteiramente aquilo; use `e` ou `o` quando aparecer, ainda que breve.

Um clipe de caminhada-rearing-caminhada leva `o`, mesmo com a caminhada ocupando
dois terços do tempo: a pergunta é "pular este trecho perderia algo?", e ali
perderia o rearing.

Quando `e` e `o` se aplicam juntos, `e` tem prioridade: exploração é a
variável de desfecho do NOR.

Uso:
    python scripts/label_clips.py --clips data/clips.csv --videos .../arenas \
                                  --output data/clip_labels.csv

Teclas:
    c = caminhando    p = parado    e = explorando objeto    o = outros
    x = não dá para ver (ocluso, animal fora)
    volta = desfazer o último    q = salvar e sair
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
from matplotlib.animation import FuncAnimation  # noqa: E402

# O matplotlib tem atalhos padrão em quase todas as teclas de rótulo — p arrasta,
# g liga a grade, r volta a visão, c e backspace navegam no histórico, o faz zoom.
# O teclado desta janela é só da rotulagem; a barra de ferramentas segue no mouse.
for _keymap in [k for k in plt.rcParams if k.startswith("keymap.")]:
    plt.rcParams[_keymap] = []

LABELS = {
    "c": "walking",
    "p": "still",
    "e": "object_interaction",
    "o": "other",
    "x": "unscorable",
}

LEGEND = ("c=caminhando   p=parado   e=explorando objeto   o=outros (qualquer coisa alem disso)\n"
          "x=nao da para ver   |   presenca, nao dominancia   |   volta=desfazer   q=sair")


def read_clip(video_path: Path, start: int, end: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    frames = []
    for _ in range(int(end - start + 1)):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def label_clip(frames, position: str):
    state = {"label": None, "action": None}

    figure, axes = plt.subplots(figsize=(11, 8))
    image = axes.imshow(frames[0])
    axes.set_axis_off()
    axes.set_title(f"{position}\n{LEGEND}", fontsize=10)

    def update(index):
        image.set_data(frames[index % len(frames)])
        return (image,)

    animation = FuncAnimation(figure, update, frames=len(frames) * 100,
                              interval=33, blit=True, repeat=False,
                              cache_frame_data=False)

    def on_key(event):
        if event.key in LABELS:
            state["label"] = LABELS[event.key]
            plt.close(figure)
        elif event.key == "backspace":
            state["action"] = "undo"
            plt.close(figure)
        elif event.key == "q":
            state["action"] = "quit"
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if animation.event_source is not None:
        animation.event_source.stop()

    return state["label"], state["action"]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--annotator", default="heittor")
    args = parser.parse_args()

    clips = pd.read_csv(args.clips, encoding="utf-8-sig")
    done = {}
    if args.output.exists():
        previous = pd.read_csv(args.output, encoding="utf-8-sig")
        done = dict(zip(previous.clip_id, previous.label))
        print(f"{len(done)} clipes já rotulados — serão pulados\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [{"clip_id": k, "label": v, "annotator": args.annotator} for k, v in done.items()]
    index = 0

    while index < len(clips):
        clip = clips.iloc[index]
        if clip.clip_id in done:
            index += 1
            continue

        video = args.videos / f"{clip.session_id}.mp4"
        if not video.exists():
            print(f"  vídeo ausente: {video.name}")
            index += 1
            continue

        frames = read_clip(video, clip.start_frame, clip.end_frame)
        if not frames:
            print(f"  não consegui ler {clip.clip_id}")
            index += 1
            continue

        label, action = label_clip(frames, f"clipe {index + 1} de {len(clips)}")

        if action == "quit":
            break
        if action == "undo":
            if rows:
                removed = rows.pop()
                done.pop(removed["clip_id"], None)
                index = max(0, index - 1)
                print(f"  desfeito: {removed['clip_id']}")
            continue
        if label is None:
            continue

        rows.append({"clip_id": clip.clip_id, "label": label, "annotator": args.annotator})
        done[clip.clip_id] = label
        pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"  [{len(rows)}/{len(clips)}] {clip.clip_id} -> {label}")
        index += 1

    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n{len(frame)} clipes rotulados em {args.output}")
        print(frame.label.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
