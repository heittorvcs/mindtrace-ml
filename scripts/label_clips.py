"""Reproduz os clipes sorteados e registra os rótulos humanos, às cegas.

A janela não mostra o que a triagem achou do clipe. Isso é deliberado: ver o
palpite da máquina leva a confirmar em vez de julgar, e a medição de escape
viraria circular.

**Rotule fino, a triagem modela grosso.** As teclas distinguem grooming, rearing
e sniffing, embora a triagem só precise saber o que é rotina e o que é "outros":
fundir rótulos finos em "outros" é uma linha de código, e separar rótulos grossos
depois exigiria reanotar. Os rótulos finos são os dados de treino de uma segunda
etapa, que classifica o que a triagem mandou para revisão.

**Seleção múltipla.** Marque todas as teclas que se aplicam ao clipe e confirme
com Enter; apertar de novo desmarca. Um clipe de caminhada com um rearing no
meio leva `c` e `r`.

O critério continua sendo **presença, não dominância**: marque o que aparecer,
ainda que breve.

Uso:
    python scripts/label_clips.py --clips data/clips_teste.csv --videos .../arenas \\
                                  --output data/clip_labels_teste.csv

Teclas:
    c = caminhando   p = parado   e = explorando objeto
    g = grooming     r = rearing  s = sniffing fora do objeto   o = outros
    x = não dá para ver (exclui as demais)
    Enter = confirmar   volta = desfazer o clipe anterior   q = salvar e sair
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
# g liga a grade, r volta a visão, s salva, o faz zoom, c e backspace navegam no
# histórico. O teclado desta janela é só da rotulagem.
for _keymap in [k for k in plt.rcParams if k.startswith("keymap.")]:
    plt.rcParams[_keymap] = []

LABELS = {
    "c": "walking",
    "p": "still",
    "e": "object_interaction",
    "g": "grooming",
    "r": "rearing",
    "s": "sniffing",
    "o": "other",
    "x": "unscorable",
}

SHORT = {"c": "caminhando", "p": "parado", "e": "explorando", "g": "grooming",
         "r": "rearing", "s": "sniffing", "o": "outros", "x": "nao da para ver"}

# Ordem de prioridade para a coluna `label`, que guarda um rótulo único por
# compatibilidade com as rodadas anteriores. `labels` guarda todos.
PRIORITY = ("object_interaction", "rearing", "grooming", "other", "sniffing",
            "walking", "still", "unscorable")

LEGEND = ("c=caminhando  p=parado  e=explorando  g=grooming  r=rearing  s=sniffing  o=outros\n"
          "x=nao da para ver  |  marque TODAS que aparecem  |  Enter=confirmar  volta=desfazer  q=sair")


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
    state = {"selected": [], "action": None}

    figure, axes = plt.subplots(figsize=(11, 8))
    image = axes.imshow(frames[0])
    axes.set_axis_off()

    def refresh_title():
        chosen = " + ".join(SHORT[k] for k in state["selected"]) or "nada marcado"
        axes.set_title(f"{position}   |   marcado: {chosen}\n{LEGEND}", fontsize=10)

    def update(index):
        image.set_data(frames[index % len(frames)])
        return (image,)

    refresh_title()
    # blit=False: com blit o título não é redesenhado, e a seleção ficaria invisível.
    animation = FuncAnimation(figure, update, frames=len(frames) * 200,
                              interval=33, blit=False, repeat=False,
                              cache_frame_data=False)

    def on_key(event):
        key = event.key
        if key in LABELS:
            if key == "x":
                state["selected"] = [] if state["selected"] == ["x"] else ["x"]
            else:
                if "x" in state["selected"]:
                    state["selected"].remove("x")
                if key in state["selected"]:
                    state["selected"].remove(key)
                else:
                    state["selected"].append(key)
            refresh_title()
            figure.canvas.draw_idle()
        elif key == "enter" and state["selected"]:
            state["action"] = "confirm"
            plt.close(figure)
        elif key == "backspace":
            state["action"] = "undo"
            plt.close(figure)
        elif key == "q":
            state["action"] = "quit"
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if animation.event_source is not None:
        animation.event_source.stop()

    names = [LABELS[k] for k in state["selected"]]
    return names, state["action"]


def primary(names) -> str:
    return next(n for n in PRIORITY if n in names)


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
    rows = []
    if args.output.exists():
        previous = pd.read_csv(args.output, encoding="utf-8-sig").fillna("")
        rows = previous.to_dict("records")
        print(f"{len(rows)} clipes já rotulados — serão pulados\n")
    done = {row["clip_id"] for row in rows}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    index = 0

    while index < len(clips):
        clip = clips.iloc[index]
        if clip.clip_id in done:
            index += 1
            continue

        video = args.videos / f"{clip.session_id}.mp4"
        frames = read_clip(video, clip.start_frame, clip.end_frame) if video.exists() else []
        if not frames:
            print(f"  não consegui ler {clip.clip_id}")
            index += 1
            continue

        names, action = label_clip(frames, f"clipe {index + 1} de {len(clips)}")

        if action == "quit":
            break
        if action == "undo":
            if rows:
                removed = rows.pop()
                done.discard(removed["clip_id"])
                index = max(0, index - 1)
                pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
                print(f"  desfeito: {removed['clip_id']}")
            continue
        if action != "confirm":
            continue

        rows.append({"clip_id": clip.clip_id, "label": primary(names),
                     "labels": ";".join(names), "annotator": args.annotator})
        done.add(clip.clip_id)
        pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"  [{len(rows)}/{len(clips)}] {clip.clip_id} -> {' + '.join(names)}")
        index += 1

    if rows:
        frame = pd.DataFrame(rows)
        print(f"\n{len(frame)} clipes rotulados em {args.output}")
        print(frame["labels"].str.split(";").explode().value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
