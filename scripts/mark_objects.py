"""Marca a posição e o tamanho dos objetos de cada arena.

Os objetos não se movem durante a sessão, então são marcados uma vez por vídeo.
Guardar **centro e raio** em vez de só um ponto importa porque o critério de
exploração do NOR é a distância do focinho à *borda* do objeto: com objetos de
tamanhos diferentes, um ponto único produziria zonas de exploração desiguais sem
que isso ficasse visível em lugar nenhum.

O quadro exibido é a **mediana** de várias amostras do vídeo, o que remove o
animal em movimento e deixa só o fundo estático — os objetos aparecem limpos,
sem nada por cima.

Uso:
    python scripts/mark_objects.py --videos .../arenas --output data/objects.csv

Controles:
    clique 1 = centro do objeto, clique 2 = borda (define o raio)
    n = próximo vídeo      r = refazer este vídeo
    c = copiar do vídeo anterior
    q = salvar e sair
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ZOOM = 3
SAMPLES = 25
OBJECTS_PER_ARENA = 2


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


def draw(plate, marks, pending):
    canvas = cv2.resize(plate, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    for index, (cx, cy, radius) in enumerate(marks):
        color = (0, 220, 255) if index == 0 else (255, 160, 0)
        cv2.circle(canvas, (int(cx * ZOOM), int(cy * ZOOM)), int(radius * ZOOM), color, 2)
        cv2.circle(canvas, (int(cx * ZOOM), int(cy * ZOOM)), 3, color, -1)
    if pending is not None:
        cv2.circle(canvas, (int(pending[0] * ZOOM), int(pending[1] * ZOOM)), 3, (0, 255, 0), -1)
    return canvas


def mark_one(video_path: Path, previous):
    plate = background_plate(video_path)
    state = {"marks": [], "pending": None, "action": None}
    window = video_path.stem

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(state["marks"]) >= OBJECTS_PER_ARENA:
            return
        point = (x / ZOOM, y / ZOOM)
        if state["pending"] is None:
            state["pending"] = point
        else:
            cx, cy = state["pending"]
            radius = float(np.hypot(point[0] - cx, point[1] - cy))
            state["marks"].append((cx, cy, radius))
            state["pending"] = None

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = draw(plate, state["marks"], state["pending"])
        faltam = OBJECTS_PER_ARENA - len(state["marks"])
        legenda = f"{window}  |  faltam {faltam}  |  n=proximo r=refazer c=copiar q=sair"
        cv2.putText(canvas, legenda, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            state["marks"], state["pending"] = [], None
        elif key == ord("c") and previous:
            state["marks"] = list(previous)
            state["pending"] = None
        elif key == ord("n") and len(state["marks"]) == OBJECTS_PER_ARENA:
            state["action"] = "next"
            break
        elif key == ord("q"):
            state["action"] = "quit"
            break

    cv2.destroyWindow(window)
    return state["marks"], state["action"]


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
    rows = []

    for index, video in enumerate(videos, start=1):
        if video.stem in done:
            previous = done[video.stem]
            continue

        print(f"[{index}/{len(videos)}] {video.name}")
        marks, action = mark_one(video, previous)

        if marks:
            for object_id, (cx, cy, radius) in enumerate(marks, start=1):
                rows.append({
                    "session_id": video.stem,
                    "object_id": object_id,
                    "center_x": round(cx, 2),
                    "center_y": round(cy, 2),
                    "radius": round(radius, 2),
                })
            previous = marks
            # Grava a cada vídeo: interromper no meio não perde o que já foi feito.
            combined = pd.DataFrame(rows)
            if args.output.exists():
                combined = pd.concat([pd.read_csv(args.output, encoding="utf-8-sig"), combined])
                combined = combined.drop_duplicates(["session_id", "object_id"], keep="last")
            combined.to_csv(args.output, index=False, encoding="utf-8-sig")
            rows = []

        if action == "quit":
            print("\ninterrompido — o que foi marcado está salvo")
            break

    cv2.destroyAllWindows()
    if args.output.exists():
        final = pd.read_csv(args.output, encoding="utf-8-sig")
        print(f"\n{final['session_id'].nunique()} vídeos marcados em {args.output}")
        print(f"raio mediano: {final['radius'].median():.1f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
