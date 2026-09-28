"""Compara o rearing detectado com as contagens de referência do laboratório.

As contagens da planilha foram feitas sem o nosso sistema, em animais e numa
arena que o modelo nunca viu — é o teste mais independente disponível. Mas é
uma comparação de totais: um rearing perdido e um alarme falso se anulam, e
quem contou pode ter usado outro critério. Serve para dizer se o modelo
acompanha os animais (quem levanta mais, quem levanta menos), não onde ele erra.

O relógio começa quando o rato aparece na arena, e não no primeiro quadro: o
vídeo tem alguns segundos a mais que os 20 min da sessão, gastos em colocar o
animal. A distância percorrida entra também, para ver se pixels e metros andam
juntos — e, se andarem, dar a escala.

Uso:
    python scripts/compare_reference_counts.py --reference data/reference/habituacao_femeas_2025.csv \\
        --model models/rearing_model.joblib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.behavior_model import bouts, frame_mask, gate, session_features  # noqa: E402
from mindtrace_ml.kinematics import frame_kinematics  # noqa: E402
from mindtrace_ml.movement import movement_layer  # noqa: E402
from mindtrace_ml.triage_model import KEYPOINTS  # noqa: E402

THRESHOLDS = (0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
WINDOWS_MIN = (5, 20)


def arrival(pose: pd.DataFrame, fps: float, min_points: int = 3, hold_sec: float = 1.0) -> int:
    """Primeiro quadro em que o rato está na arena, visto por 1 s seguido."""
    seen = (pose[[f"{k}_p" for k in KEYPOINTS]].to_numpy() >= 0.25).sum(axis=1) >= min_points
    hold = int(round(hold_sec * fps))
    run = 0
    for index, value in enumerate(seen):
        run = run + 1 if value else 0
        if run >= hold:
            return index - hold + 1
    return 0


def merged(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Episódios, fundindo os separados por menos de `gap` quadros.

    Uma queda breve da nota no meio de um rearing o partiria em dois, e contar
    episódios — ao contrário de somar tempo — é sensível a isso.
    """
    found = bouts(mask)
    if not found:
        return []
    out = [list(found[0])]
    for start, end in found[1:]:
        if start - out[-1][1] <= gap:
            out[-1][1] = end
        else:
            out.append([start, end])
    return [tuple(b) for b in out]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--behavior", default="rearing")
    parser.add_argument("--pose", type=Path, default=ROOT / "data" / "pose")
    parser.add_argument("--video-features", type=Path, default=ROOT / "data" / "video_features")
    parser.add_argument("--objects", type=Path, default=ROOT / "data" / "objects.csv")
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--merge-gap-sec", type=float, default=0.3)
    args = parser.parse_args()

    import joblib
    bundle = joblib.load(args.model)
    reference = pd.read_csv(args.reference, encoding="utf-8-sig")
    objects = pd.read_csv(args.objects, encoding="utf-8-sig")
    gap = int(round(args.merge_gap_sec * args.fps))

    rows = []
    available = [s for s in reference.session_id
                 if (args.pose / f"{s}.csv").exists() and (args.video_features / f"{s}.csv").exists()]
    print(f"{len(available)} de {len(reference)} sessões da referência com pose e sinais de vídeo\n")
    for session in available:
        pose = pd.read_csv(args.pose / f"{session}.csv", encoding="utf-8-sig")
        pixels = pd.read_csv(args.video_features / f"{session}.csv", encoding="utf-8-sig")
        session_objects = objects[objects.session_id == session]
        data = session_features(pose, session_objects, args.fps, pixels)
        scores = bundle["models"][args.behavior].predict_proba(data["X"][bundle["features"]])[:, 1]
        start = arrival(pose, args.fps)
        layer = movement_layer(pose, frame_kinematics(pose, KEYPOINTS, args.fps), session_objects, args.fps)

        row = {"session_id": session, "start_s": start / args.fps}
        for minutes in WINDOWS_MIN:
            end = start + int(round(minutes * 60 * args.fps))
            row[f"distance_{minutes}min_px"] = float(layer.distance.iloc[min(end, len(layer)) - 1]
                                                     - layer.distance.iloc[start])
        for threshold in THRESHOLDS:
            mask = gate(args.behavior, frame_mask(scores, data["centers"], len(data["frames"]),
                                                  threshold, args.fps), data["freezing"], args.fps)
            episodes = merged(mask, gap)
            for minutes in WINDOWS_MIN:
                end = start + int(round(minutes * 60 * args.fps))
                row[f"count_{minutes}min_t{threshold}"] = sum(start <= a < end for a, _ in episodes)
        rows.append(row)
        print(f"  {session}: rato aparece em {row['start_s']:.1f} s", flush=True)

    detected = pd.DataFrame(rows).merge(reference, on="session_id")
    if detected.empty:
        return 0

    for minutes in WINDOWS_MIN:
        truth = detected[f"rearings_{minutes}min"]
        print(f"\nrearings em {minutes} min — referência: média {truth.mean():.0f} por sessão")
        print(f"{'limiar':>7} {'média detectada':>16} {'razão':>7} {'erro médio':>11} {'correlação':>11}")
        for threshold in THRESHOLDS:
            counts = detected[f"count_{minutes}min_t{threshold}"]
            r = np.corrcoef(truth, counts)[0, 1] if len(detected) > 2 and counts.std() > 0 else np.nan
            print(f"{threshold:>7.2f} {counts.mean():>16.0f} {counts.sum() / truth.sum():>7.2f} "
                  f"{(counts - truth).abs().mean():>11.1f} {r:>11.2f}")

    print("\ndistância: pixels × metros da referência")
    for minutes in WINDOWS_MIN:
        px, meters = detected[f"distance_{minutes}min_px"], detected[f"distance_{minutes}min_m"]
        scale = px / meters
        r = np.corrcoef(px, meters)[0, 1] if len(detected) > 2 else np.nan
        print(f"  {minutes} min: {scale.median():.0f} px por metro (varia {scale.min():.0f}–{scale.max():.0f}), "
              f"correlação {r:.2f}")

    view = ["session_id", "rearings_5min", "count_5min_t0.3", "rearings_20min", "count_20min_t0.3",
            "distance_20min_m", "distance_20min_px"]
    print("\npor sessão (limiar 0,3):")
    print(detected[view].round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
