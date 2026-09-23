"""Mede o modelo congelado numa rodada sorteada por faixa de nota.

A rodada gerada por `sample_clips.py --model` guarda no gabarito a faixa de cada
clipe e a parcela do vídeo que a faixa ocupa. Isso basta para medir o modelo sem
recarregá-lo: pulando 70% do vídeo, tudo o que está nas faixas abaixo de 70 fica
de fora e o resto chega ao pesquisador. Cada ponto de operação sai da mesma
rotulagem.

As regras são medidas nos mesmos clipes, com os mesmos pesos: a estimativa
ponderada vale para qualquer classificador, desde que se conheça como a amostra
foi sorteada.

O intervalo vem de bootstrap dentro de cada faixa — reamostra os clipes como o
sorteio os escolheu — e mostra quanto da diferença é ruído de amostra pequena.

Uso:
    python scripts/evaluate_model_round.py --labels data/clip_labels_modelo.csv \\
        --key data/clips_modelo_gabarito.csv --pose data/pose --objects data/objects.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.detectors import Thresholds, triage  # noqa: E402
from mindtrace_ml.kinematics import frame_kinematics  # noqa: E402
from mindtrace_ml.triage_model import KEYPOINTS, label_sets, notable_set  # noqa: E402


def band_edges(name: str) -> tuple[float, float]:
    _, low, high = name.split("_")
    return int(low) / 100, int(high) / 100


def weighted_escape(clips: pd.DataFrame, surfaced: np.ndarray, target: np.ndarray) -> float:
    """Fração do alvo que fica na parte pulada, com o peso de cada faixa no vídeo."""
    total = escaped = 0.0
    for band, index in clips.groupby("triage_label").indices.items():
        weight = clips.band_share.iloc[index[0]]
        total += weight * target[index].mean()
        escaped += weight * (target[index] & ~surfaced[index]).mean()
    return escaped / total if total else float("nan")


def bootstrap(clips, surfaced, target, rng, reps: int = 2000) -> tuple[float, float]:
    groups = list(clips.groupby("triage_label").indices.values())
    values = []
    for _ in range(reps):
        index = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        values.append(weighted_escape(clips.iloc[index].reset_index(drop=True),
                                      surfaced[index], target[index]))
    return tuple(np.nanpercentile(values, [5, 95]))


def rules_surfaced(clips, pose_dir, objects, fps, min_presence) -> np.ndarray:
    out = np.zeros(len(clips), dtype=bool)
    for session, group in clips.groupby("session_id"):
        pose = pd.read_csv(pose_dir / f"{session}.csv", encoding="utf-8-sig")
        triaged = triage(pose, frame_kinematics(pose, KEYPOINTS, fps),
                         objects[objects.session_id == session], fps, Thresholds())
        for position, clip in zip(group.index, group.itertuples()):
            window = triaged[(triaged.frame >= clip.start_frame) & (triaged.frame <= clip.end_frame)]
            out[position] = (window.review.mean() >= min_presence
                             or window.unscorable.mean() >= min_presence)
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--min-presence", type=float, default=0.10)
    parser.add_argument("--sniffing-is-notable", action="store_true")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    labels["names"] = label_sets(labels)
    key = pd.read_csv(args.key, encoding="utf-8-sig")
    clips = key.merge(labels[["clip_id", "names"]], on="clip_id")
    clips = clips[clips.names.map(lambda n: n != {"unscorable"})].reset_index(drop=True)

    notable = notable_set(args.sniffing_is_notable)
    is_notable = clips.names.map(lambda n: bool(n & notable)).to_numpy()
    objects = pd.read_csv(args.objects, encoding="utf-8-sig")
    rules = rules_surfaced(clips, args.pose, objects, args.fps, args.min_presence)
    lower = clips.triage_label.map(lambda b: band_edges(b)[0]).to_numpy()
    rng = np.random.default_rng(0)

    print(f"{len(clips)} clipes, {int(is_notable.sum())} com algo notável "
          f"(sniffing conta como {'outros' if args.sniffing_is_notable else 'rotina'})\n")
    print(f"{'faixa':<14} {'% do vídeo':>10} {'notáveis':>10}")
    for band, group in clips.groupby("triage_label"):
        index = group.index.to_numpy()
        print(f"{band:<14} {group.band_share.iloc[0]:>10.0%} "
              f"{int(is_notable[index].sum()):>6}/{len(index)}")

    kinds = sorted(k for k in notable if clips.names.map(lambda n, k=k: k in n).any())
    print(f"\n{'vídeo pulado':<14} {'escapam (IC 90%)':>22}  " + "  ".join(f"{k:>9}" for k in kinds))
    print("-" * (40 + 11 * len(kinds)))
    for cut in sorted({band_edges(b)[1] for b in clips.triage_label} - {1.0}):
        surfaced = lower >= cut
        escape = weighted_escape(clips, surfaced, is_notable)
        low, high = bootstrap(clips, surfaced, is_notable, rng)
        seen = []
        for kind in kinds:
            has = clips.names.map(lambda n, k=kind: k in n).to_numpy()
            seen.append(1 - weighted_escape(clips, surfaced, has) if has.any() else float("nan"))
        print(f"modelo {cut:>5.0%}   {escape:>8.0%} ({low:.0%}–{high:.0%})    "
              + "  ".join(f"{s:>9.0%}" for s in seen))

    escape = weighted_escape(clips, rules, is_notable)
    low, high = bootstrap(clips, rules, is_notable, rng)
    seen = []
    for kind in kinds:
        has = clips.names.map(lambda n, k=kind: k in n).to_numpy()
        seen.append(1 - weighted_escape(clips, rules, has) if has.any() else float("nan"))
    print(f"regras   79%   {escape:>8.0%} ({low:.0%}–{high:.0%})    "
          + "  ".join(f"{s:>9.0%}" for s in seen))
    print("\ncolunas por tipo: fração vista pelo pesquisador")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
