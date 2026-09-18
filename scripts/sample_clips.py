"""Sorteia clipes estratificados pelas faixas da triagem, para rotulagem às cegas.

A amostra precisa cobrir **todas** as faixas, não só a de revisão: taxa de escape
é comportamento interessante que caiu numa faixa de rotina, e rotular apenas o
que já foi para revisão o tornaria invisível por construção.

A saída não carrega o rótulo da triagem — ele fica num arquivo à parte, para que
a rotulagem seja cega. Ver o palpite da máquina leva o anotador a confirmar em
vez de julgar, e a medição vira circular.

Uso:
    python scripts/sample_clips.py --pose data/pose --objects data/objects.csv \
                                   --videos .../arenas --output data/clips.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.detectors import BEHAVIOR_ORDER, Thresholds, triage  # noqa: E402
from mindtrace_ml.kinematics import frame_kinematics  # noqa: E402

KEYPOINTS = ("nose", "ear_left", "ear_right", "neck", "body", "tail_base")
STRATA = (*BEHAVIOR_ORDER, "review")


def sample_from(mask: np.ndarray, frames: np.ndarray, clip_frames: int,
                count: int, rng) -> list[int]:
    """Início de clipes inteiramente contidos num trecho da faixa.

    Exigir que o clipe todo caia dentro da faixa evita amostrar transições, que
    seriam ambíguas para o anotador e não dizem nada sobre a faixa em si.
    """
    valid = []
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= clip_frames:
            valid.append(index - clip_frames + 1)

    if not valid:
        return []
    picks = rng.choice(len(valid), size=min(count, len(valid)), replace=False)
    return [int(frames[valid[p]]) for p in sorted(picks)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip-sec", type=float, default=3.0)
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--walking-speed", type=float, default=15.0)
    parser.add_argument("--freezing-speed", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    objects = pd.read_csv(args.objects, encoding="utf-8-sig")
    thresholds = Thresholds(walking_speed=args.walking_speed,
                            freezing_speed=args.freezing_speed)
    clip_frames = int(round(args.clip_sec * args.fps))
    rng = np.random.default_rng(args.seed)

    sessions = sorted(args.pose.glob("*.csv"))
    if not sessions:
        print(f"nenhum CSV de pose em {args.pose}")
        return 1

    # Junta todos os candidatos por faixa e só depois sorteia, para que o número
    # de clipes por faixa não dependa de quantas sessões cada uma domina.
    candidates = {stratum: [] for stratum in STRATA}

    for path in sessions:
        session = path.stem
        pose = pd.read_csv(path, encoding="utf-8-sig")
        session_objects = objects[objects.session_id == session]
        if session_objects.empty:
            print(f"  {session}: sem objetos marcados, pulado")
            continue

        kinematics = frame_kinematics(pose, KEYPOINTS, args.fps, min_confidence=0.25)
        triaged = triage(pose, kinematics, session_objects, args.fps, thresholds)
        frames = triaged["frame"].to_numpy()

        for stratum in STRATA:
            starts = sample_from(triaged[stratum].to_numpy(bool), frames,
                                 clip_frames, args.per_stratum * 3, rng)
            candidates[stratum] += [(session, start) for start in starts]

    rows = []
    print(f"\n{'faixa':<20} {'candidatos':>11} {'sorteados':>10}")
    print("-" * 44)
    for stratum in STRATA:
        pool = candidates[stratum]
        take = min(args.per_stratum, len(pool))
        picks = rng.choice(len(pool), size=take, replace=False) if pool else []
        print(f"{stratum:<20} {len(pool):>11} {take:>10}")
        for p in picks:
            session, start = pool[p]
            rows.append({
                "clip_id": f"{session}_{start}",
                "session_id": session,
                "start_frame": start,
                "end_frame": start + clip_frames - 1,
                "triage_label": stratum,
            })

    clips = pd.DataFrame(rows).sample(frac=1.0, random_state=args.seed, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Ordem embaralhada e sem a coluna de palpite: o arquivo que o anotador abre
    # não pode revelar o que a triagem achou.
    clips.drop(columns=["triage_label"]).to_csv(args.output, index=False, encoding="utf-8-sig")

    key = args.output.with_name(args.output.stem + "_gabarito.csv")
    clips.to_csv(key, index=False, encoding="utf-8-sig")

    print(f"\n{len(clips)} clipes de {args.clip_sec}s em {args.output}")
    print(f"gabarito (NÃO abrir antes de rotular): {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
