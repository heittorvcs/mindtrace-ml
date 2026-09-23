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
    # 25 por faixa dá 100 clipes de rotina, que é onde a estimativa de escape
    # deixa de ser ruído: em torno de 5%, a margem cai de ±6 para ±4 pontos.
    # Escape só é observável nos clipes classificados como rotina, então é esse
    # número — não o total — que governa a precisão.
    parser.add_argument("--per-stratum", type=int, default=25)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--strata", default=None,
                        help="ex.: low_activity:50,walking:30 — padrão: todos, --per-stratum cada")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[],
                        help="CSVs de clipes já sorteados, para não repeti-los")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    objects = pd.read_csv(args.objects, encoding="utf-8-sig")
    # Limiares padrão do módulo: o sorteio precisa usar os mesmos que a triagem
    # avaliada, senão o estrato do clipe não corresponde ao que se mede.
    thresholds = Thresholds()

    quotas = {stratum: args.per_stratum for stratum in STRATA}
    if args.strata:
        quotas = {}
        for item in args.strata.split(","):
            name, count = item.split(":")
            if name not in STRATA:
                print(f"estrato desconhecido: {name}. Válidos: {', '.join(STRATA)}")
                return 1
            quotas[name] = int(count)

    # Janelas já rotuladas, por sessão, para excluir sobreposição e não só
    # clip_id idêntico — um clipe deslocado de poucos quadros seria o mesmo trecho.
    taken = {}
    for path in args.exclude:
        previous = pd.read_csv(path, encoding="utf-8-sig")
        for row in previous.itertuples():
            taken.setdefault(row.session_id, []).append((row.start_frame, row.end_frame))
    clip_frames = int(round(args.clip_sec * args.fps))
    rng = np.random.default_rng(args.seed)

    sessions = sorted(args.pose.glob("*.csv"))
    if not sessions:
        print(f"nenhum CSV de pose em {args.pose}")
        return 1

    # Junta todos os candidatos por faixa e só depois sorteia, para que o número
    # de clipes por faixa não dependa de quantas sessões cada uma domina.
    candidates = {stratum: [] for stratum in quotas}

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

        for stratum, quota in quotas.items():
            starts = sample_from(triaged[stratum].to_numpy(bool), frames,
                                 clip_frames, quota * 3, rng)
            for start in starts:
                end = start + clip_frames - 1
                if any(start <= e and end >= s0 for s0, e in taken.get(session, [])):
                    continue
                candidates[stratum].append((session, start))

    rows = []
    print(f"\n{'faixa':<20} {'candidatos':>11} {'sorteados':>10}")
    print("-" * 44)
    for stratum, quota in quotas.items():
        pool = candidates[stratum]
        take = min(quota, len(pool))
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
