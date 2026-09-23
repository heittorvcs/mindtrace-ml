"""Compara os rótulos humanos com a triagem, por presença.

Responde a pergunta que a taxa de redução sozinha não responde: **o que está
sendo perdido**. Redução alta com escape alto é uma ferramenta que economiza
tempo jogando dado fora, e os dois números só existem juntos.

A avaliação é por **presença**, igual à rotulagem. Uma versão anterior comparava
o rótulo humano com a faixa *dominante* do clipe, e isso contava como escape os
clipes em que o detector disparou corretamente num trecho: a triagem é
multi-etiqueta, e um quadro pode ser movimento lento e exploração ao mesmo tempo.
Num clipe com 40% de exploração o detector de objeto acertava, mas o veredito
dominante dizia "movimento lento".

Medidas:

- **detecção de exploração** — nos clipes que o humano marcou como exploração, o
  detector de objeto disparou em ao menos `--min-presence` do clipe?
- **escape** — nos clipes com algo notável (grooming, rearing, outro), o trecho
  chegou ao pesquisador, seja pela fila de revisão, seja rotulado por um detector
  daquele comportamento? Os que não chegaram são escapes. É a métrica de
  segurança.

O escape é reportado também ponderado pelo tamanho de cada estrato na sessão:
a amostragem sorteia o mesmo número de clipes por estrato, então a média simples
superrepresenta os estratos pequenos.

Uso:
    python scripts/evaluate_triage.py --labels data/clip_labels.csv \\
        --key data/clips_gabarito.csv --pose data/pose --objects data/objects.csv
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

# Rótulos das rodadas de anotação, normalizados. A primeira tinha `other`
# genérico e distinguia congelamento de movimento lento; a segunda separou
# grooming e rearing; a de teste acrescenta sniffing e seleção múltipla.
STILL = {"freezing", "low_activity", "still"}
NOTABLE = {"other", "grooming", "rearing"}


def label_sets(labels: pd.DataFrame) -> pd.Series:
    """Conjunto de rótulos por clipe, qualquer que seja a rodada.

    Rodadas antigas têm um rótulo em `label` e, em dois casos, extras em `also`;
    a de teste tem todos em `labels`, separados por ponto e vírgula.
    """
    def one(row):
        if isinstance(row.get("labels"), str) and row["labels"]:
            return set(row["labels"].split(";"))
        names = {row["label"]}
        if isinstance(row.get("also"), str) and row["also"]:
            names |= set(row["also"].split(";"))
        return names
    return labels.apply(one, axis=1)


def triage_sessions(pose_dir: Path, objects: pd.DataFrame, fps: float,
                    thresholds: Thresholds) -> dict:
    out = {}
    for path in sorted(pose_dir.glob("*.csv")):
        session_objects = objects[objects.session_id == path.stem]
        if session_objects.empty:
            continue
        pose = pd.read_csv(path, encoding="utf-8-sig")
        out[path.stem] = triage(pose, frame_kinematics(pose, KEYPOINTS, fps),
                                session_objects, fps, thresholds)
    return out


def clip_presence(triaged: pd.DataFrame, start: int, end: int) -> dict:
    """Fração do clipe coberta por cada detector, pela revisão e pelo dado faltante."""
    window = triaged[(triaged.frame >= start) & (triaged.frame <= end)]
    columns = [c for c in triaged.columns if c not in ("frame", "time_ms")]
    if window.empty:
        return {c: 0.0 for c in columns}
    return {c: float(window[c].mean()) for c in columns}


def evaluate(merged: pd.DataFrame, triages: dict, min_presence: float,
             notable_set: set) -> pd.DataFrame:
    rows = []
    for clip in merged.itertuples():
        names = clip.names
        if clip.session_id not in triages or names == {"unscorable"}:
            continue
        presence = clip_presence(triages[clip.session_id], clip.start_frame, clip.end_frame)

        surfaced = (presence.get("review", 0) >= min_presence
                    or presence.get("unscorable", 0) >= min_presence)
        notable = names & notable_set

        rows.append({
            "clip_id": clip.clip_id,
            "stratum": clip.triage_label,
            "label": "+".join(sorted(names)),
            "notable": bool(notable),
            "notable_kinds": sorted(notable),
            "exploration": "object_interaction" in names,
            "object_detected": presence.get("object_interaction", 0) >= min_presence,
            "surfaced": surfaced,
        })
    return pd.DataFrame(rows)


def stratum_weights(triages: dict, strata) -> dict:
    """Parcela aproximada de quadros de cada estrato, média entre sessões."""
    shares = {s: np.mean([t[s].mean() for t in triages.values() if s in t]) for s in strata}
    total = sum(shares.values())
    return {s: v / total for s, v in shares.items()} if total else shares


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--min-presence", type=float, default=0.10,
                        help="fração mínima do clipe (0,10 de 3 s = 0,3 s)")
    parser.add_argument("--sniffing-is-notable", action="store_true",
                        help="conta sniffing fora do objeto como 'outros' em vez de rotina")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    labels["names"] = label_sets(labels)
    key = pd.read_csv(args.key, encoding="utf-8-sig")
    objects = pd.read_csv(args.objects, encoding="utf-8-sig")
    merged = key.merge(labels[["clip_id", "names"]], on="clip_id")

    notable_set = NOTABLE | ({"sniffing"} if args.sniffing_is_notable else set())

    triages = triage_sessions(args.pose, objects, args.fps, Thresholds())
    result = evaluate(merged, triages, args.min_presence, notable_set)

    print(f"{len(result)} clipes avaliados | presença mínima {args.min_presence:.0%} do clipe")
    print(f"sniffing conta como: {'outros' if args.sniffing_is_notable else 'rotina'}\n")
    counts = merged["names"].explode().value_counts().to_dict()
    print("rótulos humanos (um clipe pode ter vários):", counts, "\n")

    exploration = result[result.exploration]
    if len(exploration):
        print(f"exploração detectada: {exploration.object_detected.mean():.0%} "
              f"({int(exploration.object_detected.sum())}/{len(exploration)})")

    notable = result[result.notable]
    if notable.empty:
        print("nenhum clipe notável rotulado")
        return 0

    print(f"\n{'estrato':<20} {'notáveis':>9} {'sinalizados':>12} {'escapes':>8}")
    print("-" * 52)
    weights = stratum_weights(triages, list(result.stratum.unique()))
    weighted_notable = weighted_escaped = 0.0
    for stratum, group in notable.groupby("stratum"):
        escaped = int((~group.surfaced).sum())
        print(f"{stratum:<20} {len(group):>9} {group.surfaced.mean():>11.0%} {escaped:>8}")

        total_in_stratum = (result.stratum == stratum).sum()
        rate_notable = len(group) / total_in_stratum
        weighted_notable += weights.get(stratum, 0) * rate_notable
        weighted_escaped += weights.get(stratum, 0) * rate_notable * (1 - group.surfaced.mean())

    print("-" * 52)
    print(f"escape simples: {(~notable.surfaced).mean():.0%} "
          f"({int((~notable.surfaced).sum())}/{len(notable)} clipes notáveis)")
    if weighted_notable:
        print(f"escape ponderado pelo tamanho dos estratos: {weighted_escaped / weighted_notable:.0%}")

    by_type = notable.explode("notable_kinds").groupby("notable_kinds").surfaced.agg(["size", "mean"])
    if len(by_type) > 1:
        print("\npor tipo de evento:")
        for label, row in by_type.iterrows():
            print(f"  {label:<10} n={int(row['size']):>3}  sinalizados {row['mean']:.0%}")

    print("\nAmostra pequena: trate as taxas como ordem de grandeza.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
