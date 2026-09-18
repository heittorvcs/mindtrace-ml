"""Compara os rótulos humanos com a triagem e varre os limiares.

Responde a pergunta que a taxa de redução sozinha não responde: **o que está
sendo perdido**. Redução alta com escape alto é uma ferramenta que economiza
tempo jogando dado fora, e os dois números só existem juntos.

Duas medidas:

- **precisão por faixa** — quando a triagem diz "caminhando", o humano concorda?
  Precisão baixa numa faixa de rotina significa que ela está engolindo coisa
  alheia.
- **taxa de escape** — dos clipes que a triagem classificou como rotina, quantos
  o humano marcou como `other`, isto é, algo que mereceria revisão e nunca
  chegaria a ela. É a métrica de segurança.

Uso:
    python scripts/evaluate_triage.py --labels data/clip_labels.csv \
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


def triage_all(pose_dir: Path, objects: pd.DataFrame, fps: float,
               thresholds: Thresholds) -> dict:
    """Triagem de cada sessão, indexada por sessão."""
    out = {}
    for path in sorted(pose_dir.glob("*.csv")):
        session_objects = objects[objects.session_id == path.stem]
        if session_objects.empty:
            continue
        pose = pd.read_csv(path, encoding="utf-8-sig")
        kinematics = frame_kinematics(pose, KEYPOINTS, fps, min_confidence=0.25)
        out[path.stem] = triage(pose, kinematics, session_objects, fps, thresholds)
    return out


def clip_verdict(triaged: pd.DataFrame, start: int, end: int) -> str:
    """Faixa dominante no clipe — o mesmo critério que o humano usou."""
    window = triaged[(triaged.frame >= start) & (triaged.frame <= end)]
    if window.empty:
        return "unscorable"
    if "unscorable" in window and window["unscorable"].mean() > 0.5:
        return "unscorable"

    shares = {name: float(window[name].mean()) for name in BEHAVIOR_ORDER}
    best = max(shares, key=shares.get)
    return best if shares[best] > 0.5 else "review"


def score(labels: pd.DataFrame, triages: dict, clips: pd.DataFrame) -> dict:
    merged = clips.merge(labels, on="clip_id")
    merged = merged[merged.label != "unscorable"]
    if merged.empty:
        return {}

    verdicts = [clip_verdict(triages[row.session_id], row.start_frame, row.end_frame)
                if row.session_id in triages else "unscorable"
                for row in merged.itertuples()]
    merged = merged.assign(verdict=verdicts)
    merged = merged[merged.verdict != "unscorable"]

    routine = merged[merged.verdict.isin(BEHAVIOR_ORDER)]
    escapes = routine[routine.label == "other"]

    per_stratum = {}
    for name in BEHAVIOR_ORDER:
        claimed = merged[merged.verdict == name]
        if len(claimed):
            per_stratum[name] = {
                "n": len(claimed),
                "precision": float((claimed.label == name).mean()),
                "escape": float((claimed.label == "other").mean()),
            }

    return {
        "n_clips": len(merged),
        "escape_rate": float(len(escapes) / len(routine)) if len(routine) else float("nan"),
        "per_stratum": per_stratum,
        "escaped_clips": list(escapes.clip_id),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=29.97)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    clips = pd.read_csv(args.key, encoding="utf-8-sig")
    objects = pd.read_csv(args.objects, encoding="utf-8-sig")

    print(f"{len(labels)} clipes rotulados\n")
    print("distribuição dos rótulos humanos:")
    print(labels.label.value_counts().to_string(), "\n")

    print(f"{'caminhada':>10} {'congelam.':>10} {'reducao':>9} {'escape':>8} {'n':>5}")
    print("-" * 48)

    best = None
    for walking in (12.0, 15.0, 18.0, 22.0):
        for freezing in (5.0, 8.0, 11.0):
            thresholds = Thresholds(walking_speed=walking, freezing_speed=freezing)
            triages = triage_all(args.pose, objects, args.fps, thresholds)
            if not triages:
                continue

            reduction = float(np.mean([1 - t["review"].mean() for t in triages.values()]))
            result = score(labels, triages, clips)
            escape = result.get("escape_rate", float("nan"))

            print(f"{walking:>10.0f} {freezing:>10.0f} {reduction:>8.1%} "
                  f"{escape:>7.1%} {result.get('n_clips', 0):>5}")

            if not np.isnan(escape) and (best is None or
                                         (escape <= 0.05 and reduction > best[2])):
                best = (walking, freezing, reduction, escape, result)

    if best:
        walking, freezing, reduction, escape, result = best
        print(f"\n== melhor ponto com escape <= 5% ==")
        print(f"caminhada {walking:.0f} px/s | congelamento {freezing:.0f} px/s")
        print(f"reducao {reduction:.1%} | escape {escape:.1%}\n")
        print(f"{'faixa':<22} {'n':>4} {'precisao':>10} {'escape':>8}")
        print("-" * 48)
        for name, stats in result["per_stratum"].items():
            print(f"{name:<22} {stats['n']:>4} {stats['precision']:>9.1%} {stats['escape']:>7.1%}")
        if result["escaped_clips"]:
            print(f"\nclipes que escaparam (revise estes primeiro):")
            for clip_id in result["escaped_clips"][:10]:
                print(f"  {clip_id}")

    print("\nA amostra é pequena: trate estas taxas como ordem de grandeza, não")
    print("como medida precisa. Mais clipes estreitam o intervalo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
