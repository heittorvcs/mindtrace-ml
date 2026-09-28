"""Mede um modelo salvo nas sessões rotuladas depois dele — o teste honesto.

O modelo foi congelado com as sessões listadas em `bundle["sessions"]`; toda
sessão rotulada às cegas depois disso é de um animal ou de um trecho que ele
nunca viu. Diferente da validação deixando uma sessão de fora, aqui nada foi
escolhido olhando estas sessões — nem o limiar.

Como o ponto de operação é decisão do laboratório (pegar tudo e revisar mais,
ou revisar menos e perder alguns), a tabela mostra vários limiares.

Uso:
    python scripts/evaluate_behavior_model.py --model models/rearing_model.joblib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.behavior_model import bouts, compare, frame_mask, gate, session_features  # noqa: E402
from mindtrace_ml.labels import bouts_for, intervals_to_mask  # noqa: E402

THRESHOLDS = (0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--behavior", default="rearing")
    parser.add_argument("--pose", type=Path, default=ROOT / "data" / "pose")
    parser.add_argument("--objects", type=Path, default=ROOT / "data" / "objects.csv")
    parser.add_argument("--video-features", type=Path, default=ROOT / "data" / "video_features")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "session_labels.csv")
    parser.add_argument("--progress", type=Path, default=ROOT / "data" / "session_progress.csv")
    parser.add_argument("--fps", type=float, default=29.97)
    args = parser.parse_args()

    import joblib
    bundle = joblib.load(args.model)
    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    progress = pd.read_csv(args.progress, encoding="utf-8-sig")
    objects = pd.read_csv(args.objects, encoding="utf-8-sig")

    covers = progress.get("behaviors", pd.Series(np.nan, index=progress.index)).fillna(args.behavior)
    assisted = progress.get("assisted", pd.Series(False, index=progress.index)).fillna(False).astype(bool)
    new = progress[~progress.session_id.isin(bundle["sessions"]) & ~assisted
                   & covers.str.contains(args.behavior)]
    if new.empty:
        print("nenhuma sessão rotulada às cegas depois deste modelo")
        return 0

    print(f"modelo treinado em {len(bundle['sessions'])} sessões; testando em {len(new)} novas\n")
    rows = []
    for record in new.itertuples():
        session = record.session_id
        pose = pd.read_csv(args.pose / f"{session}.csv", encoding="utf-8-sig")
        pixels = pd.read_csv(args.video_features / f"{session}.csv", encoding="utf-8-sig")
        data = session_features(pose, objects[objects.session_id == session], args.fps, pixels)
        seen = data["frames"] <= record.watched_until
        marked = intervals_to_mask(bouts_for(labels[labels.session_id == session], args.behavior),
                                   data["frames"])[seen]
        scores = bundle["models"][args.behavior].predict_proba(data["X"][bundle["features"]])[:, 1]
        for threshold in THRESHOLDS:
            mask = gate(args.behavior, frame_mask(scores, data["centers"], len(data["frames"]),
                                                  threshold, args.fps), data["freezing"], args.fps)[seen]
            result = compare(marked, mask, args.fps)
            hits = sum(mask[a:b + 1].any() for a, b in bouts(marked))
            real = sum(marked[a:b + 1].any() for a, b in bouts(mask))
            rows.append({"session": session, "threshold": threshold, "episodes": len(bouts(marked)),
                         "found": hits, "false": len(bouts(mask)) - real,
                         "manual_s": result["manual_s"], "detected_s": result["detected_s"]})
    table = pd.DataFrame(rows)

    for session, group in table.groupby("session", sort=False):
        first = group.iloc[0]
        print(f"{session}: {int(first.episodes)} episódios marcados, {first.manual_s:.0f} s")
    total = table.groupby("threshold", sort=False)[["episodes", "found", "false", "manual_s", "detected_s"]].sum()
    sessions = table.session.nunique()

    print(f"\n{'limiar':>7} {'achados':>12} {'alarmes falsos':>16} {'tempo apontado':>16}")
    for threshold, row in total.iterrows():
        print(f"{threshold:>7.2f} {int(row.found):>4} de {int(row.episodes):<4} "
              f"{row.false / sessions:>10.1f} /sessão {row.detected_s:>9.0f} s (marcado {row.manual_s:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
