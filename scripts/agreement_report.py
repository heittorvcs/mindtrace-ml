"""Concordância, sessão a sessão, entre o rearing detectado e o marcado à mão.

Compara as duas grandezas que o laboratório reporta — tempo total de rearing e
número de episódios — com as estatísticas de agreement.py: viés, equivalência
(TOST) e limites de Bland-Altman.

Duas formas de obter o valor automático, ambas com sessões que o modelo não viu:

- `--model`: o modelo salvo, nas sessões rotuladas às cegas depois dele;
- `--leave-one-out`: cada sessão rotulada às cegas é medida por um modelo
  treinado em todas as outras. Usa todas as sessões, então tem mais poder; o
  limiar é o mesmo, mas o modelo muda de uma sessão para outra.

Uso:
    python scripts/agreement_report.py --model models/rearing_model.joblib --threshold 0.2
    python scripts/agreement_report.py --leave-one-out --threshold 0.2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.agreement import agreement  # noqa: E402
from mindtrace_ml.behavior_model import (  # noqa: E402
    bouts,
    frame_mask,
    gate,
    label_mask,
    make_model,
    session_features,
    targets,
)


def episodes(mask: np.ndarray) -> int:
    """Episódios. Rótulos e detecção já vêm com as descidas breves fundidas."""
    return len(bouts(mask))


def blind_sessions(progress: pd.DataFrame, behavior: str) -> pd.DataFrame:
    covers = progress.get("behaviors", pd.Series(np.nan, index=progress.index)).fillna(behavior)
    assisted = progress.get("assisted", pd.Series(False, index=progress.index)).fillna(False).astype(bool)
    return progress[covers.str.contains(behavior) & ~assisted]


def describe(name: str, unit: str, result: dict) -> None:
    low95, high95 = result["ci95"]
    lower, upper = result["limits_of_agreement"]
    low_pct, high_pct = result["limits_pct"]
    print(f"\n{name} ({result['n']} sessões)")
    print(f"  média manual {result['manual_mean']:.1f} {unit}, automática {result['automatic_mean']:.1f} {unit}")
    print(f"  diferença média {result['bias']:+.1f} {unit} ({result['bias_pct']:+.0f}%), "
          f"IC 95% {low95:+.1f} a {high95:+.1f}")
    print(f"  teste t pareado p = {result['paired_t_p']:.2f} | Wilcoxon p = {result['wilcoxon_p']:.2f}")
    print(f"  equivalência com margem ±{result['margin']:.1f} {unit} (±15% da média): "
          f"p = {result['tost_p']:.3f} -> {'EQUIVALENTE' if result['tost_p'] < 0.05 else 'não demonstrada'}")
    print(f"  menor margem que os dados sustentam: ±{result['smallest_margin_pct']:.0f}% da média")
    print(f"  uma sessão isolada erra entre {lower:+.1f} e {upper:+.1f} {unit} "
          f"({low_pct:+.0f}% a {high_pct:+.0f}%) — limites de Bland-Altman")
    print(f"  concordância de Lin {result['ccc']:.2f} | Pearson {result['pearson_r']:.2f} | "
          f"erro médio por sessão {result['mean_abs_pct_error']:.0f}%")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--behavior", default="rearing")
    parser.add_argument("--margin", type=float, default=0.15, help="margem de equivalência, fração da média")
    parser.add_argument("--pose", type=Path, default=ROOT / "data" / "pose")
    parser.add_argument("--objects", type=Path, default=ROOT / "data" / "objects.csv")
    parser.add_argument("--video-features", type=Path, default=ROOT / "data" / "video_features")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "session_labels.csv")
    parser.add_argument("--progress", type=Path, default=ROOT / "data" / "session_progress.csv")
    parser.add_argument("--fps", type=float, default=29.97)
    args = parser.parse_args()
    if not args.model and not args.leave_one_out:
        parser.error("use --model ou --leave-one-out")

    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    progress = blind_sessions(pd.read_csv(args.progress, encoding="utf-8-sig"), args.behavior)
    objects = pd.read_csv(args.objects, encoding="utf-8-sig")

    bundle = None
    if args.model:
        import joblib
        bundle = joblib.load(args.model)
        progress = progress[~progress.session_id.isin(bundle["sessions"])]

    sessions = {}
    for record in progress.itertuples():
        pose = pd.read_csv(args.pose / f"{record.session_id}.csv", encoding="utf-8-sig")
        pixels = pd.read_csv(args.video_features / f"{record.session_id}.csv", encoding="utf-8-sig")
        data = session_features(pose, objects[objects.session_id == record.session_id], args.fps, pixels)
        data["seen"] = data["centers"] <= record.watched_until
        data["watched"] = record.watched_until
        own = labels[labels.session_id == record.session_id]
        data["y"] = targets(own, data["frames"], data["centers"], args.fps)
        data["marked"] = label_mask(own, args.behavior, data["frames"], args.fps)
        sessions[record.session_id] = data

    rows = []
    for name, data in sessions.items():
        if bundle:
            model = bundle["models"][args.behavior]
            X = data["X"][bundle["features"]]
        else:
            others = [s for s in sessions if s != name]
            model = make_model().fit(
                pd.concat([sessions[s]["X"][sessions[s]["seen"]] for s in others]),
                np.concatenate([sessions[s]["y"][args.behavior].to_numpy()[sessions[s]["seen"]]
                                for s in others]))
            X = data["X"]
        scores = model.predict_proba(X)[:, 1]
        mask = gate(args.behavior, frame_mask(scores, data["centers"], len(data["frames"]),
                                              args.threshold, args.fps), data["freezing"], args.fps)
        seen = data["frames"] <= data["watched"]
        marked, detected = data["marked"][seen], mask[seen]
        rows.append({"sessão": name,
                     "tempo manual (s)": marked.sum() / args.fps,
                     "tempo automático (s)": detected.sum() / args.fps,
                     "episódios manual": episodes(marked),
                     "episódios automático": episodes(detected)})
        print(".", end="", flush=True)
    table = pd.DataFrame(rows)

    source = "modelo salvo, sessões rotuladas depois dele" if bundle else "cada sessão medida sem entrar no treino"
    print(f"\n\nlimiar {args.threshold} — {source}\n")
    print(table.round(1).to_string(index=False))
    describe("tempo de rearing", "s", agreement(table["tempo manual (s)"], table["tempo automático (s)"], args.margin))
    describe("número de rearings", "episódios",
             agreement(table["episódios manual"], table["episódios automático"], args.margin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
