"""Avalia a cadeia de regras do MindTrace contra os rótulos humanos.

Uso:
    python scripts/evaluate_baseline.py --features data/features/S001.csv \
                                        --labels   data/labels/S001.csv \
                                        --field 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml import (  # noqa: E402
    bout_metrics,
    cooccurrence,
    coverage,
    frame_index,
    frame_metrics,
    label_matrix,
    load_features,
    load_labels,
    multilabel_rate,
    overlapping_within_behavior,
    rule_predictions,
    unreachable_recall,
)


def main() -> int:
    # O console do Windows abre em cp1252 e mutila os acentos do relatório.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--field", type=int, default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    features = load_features(args.features)
    labels = load_labels(args.labels)
    frames = frame_index(features)

    y_true, scorable = label_matrix(labels, frames, args.field)
    y_pred = rule_predictions(features)

    stats = coverage(features)
    print("\n== Cobertura da sessão ==")
    print(f"quadros registrados : {stats['recorded']}")
    print(f"intervalo total     : {stats['span']}")
    print(f"cobertura           : {stats['coverage']:.1%}  (perdas por confiança de pose < 0,75)")
    print(f"maior lacuna        : {stats['largest_gap']} quadros")
    print(f"quadros pontuáveis  : {int(scorable.sum())} de {len(frames)}")

    overlaps = overlapping_within_behavior(labels)
    if not overlaps.empty:
        print(f"\n!! {len(overlaps)} sobreposições dentro do mesmo comportamento — bouts a fundir")

    print("\n== Estrutura multi-etiqueta ==")
    print(f"quadros com 2+ comportamentos: {multilabel_rate(y_true, scorable):.1%}")
    print("\ncoocorrência (fração de quadros pontuáveis):")
    print(cooccurrence(y_true, scorable).round(3).to_string())

    print("\n== Teto estrutural das regras ==")
    print("revocação inatingível pela exclusividade mútua:")
    print(unreachable_recall(y_true, scorable).round(2).to_string(index=False))

    print("\n== Regras: desempenho por quadro ==")
    print(frame_metrics(y_true, y_pred, scorable).round(3).to_string(index=False))

    print("\n== Regras: desempenho por bout ==")
    print(bout_metrics(y_true, y_pred, frames, scorable, args.iou).round(2).to_string(index=False))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
