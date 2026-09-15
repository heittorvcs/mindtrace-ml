"""Relatório do andamento da rotulagem de pose, lido direto dos arquivos do DLC.

Distingue três situações que a GUI mistura:

- **completo**: todos os pontos marcados
- **parcial**: alguns pontos marcados, o resto ausente — esperado em oclusão, e
  correto: marcar só o que é identificável é melhor que chutar
- **vazio**: nenhum ponto. Pode ser decisão deliberada (mão na caixa, animal
  fora da arena) ou pasta interrompida no meio. Uma pasta com muitos vazios em
  meio a pastas completas costuma ser interrupção, não critério.

Uso:
    python scripts/label_progress.py --project C:/Users/heitt/Documents/dlc/mindtrace-pose-...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def count_points(labels_csv: Path) -> dict[str, int]:
    """Pontos marcados por imagem. O CSV do DLC indexa pelo caminho em 3 níveis."""
    frame = pd.read_csv(labels_csv, header=[0, 1, 2], index_col=[0, 1, 2])
    counts = {}
    for index, row in frame.iterrows():
        values = row.to_numpy(dtype=float)
        counts[str(index[-1])] = int(np.sum(~np.isnan(values)) // 2)
    return counts


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--keypoints", type=int, default=6)
    args = parser.parse_args()

    labeled_data = args.project / "labeled-data"
    if not labeled_data.is_dir():
        print(f"não encontrei labeled-data em {args.project}")
        return 1

    folders = sorted(d for d in labeled_data.iterdir()
                     if d.is_dir() and not d.name.endswith("_labeled"))

    print(f"{'pasta':<16} {'imgs':>5} {'ok':>4} {'parc':>5} {'vazio':>6}   quadros vazios")
    print("-" * 92)

    totals = {"images": 0, "complete": 0, "partial": 0, "empty": 0}
    untouched = []

    for folder in folders:
        images = sorted(p.name for p in folder.glob("*.png"))
        label_files = list(folder.glob("CollectedData_*.csv"))
        totals["images"] += len(images)

        if not label_files:
            untouched.append(folder.name)
            totals["empty"] += len(images)
            print(f"{folder.name:<16} {len(images):>5} {'-':>4} {'-':>5} {len(images):>6}   (não iniciada)")
            continue

        counts = count_points(label_files[0])
        complete = [i for i in images if counts.get(i, 0) == args.keypoints]
        partial = [i for i in images if 0 < counts.get(i, 0) < args.keypoints]
        empty = [i for i in images if counts.get(i, 0) == 0]

        totals["complete"] += len(complete)
        totals["partial"] += len(partial)
        totals["empty"] += len(empty)

        print(f"{folder.name:<16} {len(images):>5} {len(complete):>4} {len(partial):>5} "
              f"{len(empty):>6}   {', '.join(empty)}")

    print("-" * 92)
    print(f"{'TOTAL':<16} {totals['images']:>5} {totals['complete']:>4} "
          f"{totals['partial']:>5} {totals['empty']:>6}")

    labelled = totals["complete"] + totals["partial"]
    print(f"\nprogresso: {labelled}/{totals['images']} quadros com ao menos um ponto "
          f"({labelled / totals['images']:.0%})" if totals["images"] else "")
    if untouched:
        print(f"pastas não iniciadas ({len(untouched)}): {', '.join(untouched)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
