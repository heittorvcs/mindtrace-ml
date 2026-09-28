"""Converte a planilha da habituação em contagens de referência, uma linha por sessão.

A planilha do laboratório (HAB_F_GERAL, aba ORG) tem, por rata e por dia, o
número de rearings em 20 min e nos primeiros 5 min, e a distância percorrida
nos dois intervalos. O vídeo "Dia 1 - 1-2.MPG" tem a rata ímpar na arena 1 e a
par na arena 2 — a mesma ordem da coluna "Campo" —, e split_arenas os separa em
HD1_F1_cam1 e HD1_F2_cam2.

Uso:
    python scripts/import_habituation_counts.py --input ".../HAB_F_GERAL.xlsx - ORG.csv" \\
        --output data/reference/habituacao_femeas_2025.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def number(value) -> float:
    """Planilha brasileira: vírgula decimal, célula vazia vira NaN."""
    text = str(value).strip().replace(".", "").replace(",", ".") if isinstance(value, str) else value
    return pd.to_numeric(text, errors="coerce")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default="F")
    args = parser.parse_args()

    raw = pd.read_csv(args.input, header=None, dtype=str, encoding="utf-8")
    rows, day = [], None
    for record in raw.itertuples(index=False):
        first = str(record[0]) if isinstance(record[0], str) else ""
        if first.startswith("Dia"):
            day = int(first.split()[1])
        if day is None or not isinstance(record[1], str) or not record[1].strip().isdigit():
            continue
        rat = int(record[1])
        field = int(str(record[2]).strip().rstrip("a"))
        rows.append({
            "session_id": f"HD{day}_{args.prefix}{rat}_cam{field}",
            "day": day,
            "rat": f"{args.prefix}{rat}",
            "box": str(record[2]).strip(),
            "rearings_20min": number(record[5]),
            "rearings_5min": number(record[12]),
            "distance_20min_m": number(record[3]),
            "distance_5min_m": number(record[11]),
        })

    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"{len(table)} sessões em {args.output}")
    print(table.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
