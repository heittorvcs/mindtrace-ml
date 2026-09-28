"""Extrai os sinais de pixel (movimento e silhueta) de todas as sessões.

Decodificar o vídeo é a parte lenta, então o resultado fica em cache, um CSV por
sessão, alinhado linha a linha com o CSV de pose. Retomável: pula o que já existe.

Uso:
    python scripts/extract_video_features.py --videos .../arenas --pose data/pose
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.video_features import video_signals  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "video_features")
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.pose.glob("*.csv"))
    if args.sessions:
        paths = [p for p in paths if p.stem in set(args.sessions)]

    for index, path in enumerate(paths, start=1):
        destination = args.output / path.name
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(paths)}] {path.stem} — já existe", flush=True)
            continue
        started = time.perf_counter()
        pose = pd.read_csv(path, encoding="utf-8-sig")
        signals = video_signals(args.videos / f"{path.stem}.mp4", pose)
        signals.insert(0, "frame", pose["frame"].to_numpy())
        signals.to_csv(destination, index=False, encoding="utf-8-sig")
        found = np.isfinite(signals.blob_area).mean()
        print(f"[{index}/{len(paths)}] {path.stem} — silhueta em {found:.0%} dos quadros, "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
