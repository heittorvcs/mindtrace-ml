"""Extrai pose de todos os vídeos de uma pasta, um CSV por vídeo.

Ao contrário do CSV do MindTrace, o índice de quadro aqui é o do vídeo e há
timestamp — nenhum quadro é descartado e o resultado independe da velocidade da
máquina. É isso que permite alinhar uma anotação humana feita em tempo de vídeo.

Retomável: pula os vídeos que já têm CSV completo.

Uso em CPU:
    python scripts/extract_pose.py --videos .../arenas --model .../Network-MemoryLab-v2.onnx --output data/pose

Em GPU AMD via DirectML (requer: pip install onnxruntime-directml):
    python scripts/extract_pose.py ... --provider DmlExecutionProvider
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.pose import PoseModel, extract_video  # noqa: E402
from mindtrace_ml.schema import CURRENT_KEYPOINTS, PROPOSED_KEYPOINTS  # noqa: E402


def keypoints_for(count: int):
    """Nomeia os pontos pela contagem do modelo, com fallback genérico."""
    if count == len(CURRENT_KEYPOINTS):
        return CURRENT_KEYPOINTS
    if count == len(PROPOSED_KEYPOINTS):
        return PROPOSED_KEYPOINTS
    return tuple(f"kp{i}" for i in range(count))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", default="CPUExecutionProvider",
                        help="CPUExecutionProvider ou DmlExecutionProvider")
    parser.add_argument("--layout", default="full", choices=["full", "mosaic2x2"],
                        help="'full' para as arenas já separadas")
    parser.add_argument("--fields", default="0", help="lista separada por vírgula")
    parser.add_argument("--every", type=int, default=1, help="processa 1 a cada N quadros")
    parser.add_argument("--equalize", action="store_true",
                        help="só para vídeos de baixo contraste de outro domínio")
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    videos = sorted(args.videos.glob(args.pattern))
    if not videos:
        print(f"nenhum vídeo em {args.videos} com padrão {args.pattern}")
        return 1

    import onnxruntime as ort
    available = ort.get_available_providers()
    if args.provider not in available:
        print(f"provider '{args.provider}' indisponível. Disponíveis: {available}")
        return 1

    model = PoseModel(args.model, providers=[args.provider])
    names = keypoints_for(model.n_keypoints)
    fields = tuple(int(f) for f in args.fields.split(","))

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(videos)} vídeos | provider: {args.provider}")
    print(f"modelo: {model.n_keypoints} pontos -> {', '.join(names)}\n")

    started = time.perf_counter()
    processed = 0

    for index, video in enumerate(videos, start=1):
        destination = args.output / f"{video.stem}.csv"
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(videos)}] {video.name} — já existe, pulado")
            continue

        video_started = time.perf_counter()
        rows, meta = extract_video(
            video, model, fields=fields, layout=args.layout,
            every=args.every, use_equalize=args.equalize, keypoints=names,
        )
        frame = pd.DataFrame(rows)
        frame.to_csv(destination, index=False, encoding="utf-8-sig")

        elapsed = time.perf_counter() - video_started
        valid = frame[f"{names[0]}_p"].ge(0.75).mean() if len(frame) else 0.0
        print(f"[{index}/{len(videos)}] {video.name} — {len(frame)} linhas, "
              f"{meta['fps']:.2f} fps, cobertura {valid:.0%}, {elapsed:.0f}s")
        processed += 1

    total = time.perf_counter() - started
    print(f"\n{processed} vídeos processados em {total/60:.1f} min")
    if processed:
        print(f"média: {total/processed:.0f}s por vídeo")
    print(f"CSVs em {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
