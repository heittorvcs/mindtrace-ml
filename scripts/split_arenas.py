"""Separa os vídeos do DVR em um vídeo por arena, e gera o manifesto da sessão.

O DVR Intelbras grava um mosaico 2×2 de 720×480: CAM 1, 2 e 3 são as arenas e
CAM 4 fica vazia. Cada quadrante tem 360×240, que é exatamente a entrada do
modelo de pose — o recorte não precisa de reescala.

O nome do arquivo carrega os animais por posição: "TR 19-20-21.MPG" são os
animais 19, 20 e 21 nas câmeras 1, 2 e 3; "TT 15-X-17.MPG" tem a câmera 2 vazia.
Arenas marcadas com X são puladas.

Uso:
    python scripts/split_arenas.py --input  C:/Users/heitt/Documents/videos_lab \
                                   --output C:/Users/heitt/Documents/videos_lab/arenas
"""

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

# Deslocamento de cada câmera no mosaico, em (x, y). CAM 4 é descartada.
CAMERA_OFFSETS = {1: (0, 0), 2: (360, 0), 3: (0, 240)}
QUADRANT_WIDTH, QUADRANT_HEIGHT = 360, 240

NAME_PATTERN = re.compile(r"^(?P<phase>[A-Z]{2})\s+(?P<ids>[\dX]+(?:-[\dX]+)*)", re.IGNORECASE)


def parse_name(stem: str):
    """'TR 19-20-21' -> ('TR', {1: '19', 2: '20', 3: '21'}); X vira None."""
    match = NAME_PATTERN.match(stem.strip())
    if not match:
        return None, {}

    parts = match.group("ids").split("-")
    animals = {}
    for index, part in enumerate(parts[:3], start=1):
        animals[index] = None if part.upper() == "X" else part
    return match.group("phase").upper(), animals


def file_digest(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_with_ffmpeg(source: Path, destination: Path, offset, crf: int) -> bool:
    x, y = offset
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-filter:v", f"crop={QUADRANT_WIDTH}:{QUADRANT_HEIGHT}:{x}:{y}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-an",
        str(destination),
    ]
    return subprocess.run(command, capture_output=True).returncode == 0


def crop_with_opencv(source: Path, destination: Path, offset) -> bool:
    """Reserva para quando o ffmpeg não estiver disponível. Mais lento."""
    x, y = offset
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return False

    fps = capture.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (QUADRANT_WIDTH, QUADRANT_HEIGHT),
    )
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame[y:y + QUADRANT_HEIGHT, x:x + QUADRANT_WIDTH])

    capture.release()
    writer.release()
    return True


def probe(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    info = {
        "fps": round(capture.get(cv2.CAP_PROP_FPS), 4),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return info


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crf", type=int, default=18, help="18 é visualmente sem perdas")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="retoma sem refazer o que já foi cortado")
    args = parser.parse_args()

    sources = sorted(p for p in args.input.iterdir() if p.suffix.upper() == ".MPG")
    if not sources:
        print(f"nenhum .MPG em {args.input}")
        return 1

    have_ffmpeg = shutil.which("ffmpeg") is not None
    print(f"{len(sources)} vídeos encontrados | ffmpeg: {'sim' if have_ffmpeg else 'não (usando OpenCV)'}\n")

    seen_digests = {}
    manifest_rows = []
    skipped_duplicates = []

    args.output.mkdir(parents=True, exist_ok=True)

    for source in sources:
        digest = file_digest(source)
        if digest in seen_digests:
            skipped_duplicates.append((source.name, seen_digests[digest]))
            print(f"  DUPLICATA  {source.name}  (idêntico a {seen_digests[digest]}) — pulado")
            continue
        seen_digests[digest] = source.name

        phase, animals = parse_name(source.stem)
        if phase is None:
            print(f"  ?  {source.name} — nome fora do padrão, pulado")
            continue

        info = probe(source)
        occupied = {camera: animal for camera, animal in animals.items() if animal}
        print(f"  {source.name}  [{phase}]  {info['width']}x{info['height']} @ {info['fps']} fps"
              f"  -> câmeras {sorted(occupied)}")

        for camera, animal in occupied.items():
            destination = args.output / f"{phase}_{animal}_cam{camera}.mp4"
            if not args.dry_run and not (args.skip_existing and destination.exists()):
                offset = CAMERA_OFFSETS[camera]
                ok = (crop_with_ffmpeg(source, destination, offset, args.crf)
                      if have_ffmpeg else crop_with_opencv(source, destination, offset))
                if not ok:
                    print(f"      FALHA ao cortar câmera {camera}")
                    continue

            manifest_rows.append({
                "session_id": destination.stem,
                "animal_id": animal,
                "phase": phase,
                "camera": camera,
                "field": camera - 1,
                "source_video": source.name,
                "video": destination.name,
                "fps": info["fps"],
                "frames": info["frames"],
            })

    manifest = pd.DataFrame(manifest_rows)
    if not args.dry_run and not manifest.empty:
        manifest_path = args.output / "manifest.csv"
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"\nmanifesto: {manifest_path}")

    print(f"\n{len(manifest)} arenas extraídas de {len(seen_digests)} vídeos únicos")
    if skipped_duplicates:
        print(f"{len(skipped_duplicates)} duplicata(s) ignorada(s)")
    if not manifest.empty:
        print(f"animais distintos: {manifest['animal_id'].nunique()}")
        print(f"fases: {dict(manifest['phase'].value_counts())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
