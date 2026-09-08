"""Cria o projeto DeepLabCut de 6 pontos e extrai os quadros para anotação.

Roda no ambiente conda 'deeplabcut' (Python 3.10), não no Python do sistema:

    C:/Users/heitt/miniconda3/envs/deeplabcut/python.exe scripts/create_dlc_project.py

Escolha dos pontos — cada um precisa justificar seu custo de anotação, já que
rotular é o recurso mais caro do projeto:

    nose        já existe no modelo de 2 pontos
    ear_left    orientação da cabeça: separa cheirar de apenas olhar
    ear_right   idem; o par dá o ângulo cabeça-corpo do grooming
    neck        curvatura do corpo
    body        já existe no modelo de 2 pontos
    tail_base   eixo do corpo — é o sinal de rearing, porque a projeção
                focinho-cauda encurta quando o animal fica vertical

Flancos e cauda média foram descartados: custam anotação e acrescentam pouco
sobre o que o pescoço e a base da cauda já dão.
"""

import argparse
import glob
import sys
from pathlib import Path

KEYPOINTS = ["nose", "ear_left", "ear_right", "neck", "body", "tail_base"]

# Apenas para visualização nos gráficos do DLC; não afeta o treino.
SKELETON = [
    ["nose", "neck"],
    ["ear_left", "neck"],
    ["ear_right", "neck"],
    ["ear_left", "ear_right"],
    ["neck", "body"],
    ["body", "tail_base"],
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True, type=Path,
                        help="pasta com os vídeos de arena já separados")
    parser.add_argument("--output", required=True, type=Path,
                        help="onde criar o projeto DLC (fora do repositório git)")
    parser.add_argument("--project", default="mindtrace-pose")
    parser.add_argument("--experimenter", default="memorylab")
    parser.add_argument("--frames-per-video", type=int, default=9,
                        help="9 x 34 vídeos = ~306 quadros, cerca de 2h de anotação")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions

    videos = sorted(glob.glob(str(args.videos / "*.mp4")))
    if not videos:
        print(f"nenhum .mp4 em {args.videos}")
        return 1

    print(f"DeepLabCut {deeplabcut.__version__}")
    print(f"{len(videos)} vídeos | {args.frames_per_video} quadros cada "
          f"= ~{len(videos) * args.frames_per_video} para rotular\n")

    args.output.mkdir(parents=True, exist_ok=True)
    config_path = deeplabcut.create_new_project(
        args.project,
        args.experimenter,
        videos,
        working_directory=str(args.output),
        copy_videos=False,
    )
    print(f"projeto criado: {config_path}")

    config = auxiliaryfunctions.read_config(config_path)
    config["bodyparts"] = KEYPOINTS
    config["skeleton"] = SKELETON
    config["numframes2pick"] = args.frames_per_video
    # Sem recorte: o quadrante do DVR já é 360x240, a entrada nativa do modelo.
    config["cropping"] = False
    auxiliaryfunctions.write_config(config_path, config)

    print(f"pontos configurados: {', '.join(KEYPOINTS)}")

    if args.skip_extract:
        print("\nextração pulada (--skip-extract)")
        return 0

    print("\nextraindo quadros por k-means — isto demora...")
    deeplabcut.extract_frames(
        config_path, mode="automatic", algo="kmeans", userfeedback=False, crop=False
    )

    labeled = Path(config_path).parent / "labeled-data"
    extracted = sum(len(list(d.glob("*.png"))) for d in labeled.iterdir() if d.is_dir())
    print(f"\n{extracted} quadros extraídos em {labeled}")
    print(f"\npróximo passo — abrir a GUI de rotulagem:")
    print(f"  python -m deeplabcut")
    print(f"  (ou: deeplabcut.label_frames(r'{config_path}'))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
