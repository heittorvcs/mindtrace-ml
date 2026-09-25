"""Camada de movimento de todas as sessões: dados por quadro, resumo e validação.

Grava, para cada sessão, um CSV por quadro — posição, velocidade, movendo,
freezing, exploração de cada objeto — e uma planilha-resumo com uma linha por
sessão. É o que o laboratório usa: tempo de exploração de cada objeto,
distância percorrida, tempo em freezing.

Se houver sessões rotuladas por completo, compara a exploração medida com a
marcada à mão. É a validação que um artigo de NOR precisa mostrar: o tempo
automático bate com o de um observador?

`--min-exploration` aplica o critério de exclusão do NOR — exploração total
mínima —, o mesmo para todos os animais. O valor é decisão do laboratório.

Uso:
    python scripts/movement_report.py --pose data/pose --objects data/objects.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.kinematics import frame_kinematics  # noqa: E402
from mindtrace_ml.labels import bouts_for, intervals_to_mask  # noqa: E402
from mindtrace_ml.movement import movement_layer, session_summary  # noqa: E402
from mindtrace_ml.triage_model import KEYPOINTS  # noqa: E402


def compare_exploration(layers: dict, labels: pd.DataFrame, progress: pd.DataFrame,
                        pose_dir: Path, objects: pd.DataFrame, fps: float):
    """Tempo total e preferência entre objetos: regra × marcação manual.

    A marcação não diz qual objeto o rato explorava; cada quadro marcado é
    atribuído ao objeto mais perto do focinho. A preferência é a fração do tempo
    de exploração gasta no objeto 2 — o índice do NOR é função direta dela.
    """
    rows = []
    for session, layer in layers.items():
        done = progress[progress.session_id == session]
        if done.empty:
            continue
        # Só o trecho assistido: depois dele, "sem marcação" não quer dizer nada.
        seen = layer.frame.to_numpy() <= int(done.watched_until.max())
        marked = intervals_to_mask(bouts_for(labels[labels.session_id == session], "object_interaction"),
                                   layer.frame.to_numpy()) & seen
        measured = layer.exploring.to_numpy() & seen

        pose = pd.read_csv(pose_dir / f"{session}.csv", encoding="utf-8-sig")
        session_objects = objects[objects.session_id == session].sort_values("object_id")
        distances = np.stack([np.hypot(pose.nose_x - o.center_x, pose.nose_y - o.center_y) - o.radius
                              for o in session_objects.itertuples()])
        nearest = session_objects.object_id.to_numpy()[np.argmin(distances, axis=0)]
        manual_2 = (marked & (nearest == 2)).sum()
        measured_2 = (layer["exploring_2"].to_numpy() & seen).sum() if "exploring_2" in layer else 0

        both = (marked & measured).sum()
        rows.append({
            "sessão": session,
            "manual_s": marked.sum() / fps,
            "automático_s": measured.sum() / fps,
            "marcado que a regra pegou": both / marked.sum() if marked.any() else np.nan,
            "da regra que estava marcado": both / measured.sum() if measured.any() else np.nan,
            "objeto 2, manual": manual_2 / marked.sum() if marked.any() else np.nan,
            "objeto 2, automático": measured_2 / measured.sum() if measured.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "movement")
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "session_labels.csv")
    parser.add_argument("--progress", type=Path, default=ROOT / "data" / "session_progress.csv")
    parser.add_argument("--min-exploration", type=float, default=None,
                        help="segundos de exploração total abaixo dos quais a sessão é sinalizada")
    args = parser.parse_args()

    objects = pd.read_csv(args.objects, encoding="utf-8-sig")
    args.output.mkdir(parents=True, exist_ok=True)

    layers, rows = {}, []
    for path in sorted(args.pose.glob("*.csv")):
        session_objects = objects[objects.session_id == path.stem]
        if session_objects.empty:
            print(f"  {path.stem}: sem objetos marcados, pulada")
            continue
        pose = pd.read_csv(path, encoding="utf-8-sig")
        layer = movement_layer(pose, frame_kinematics(pose, KEYPOINTS, args.fps),
                               session_objects, args.fps)
        layer.to_csv(args.output / f"{path.stem}.csv", index=False, encoding="utf-8-sig")
        layers[path.stem] = layer
        rows.append({"session_id": path.stem, **session_summary(layer, args.fps)})

    summary = pd.DataFrame(rows)
    if args.min_exploration is not None:
        summary["below_min_exploration"] = summary.exploration_total_s < args.min_exploration
    summary.to_csv(args.output / "resumo.csv", index=False, encoding="utf-8-sig")

    columns = ["session_id", "moving_pct", "freezing_pct", "distance_px",
               "exploration_1_s", "exploration_2_s", "exploration_total_s"]
    print(summary[columns].to_string(index=False))
    print(f"\n{len(summary)} sessões | por quadro e resumo em {args.output}")

    if args.min_exploration is not None:
        flagged = summary[summary.below_min_exploration]
        print(f"\nexploração total abaixo de {args.min_exploration:g} s: "
              f"{', '.join(flagged.session_id) or 'nenhuma'}")

    if args.labels.exists() and args.progress.exists():
        labels = pd.read_csv(args.labels, encoding="utf-8-sig")
        progress = pd.read_csv(args.progress, encoding="utf-8-sig")
        comparison = compare_exploration(layers, labels, progress, args.pose, objects, args.fps)
        if len(comparison):
            print("\nexploração: regra geométrica × marcação manual nas sessões completas\n")
            print(comparison.round(2).to_string(index=False))
            if len(comparison) > 2:
                r = np.corrcoef(comparison.manual_s, comparison["automático_s"])[0, 1]
                print(f"\ncorrelação entre os tempos por sessão: r = {r:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
