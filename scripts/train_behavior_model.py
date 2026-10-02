"""Treina e avalia os detectores de grooming e rearing da camada de pose.

Duas medidas, nenhuma com dado que o modelo viu:

1. **Sessão deixada de fora.** Cada sessão rotulada por completo é detectada por
   um modelo treinado nas outras. Compara duração total, número de episódios e
   concordância quadro a quadro com a marcação manual. Para grooming, a regra de
   cabeça ativa entra como linha de base.
2. **Clipes de outros animais.** O modelo treinado nas sessões é testado nos
   clipes rotulados de animais que não aparecem em nenhuma sessão: diz se o
   comportamento esteve presente, e a resposta é comparada com a sua.

Uso:
    python scripts/train_behavior_model.py --pose data/pose --objects data/objects.csv \\
        --save models/behavior_model.joblib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.behavior_model import (  # noqa: E402
    BEHAVIORS,
    bouts,
    compare,
    frame_mask,
    gate,
    label_mask,
    make_model,
    session_features,
    targets,
)
from mindtrace_ml.schema import SESSION_BEHAVIORS  # noqa: E402
from mindtrace_ml.triage_model import animal_of, label_sets  # noqa: E402

CLIP_ROUNDS = (
    ("r1", "data/clip_labels.csv", "data/clips_gabarito.csv"),
    ("r2", "data/clip_labels_r2.csv", "data/clips_r2_gabarito.csv"),
    ("teste", "data/clip_labels_teste.csv", "data/clips_teste_gabarito.csv"),
    ("modelo", "data/clip_labels_modelo.csv", "data/clips_modelo_gabarito.csv"),
)


VIDEO_FEATURES = ROOT / "data" / "video_features"


def prepare(pose_dir: Path, session: str, objects: pd.DataFrame, fps: float,
            video: bool = True) -> dict:
    pose = pd.read_csv(pose_dir / f"{session}.csv", encoding="utf-8-sig")
    pixels = None
    if video:
        cached = VIDEO_FEATURES / f"{session}.csv"
        if not cached.exists():
            raise FileNotFoundError(f"{cached}: rode extract_video_features.py (ou use --no-video)")
        pixels = pd.read_csv(cached, encoding="utf-8-sig")
    return session_features(pose, objects[objects.session_id == session], fps, pixels)


def load_clips() -> pd.DataFrame:
    parts = []
    for name, labels_path, key_path in CLIP_ROUNDS:
        labels = pd.read_csv(ROOT / labels_path, encoding="utf-8-sig")
        labels["names"] = label_sets(labels)
        key = pd.read_csv(ROOT / key_path, encoding="utf-8-sig")
        merged = key.merge(labels[["clip_id", "names"]], on="clip_id")
        merged["round"] = name
        parts.append(merged)
    clips = pd.concat(parts, ignore_index=True)
    # "other" da primeira rodada é grooming ou rearing sem dizer qual: não serve de
    # gabarito para nenhum dos dois.
    ambiguous = (clips["round"] == "r1") & clips.names.map(lambda n: "other" in n)
    unscorable = clips.names.map(lambda n: n == {"unscorable"})
    return clips[~ambiguous & ~unscorable].reset_index(drop=True)


def stack(sessions: dict, names, behavior: str):
    X = pd.concat([sessions[s]["X"][sessions[s]["seen"]] for s in names])
    y = np.concatenate([sessions[s]["y"][behavior].to_numpy()[sessions[s]["seen"]] for s in names])
    return X, y


def fit_with_threshold(sessions: dict, names, behavior: str, fps: float,
                       target_recall: float | None = None):
    """Treina nas sessões dadas e escolhe o limiar só com elas.

    O limiar sai de uma validação interna, deixando cada sessão de treino de fora
    por vez: a sessão de teste não participa da escolha.

    Dois critérios. Sem `target_recall`, o equilíbrio entre achar e acertar (F1),
    para medir tempo sem revisão. Com ele, o limiar mais alto que ainda acha essa
    fração dos **episódios** — o modo de uso em que o modelo propõe candidatos e
    o pesquisador descarta os falsos: perder um rearing custa mais que revisar um
    alarme a mais.
    """
    scores = {}
    for inner in names:
        rest = [s for s in names if s != inner]
        model = make_model().fit(*stack(sessions, rest, behavior))
        scores[inner] = model.predict_proba(sessions[inner]["X"])[:, 1]

    threshold = 0.5
    if target_recall is None:
        predicted_all, truth_all = [], []
        for inner in names:
            seen = sessions[inner]["seen"]
            predicted_all.append(scores[inner][seen])
            truth_all.append(sessions[inner]["y"][behavior].to_numpy()[seen])
        values, truth = np.concatenate(predicted_all), np.concatenate(truth_all)
        best = -1.0
        for candidate in np.arange(0.05, 0.96, 0.05):
            predicted = values >= candidate
            f1 = 2 * (predicted & truth).sum() / max(1, predicted.sum() + truth.sum())
            if f1 > best:
                best, threshold = f1, float(candidate)
    else:
        for candidate in np.arange(0.95, 0.0, -0.01):
            hits = total = 0
            for inner in names:
                data = sessions[inner]
                seen = data["frames"] <= data["watched"]
                mask = gate(behavior, frame_mask(scores[inner], data["centers"], len(data["frames"]),
                                                 candidate, fps), data["freezing"], fps)
                for first, last in bouts(data["marked"][behavior] & seen):
                    total += 1
                    hits += mask[first:last + 1].any()
            threshold = float(candidate)
            if total and hits / total >= target_recall:
                break
    return make_model().fit(*stack(sessions, names, behavior)), threshold


def presence(clips: pd.DataFrame, masks: dict) -> np.ndarray:
    out = np.zeros(len(clips), dtype=bool)
    for index, clip in enumerate(clips.itertuples()):
        frames, mask = masks[clip.session_id]
        inside = (frames >= clip.start_frame) & (frames <= clip.end_frame)
        out[index] = mask[inside].any()
    return out


def weighted_rates(clips, truth, predicted) -> tuple[float, float]:
    """Acerto e cobertura com o peso de cada faixa no vídeo (rodada representativa)."""
    tp = fp = positives = 0.0
    for _, index in clips.groupby("triage_label").indices.items():
        weight = clips.band_share.iloc[index[0]] / len(index)
        tp += weight * (truth[index] & predicted[index]).sum()
        fp += weight * (~truth[index] & predicted[index]).sum()
        positives += weight * truth[index].sum()
    return tp / positives if positives else np.nan, tp / (tp + fp) if tp + fp else np.nan


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "session_labels.csv")
    parser.add_argument("--progress", type=Path, default=ROOT / "data" / "session_progress.csv")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true",
                        help="só sinais de pose, sem os de pixel — para comparar")
    parser.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=BEHAVIORS)
    parser.add_argument("--target-recall", type=float, default=None,
                        help="fração dos episódios a achar (ex.: 0.95); sem ela, equilíbrio F1")
    args = parser.parse_args()
    behaviors = args.behaviors

    objects = pd.read_csv(args.objects, encoding="utf-8-sig")
    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    progress = pd.read_csv(args.progress, encoding="utf-8-sig")
    # O progresso, e não os rótulos, diz o que foi rotulado: sessão sem nenhum rearing
    # não tem linha em session_labels, mas foi assistida.
    labeled = sorted(progress.session_id.unique())

    # Linhas antigas do progresso não têm estas colunas: eram rotulagens completas e cegas.
    if "behaviors" not in progress:
        progress["behaviors"] = np.nan
    if "assisted" not in progress:
        progress["assisted"] = False
    progress["behaviors"] = progress.behaviors.fillna(";".join((*SESSION_BEHAVIORS, "unscorable")))
    progress["assisted"] = progress.assisted.fillna(False).astype(bool)
    covers = {s: set(";".join(g.behaviors).split(";")) for s, g in progress.groupby("session_id")}
    assisted = progress.groupby("session_id").assisted.any().to_dict()
    eligible = {b: [s for s in labeled if b in covers.get(s, set())] for b in behaviors}

    print(f"preparando {len(labeled)} sessões rotuladas ", end="", flush=True)
    sessions = {}
    for session in labeled:
        data = prepare(args.pose, session, objects, args.fps, video=not args.no_video)
        watched = int(progress[progress.session_id == session].watched_until.max())
        data["seen"] = data["centers"] <= watched
        own = labels[labels.session_id == session]
        data["y"] = targets(own, data["frames"], data["centers"], args.fps)
        data["marked"] = {b: label_mask(own, b, data["frames"], args.fps) for b in BEHAVIORS}
        data["watched"] = watched
        sessions[session] = data
        print(".", end="", flush=True)
    print()

    # --- 1. sessão deixada de fora ------------------------------------------
    # Sessão rotulada só para um comportamento não diz nada sobre os outros; sessão
    # rotulada vendo as sugestões do modelo treina, mas não mede — quem viu a
    # sugestão tende a concordar com ela.
    results = []
    for behavior in behaviors:
        pool = eligible[behavior]
        for held in [s for s in pool if not assisted[s]]:
            train = [s for s in pool if s != held]
            data = sessions[held]
            seen_frames = data["frames"] <= data["watched"]
            model, threshold = fit_with_threshold(sessions, train, behavior, args.fps, args.target_recall)
            scores = model.predict_proba(data["X"])[:, 1]
            predicted = gate(behavior, frame_mask(scores, data["centers"], len(data["frames"]),
                                                  threshold, args.fps), data["freezing"], args.fps)
            marked = data["marked"][behavior]
            results.append({"sessão": held, "comportamento": behavior, "método": "modelo",
                            "limiar": threshold,
                            **compare(marked[seen_frames], predicted[seen_frames], args.fps)})
            if behavior == "grooming":
                rule = data["triaged"]["active_head"].to_numpy(bool)
                results.append({"sessão": held, "comportamento": behavior, "método": "cabeça ativa",
                                **compare(marked[seen_frames], rule[seen_frames], args.fps)})
    results = pd.DataFrame(results)

    print("\n1. sessão deixada de fora — cada linha foi detectada por um modelo que não a viu\n")
    for behavior in behaviors:
        print(f"{behavior}:")
        table = results[results.comportamento == behavior]
        view = table.pivot_table(index="sessão", columns="método",
                                 values=["manual_s", "detected_s", "manual_bouts", "detected_bouts"])
        print(view.round(1).to_string())
        chosen = table[table["método"] == "modelo"].limiar
        print(f"\n  limiar escolhido na validação interna: {chosen.min():.2f}–{chosen.max():.2f}")
        print("  média entre sessões:")
        table = table.assign(false_bouts=table.detected_bouts * (1 - table.bouts_right.fillna(0)))
        summary = table.groupby("método")[["frames_found", "frames_right", "bouts_found", "bouts_right",
                                           "false_bouts"]].mean()
        summary.columns = ["quadros marcados achados", "quadros detectados certos",
                           "episódios achados", "episódios detectados reais", "alarmes falsos por sessão"]
        print(summary.round(2).to_string())
        for method, group in table.groupby("método"):
            if group.manual_s.sum() > 0 and len(group) > 2:
                r = np.corrcoef(group.manual_s, group.detected_s)[0, 1]
                print(f"  {method}: tempo total manual × detectado, r = {r:.2f}")
        print()

    # --- 2. clipes de outros animais ----------------------------------------
    clips = load_clips()
    seen_animals = {animal_of(s) for s in labeled}
    clips = clips[~clips.session_id.map(animal_of).isin(seen_animals)].reset_index(drop=True)

    final, thresholds = {}, {}
    for behavior in behaviors:
        final[behavior], thresholds[behavior] = fit_with_threshold(sessions, eligible[behavior], behavior,
                                                                   args.fps, args.target_recall)

    print(f"2. clipes de {clips.session_id.map(animal_of).nunique()} animais que não estão nas sessões "
          f"({len(clips)} clipes) ", end="", flush=True)
    masks = {behavior: {} for behavior in behaviors}
    head = {}
    for session in sorted(clips.session_id.unique()):
        data = prepare(args.pose, session, objects, args.fps, video=not args.no_video)
        n = len(data["frames"])
        for behavior in behaviors:
            scores = final[behavior].predict_proba(data["X"])[:, 1]
            mask = frame_mask(scores, data["centers"], n, thresholds[behavior], args.fps)
            masks[behavior][session] = (data["frames"], gate(behavior, mask, data["freezing"], args.fps))
        head[session] = (data["frames"], data["triaged"]["active_head"].to_numpy(bool))
        print(".", end="", flush=True)
    print("\n")

    representative = (clips["round"] == "modelo").to_numpy()
    print(f"{'':<24} {'achou (dos que tinham)':>23} {'acertou (dos que disse)':>24}")
    for behavior in behaviors:
        truth = clips.names.map(lambda n, b=behavior: b in n).to_numpy()
        candidates = [("modelo", presence(clips, masks[behavior]))]
        if behavior == "grooming":
            candidates.append(("cabeça ativa", presence(clips, head)))
        for method, predicted in candidates:
            found = (truth & predicted).sum() / truth.sum()
            right = (truth & predicted).sum() / max(1, predicted.sum())
            rep = clips[representative].reset_index(drop=True)
            w_found, w_right = weighted_rates(rep, truth[representative], predicted[representative])
            print(f"{behavior + ' — ' + method:<24} {found:>12.0%} ({int(truth.sum())} clipes)"
                  f" {right:>13.0%}      | rodada representativa: achou {w_found:.0%}, acertou {w_right:.0%}")

    if args.save:
        import joblib
        args.save.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": final, "thresholds": thresholds, "target_recall": args.target_recall,
                     "features": list(sessions[labeled[0]]["X"].columns),
                     "sessions": labeled}, args.save)
        print(f"\nmodelos salvos em {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
