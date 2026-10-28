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
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.behavior_model import (  # noqa: E402
    BEHAVIORS,
    ContextModel,
    bouts,
    compare,
    frame_mask,
    gate,
    label_mask,
    make_model,
    matched_episodes,
    session_features,
    targets,
)
from mindtrace_ml.agreement import total_error  # noqa: E402
from mindtrace_ml.labels import seen_frames  # noqa: E402
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


def fit_model(sessions: dict, names, behavior: str, context: bool = True):
    """O detector treinado nas sessões dadas: dois estágios (ContextModel) ou um só."""
    names = list(names)
    if context:
        return ContextModel().fit_sessions([sessions[s]["X"] for s in names],
                                           [sessions[s]["y"][behavior].to_numpy() for s in names],
                                           [animal_of(s) for s in names],
                                           [sessions[s]["seen"] for s in names])
    X = pd.concat([sessions[s]["X"][sessions[s]["seen"]] for s in names])
    y = np.concatenate([sessions[s]["y"][behavior].to_numpy()[sessions[s]["seen"]] for s in names])
    return make_model().fit(X, y)


def fit_with_threshold(sessions: dict, names, behavior: str, fps: float,
                       target_recall: float | None = None, select_on=None, context: bool = True):
    """Treina nas sessões dadas e escolhe o limiar só com elas.

    O limiar sai de uma validação interna, deixando cada animal de treino de fora
    por vez: a sessão de teste não participa da escolha. `select_on` restringe as
    sessões que votam no limiar. As assistidas e as excluídas pelo critério do NOR
    treinam, mas não escolhem.

    Dois critérios. Sem `target_recall`, o F1 de episódios pareados um para um,
    que é o que conta rearings certo. Antes era o F1 por quadro, que mede tempo.
    Com `target_recall`, o limiar mais alto que ainda acha essa fração dos
    episódios: é o modo de uso em que o modelo propõe candidatos e o pesquisador
    descarta os falsos, e perder um rearing custa mais que revisar um alarme a mais.
    """
    select_on = list(names) if select_on is None else list(select_on)
    scores = {}
    for inner in names:
        rest = [s for s in names if animal_of(s) != animal_of(inner)]
        model = fit_model(sessions, rest, behavior, context)
        scores[inner] = model.predict_proba(sessions[inner]["X"])[:, 1]

    threshold = 0.5
    if target_recall is None:
        best = -1.0
        for candidate in np.round(np.arange(0.20, 0.81, 0.05), 2):
            pairs = manual = found = 0
            for inner in select_on:
                data = sessions[inner]
                mask = gate(behavior, frame_mask(scores[inner], data["centers"], len(data["frames"]),
                                                 candidate, fps), data["freezing"], fps) & data["visible"]
                marked, detected = bouts(data["marked"][behavior] & data["visible"]), bouts(mask)
                pairs += matched_episodes(marked, detected)
                manual += len(marked)
                found += len(detected)
            f1 = 2 * pairs / max(1, manual + found)
            if f1 > best:
                best, threshold = f1, float(candidate)
    else:
        for candidate in np.arange(0.95, 0.0, -0.01):
            hits = total = 0
            for inner in select_on:
                data = sessions[inner]
                seen = data["visible"]
                mask = gate(behavior, frame_mask(scores[inner], data["centers"], len(data["frames"]),
                                                 candidate, fps), data["freezing"], fps)
                for first, last in bouts(data["marked"][behavior] & seen):
                    total += 1
                    hits += mask[first:last + 1].any()
            threshold = float(candidate)
            if total and hits / total >= target_recall:
                break
    return fit_model(sessions, names, behavior, context), threshold


def progress_line(held: str, done: int, total: int) -> None:
    """Uma linha por sessão deixada de fora: o treino com contexto leva horas."""
    from datetime import datetime
    print(f"  {datetime.now():%H:%M} {held} pronta ({done}/{total})", flush=True)


# Sessões do treino, herdadas pelos processos filhos (fork) sem serializar.
SESSIONS: dict = {}


def held_out_scores(held: str, train: list, behavior: str, fps: float, target_recall, voters: set,
                    context: bool, threads: int | None = None):
    """Limiar e notas de uma sessão, com o modelo e a escolha feitos sem o animal dela."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(threads):
        model, threshold = fit_with_threshold(SESSIONS, train, behavior, fps, target_recall,
                                              select_on=[s for s in train if s in voters], context=context)
        return threshold, model.predict_proba(SESSIONS[held]["X"])[:, 1]


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
    parser.add_argument("--results", type=Path, default=None, help="grava os resultados por sessão (CSV)")
    parser.add_argument("--min-exploration", type=float, default=None,
                        help="tira do resumo, não do treino, as sessões do NOR com menos segundos de exploração")
    parser.add_argument("--no-context", action="store_true",
                        help="um estágio só, como até o v6 (sem as notas das janelas vizinhas)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="sessões deixadas de fora treinadas em paralelo (processos)")
    parser.add_argument("--fixed-threshold", type=float, default=0.40,
                        help="mede também cada sessão deixada de fora com este limiar único, como no uso "
                             "(padrão 0,40, o do v7, escolhido antes destas sessões); negativo desliga")
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
        data["visible"] = seen_frames(progress[progress.session_id == session], len(data["frames"]))
        data["seen"] = data["visible"][data["centers"]]
        own = labels[labels.session_id == session]
        data["y"] = targets(own, data["frames"], data["centers"], args.fps)
        data["marked"] = {b: label_mask(own, b, data["frames"], args.fps) for b in BEHAVIORS}
        sessions[session] = data
        print(".", end="", flush=True)
    print()

    # O critério de exclusão do NOR: animal que mal explorou os objetos sai da
    # análise do laboratório, então sai também da medida e da escolha do limiar.
    # Continua no treino, onde ensina como é um rato imóvel que não está em rearing.
    excluded = set()
    if args.min_exploration is not None:
        movement = pd.read_csv(ROOT / "data" / "movement" / "resumo.csv", encoding="utf-8-sig").set_index("session_id")
        excluded = {s for s in labeled if s in set(objects.session_id) and s in movement.index
                    and movement.loc[s, "exploration_total_s"] < args.min_exploration}
    voters = {s for s in labeled if not assisted[s] and s not in excluded}

    # --- 1. sessão deixada de fora ------------------------------------------
    # Sessão rotulada só para um comportamento não diz nada sobre os outros; sessão
    # rotulada vendo as sugestões do modelo treina, mas não mede — quem viu a
    # sugestão tende a concordar com ela.
    context = not args.no_context
    SESSIONS.update(sessions)
    results = []
    for behavior in behaviors:
        pool = eligible[behavior]
        held_list = [s for s in pool if not assisted[s]]
        # Fora o animal, não só a sessão: as do modo de três classes são, de
        # propósito, segundas sessões de animais já rotulados às cegas.
        jobs = [(held, [s for s in pool if animal_of(s) != animal_of(held)]) for held in held_list]
        if args.jobs > 1:
            # Processos, e não threads: cada treino é pequeno e o HistGradientBoosting
            # mal ocupa 2 núcleos. O fork herda as sessões sem copiá-las.
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            threads = max(1, (os.cpu_count() or 2) // args.jobs)
            from concurrent.futures import as_completed
            with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("fork")) as pool_exec:
                futures = {pool_exec.submit(held_out_scores, held, train, behavior, args.fps, args.target_recall,
                                            voters, context, threads): held for held, train in jobs}
                done = {}
                for future in as_completed(futures):
                    done[futures[future]] = future.result()
                    progress_line(futures[future], len(done), len(jobs))
            outcomes = [done[held] for held in held_list]
        else:
            outcomes = []
            for held, train in jobs:
                outcomes.append(held_out_scores(held, train, behavior, args.fps, args.target_recall, voters, context))
                progress_line(held, len(outcomes), len(jobs))
        for held, (threshold, scores) in zip(held_list, outcomes):
            data = sessions[held]
            visible = data["visible"]
            predicted = gate(behavior, frame_mask(scores, data["centers"], len(data["frames"]),
                                                  threshold, args.fps), data["freezing"], args.fps)
            marked = data["marked"][behavior]
            results.append({"sessão": held, "comportamento": behavior, "método": "modelo",
                            "limiar": threshold,
                            **compare(marked[visible], predicted[visible], args.fps)})
            # O uso tem um limiar só para todas as sessões. O escolhido a cada animal
            # deixado de fora variou de 0,45 a 0,70 no v7, e com ele a contagem foi a
            # -8%; nas mesmas notas, com 0,40 para todos, ficou em -1%.
            if args.fixed_threshold >= 0:
                fixed = gate(behavior, frame_mask(scores, data["centers"], len(data["frames"]),
                                                  args.fixed_threshold, args.fps), data["freezing"], args.fps)
                results.append({"sessão": held, "comportamento": behavior,
                                "método": f"limiar fixo {args.fixed_threshold:.2f}", "limiar": args.fixed_threshold,
                                **compare(marked[visible], fixed[visible], args.fps)})
            if behavior == "grooming":
                rule = data["triaged"]["active_head"].to_numpy(bool)
                results.append({"sessão": held, "comportamento": behavior, "método": "cabeça ativa",
                                **compare(marked[visible], rule[visible], args.fps)})
    results = pd.DataFrame(results)
    if args.results:
        results.to_csv(args.results, index=False, encoding="utf-8-sig")

    print("\n1. sessão deixada de fora — cada linha foi detectada por um modelo que não a viu\n")
    if excluded:
        print(f"fora do resumo (exploração < {args.min_exploration:g} s): {', '.join(sorted(excluded))}\n")
    for behavior in behaviors:
        print(f"{behavior}:")
        table = results[results.comportamento == behavior]
        view = table.pivot_table(index="sessão", columns="método",
                                 values=["manual_s", "detected_s", "manual_bouts", "detected_bouts"])
        print(view.round(1).to_string())
        table = table[~table["sessão"].isin(excluded)]
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
            # Somando os episódios de todas as sessões: a média das razões dá a uma
            # sessão com 2 rearings o mesmo peso de uma com 30.
            hits = (group.bouts_found.fillna(0) * group.manual_bouts).sum()
            real = (group.bouts_right.fillna(0) * group.detected_bouts).sum()
            recall, precision = hits / max(1, group.manual_bouts.sum()), real / max(1, group.detected_bouts.sum())
            f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
            print(f"  {method}, somando as sessões: achou {recall:.0%} dos episódios, {precision:.0%} do que "
                  f"apontou era real, F1 de episódio {f1:.3f}")
            pairs, manual, found = group.matched_bouts.sum(), group.manual_bouts.sum(), group.detected_bouts.sum()
            print(f"  {method}, um para um (cada detecção vale por um rearing): achou {pairs / max(1, manual):.0%}, "
                  f"certos {pairs / max(1, found):.0%}, F1 {2 * pairs / max(1, manual + found):.3f}, "
                  f"contagem {100 * (found - manual) / max(1, manual):+.0f}%")
            if group.manual_s.sum() > 0 and len(group) > 2:
                r = np.corrcoef(group.manual_s, group.detected_s)[0, 1]
                print(f"  {method}: tempo total manual × detectado, r = {r:.2f}")
            # O desfecho que vai para o artigo é o total por sessão: quanto ele erra,
            # e se o erro é viés (sempre para o mesmo lado) ou espalhamento.
            for what, manual_col, detected_col, unit in (("contagem", "manual_bouts", "detected_bouts", "rearings"),
                                                          ("tempo", "manual_s", "detected_s", "s")):
                e = total_error(group[manual_col], group[detected_col], group["sessão"])
                worst = (f"; mais pesa: {e['worst_session']} ({e['worst_share']:.0%} do MSE)"
                         if "worst_session" in e else "")
                print(f"  {method}, {what} por sessão: RMSE {e['rmse']:.1f} {unit} (média marcada "
                      f"{e['manual_mean']:.1f}, erro relativo mediano {e['relative_median']:.0%}), "
                      f"viés {e['bias']:+.1f} = {e['bias_share']:.0%} do MSE{worst}")
        print()

    # --- 2. clipes de outros animais ----------------------------------------
    clips = load_clips()
    seen_animals = {animal_of(s) for s in labeled}
    clips = clips[~clips.session_id.map(animal_of).isin(seen_animals)].reset_index(drop=True)

    final, thresholds = {}, {}
    for behavior in behaviors:
        print(f"treinando o modelo final de {behavior} com todas as sessões", flush=True)
        final[behavior], thresholds[behavior] = fit_with_threshold(
            sessions, eligible[behavior], behavior, args.fps, args.target_recall,
            select_on=[s for s in eligible[behavior] if s in voters], context=context)

    if clips.empty:
        # Quando todo animal com clipe já tem sessão rotulada, não sobra teste aqui;
        # a medida fica com a sessão deixada de fora.
        print("2. nenhum animal com clipes fica fora das sessões rotuladas: etapa pulada")
    else:
        clip_evaluation(clips, behaviors, final, thresholds, args, objects)

    if args.save:
        import joblib
        args.save.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": final, "thresholds": thresholds, "target_recall": args.target_recall,
                     "context": context,
                     "features": list(sessions[labeled[0]]["X"].columns),
                     "sessions": labeled}, args.save)
        print(f"\nmodelos salvos em {args.save}")
    return 0


def clip_evaluation(clips, behaviors, final, thresholds, args, objects) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
