"""Treina e avalia o modelo que decide o que vai para revisão.

Validação cruzada agrupada por animal: cada clipe recebe a nota de um modelo que
nunca viu aquele animal. Com essas notas fora da amostra, o script mede nos
mesmos clipes o que evaluate_triage mede para as regras:

- o escape no ponto em que o modelo pula **a mesma fração de vídeo** que as
  regras. É a única comparação justa: qualquer método escapa menos se mandar
  mais vídeo para revisão;
- a curva escape × redução, variando o limiar do modelo — o ponto de operação é
  decisão do laboratório;
- a curva de aprendizado, que diz se rotular mais clipes deve ajudar.

A comparação principal é na rodada de teste. As regras foram calibradas nas
rodadas 1 e 2 e ali levam vantagem; na de teste nenhum dos dois viu os clipes.

**Os números daqui são otimistas.** As rodadas 1, 2 e de teste só sortearam
clipes inteiros dentro de uma faixa das regras, e esses trechos "puros" são 5%
do vídeo. A validação disse ~12% de escape; a rodada sorteada do vídeo inteiro
(`evaluate_model_round.py`) mediu 25–34%, empatado com as regras no mesmo tempo
assistido. Use este script para comparar variantes entre si, não para estimar o
desempenho real.

Uso:
    python scripts/train_triage_model.py --pose data/pose --objects data/objects.csv \\
        --save models/triage_model.joblib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindtrace_ml.triage_model import (  # noqa: E402
    animal_of,
    clip_features,
    label_sets,
    make_model,
    notable_set,
    prepare_session,
    score_session,
    surfacing_score,
)

ROUNDS = (
    ("r1", "data/clip_labels.csv", "data/clips_gabarito.csv"),
    ("r2", "data/clip_labels_r2.csv", "data/clips_r2_gabarito.csv"),
    ("teste", "data/clip_labels_teste.csv", "data/clips_teste_gabarito.csv"),
)
TEST_ROUND = "teste"


def load_clips(notable: frozenset) -> pd.DataFrame:
    parts = []
    for name, labels_path, key_path in ROUNDS:
        labels = pd.read_csv(ROOT / labels_path, encoding="utf-8-sig")
        labels["names"] = label_sets(labels)
        key = pd.read_csv(ROOT / key_path, encoding="utf-8-sig")
        merged = key.merge(labels[["clip_id", "names"]], on="clip_id")
        merged["round"] = name
        parts.append(merged)
    clips = pd.concat(parts, ignore_index=True)
    clips = clips[clips.names.map(lambda n: n != {"unscorable"})]
    clips["notable"] = clips.names.map(lambda n: bool(n & notable))
    clips["animal"] = clips.session_id.map(animal_of)
    return clips.reset_index(drop=True)


def prepare_sessions(pose_dir: Path, objects: pd.DataFrame, fps: float,
                     window: int, step: int) -> dict:
    sessions = {}
    for path in sorted(pose_dir.glob("*.csv")):
        session_objects = objects[objects.session_id == path.stem]
        if session_objects.empty:
            continue
        pose = pd.read_csv(path, encoding="utf-8-sig")
        sessions[path.stem] = prepare_session(pose, session_objects, fps, window, step)
        sessions[path.stem]["rules"] = sessions[path.stem]["triaged"]["review"].to_numpy(float)
        print(".", end="", flush=True)
    print()
    return sessions


def cross_validate(clips, X, sessions, folds: int, seed: int,
                   fraction: float = 1.0, score_frames: bool = True):
    """Notas fora da amostra, por clipe e por quadro de cada sessão."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = clips.notable.to_numpy()
    groups = clips.animal.to_numpy()
    oof = np.full(len(clips), np.nan)
    scores = {}
    rng = np.random.default_rng(seed)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)

    model = None
    for train, test in splitter.split(X, y, groups):
        if fraction < 1.0:
            train = rng.choice(train, size=int(round(len(train) * fraction)), replace=False)
        model = make_model(seed).fit(X.iloc[train], y[train])
        oof[test] = model.predict_proba(X.iloc[test])[:, 1]
        if score_frames:
            held_out = set(groups[test])
            for name, data in sessions.items():
                if animal_of(name) in held_out:
                    scores[name] = score_session(model, data)

    if score_frames:
        # Animal sem nenhum clipe rotulado: nenhum modelo o viu, qualquer um serve.
        for name, data in sessions.items():
            if name not in scores:
                scores[name] = score_session(model, data)
    return oof, scores


def surfacing_scores(clips, sessions, scores, min_presence: float) -> np.ndarray:
    """Maior limiar em que cada clipe ainda chega ao pesquisador."""
    out = np.empty(len(clips))
    for index, clip in enumerate(clips.itertuples()):
        data = sessions[clip.session_id]
        first = np.searchsorted(data["frames"], clip.start_frame)
        last = np.searchsorted(data["frames"], clip.end_frame, side="right")
        out[index] = surfacing_score(scores[clip.session_id][first:last],
                                     data["unscorable"][first:last], min_presence)
    return out


def reduction_curve(sessions, scores, grid: np.ndarray) -> np.ndarray:
    """Fração média de vídeo que não vai para revisão, para cada limiar."""
    total = np.zeros(len(grid))
    for name, data in sessions.items():
        valid = np.sort(scores[name][~data["unscorable"]])
        reviewed = len(valid) - np.searchsorted(valid, grid, side="left")
        total += 1 - reviewed / len(data["unscorable"])
    return total / len(sessions)


def stratum_weights(sessions, strata) -> dict:
    """Parcela de quadros de cada estrato, média entre sessões — igual a evaluate_triage."""
    shares = {s: np.mean([d["triaged"][s].mean() for d in sessions.values()]) for s in strata}
    total = sum(shares.values())
    return {s: v / total for s, v in shares.items()}


def weighted_escape(clips, surfaced, weights) -> float:
    notable_share = escaped_share = 0.0
    for stratum, group in clips.groupby("triage_label"):
        weight = weights.get(stratum, 0.0)
        notable_share += weight * group.notable.mean()
        escaped_share += weight * (group.notable & ~surfaced[group.index]).mean()
    return escaped_share / notable_share if notable_share else float("nan")


def summarize(clips, surfaced, weights, kinds) -> dict:
    notable = clips.notable.to_numpy()
    test = clips["round"] == TEST_ROUND
    row = {
        "escape_all": float((~surfaced[notable]).mean()),
        "escape_test": float((~surfaced[notable & test.to_numpy()]).mean()),
        "escape_test_weighted": weighted_escape(clips[test], pd.Series(surfaced, clips.index), weights),
    }
    for kind in kinds:
        has = clips.names.map(lambda n, k=kind: k in n).to_numpy()
        row[f"seen_{kind}"] = float(surfaced[has].mean()) if has.any() else float("nan")
    return row


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--window-sec", type=float, default=2.0,
                        help="janela deslizante na aplicação — o tamanho dos clipes da rodada de teste")
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--min-presence", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5,
                        help="repetições com outra divisão de animais; a variação entre elas é o ruído")
    parser.add_argument("--sniffing-is-notable", action="store_true")
    parser.add_argument("--save", type=Path, default=None,
                        help="treina com todos os clipes e salva o modelo")
    args = parser.parse_args()

    notable = notable_set(args.sniffing_is_notable)
    clips = load_clips(notable)
    objects = pd.read_csv(args.objects, encoding="utf-8-sig")

    window = int(round(args.window_sec * args.fps))
    step = int(round(args.step_sec * args.fps))
    print(f"preparando {len(list(args.pose.glob('*.csv')))} sessões ", end="")
    sessions = prepare_sessions(args.pose, objects, args.fps, window, step)
    clips = clips[clips.session_id.isin(sessions)].reset_index(drop=True)

    parts = []
    for session, group in clips.groupby("session_id", sort=False):
        part = clip_features(sessions[session]["signals"], zip(group.start_frame, group.end_frame))
        part.index = group.index
        parts.append(part)
    X = pd.concat(parts).loc[clips.index]

    test = (clips["round"] == TEST_ROUND).to_numpy()
    print(f"{len(clips)} clipes de {clips.animal.nunique()} animais, "
          f"{int(clips.notable.sum())} com algo notável "
          f"(sniffing conta como {'outros' if args.sniffing_is_notable else 'rotina'})")
    print(f"{X.shape[1]} features por janela | janela {args.window_sec}s, passo {args.step_sec}s\n")

    from sklearn.metrics import roc_auc_score

    kinds = sorted(notable)
    weights = stratum_weights(sessions, clips.loc[test, "triage_label"].unique())

    rules_scores = {name: data["rules"] for name, data in sessions.items()}
    rules_surfaced = surfacing_scores(clips, sessions, rules_scores, args.min_presence) >= 0.5
    rules_reduction = float(reduction_curve(sessions, rules_scores, np.array([0.5]))[0])
    rules = summarize(clips, rules_surfaced, weights, kinds)

    targets = sorted({0.60, 0.70, round(rules_reduction, 3), 0.85, 0.90})
    matched, curve, aucs, chosen = [], [], [], []
    for repeat in range(args.repeats):
        oof, scores = cross_validate(clips, X, sessions, args.folds, seed=repeat)
        aucs.append((roc_auc_score(clips.notable, oof),
                     roc_auc_score(clips.notable[test], oof[test])))

        pooled = np.concatenate([s[~sessions[n]["unscorable"]] for n, s in scores.items()])
        grid = np.unique(np.quantile(pooled, np.linspace(0, 1, 2001)))
        reductions = reduction_curve(sessions, scores, grid)
        surfacing = surfacing_scores(clips, sessions, scores, args.min_presence)

        for target in targets:
            at = int(np.argmin(np.abs(reductions - target)))
            row = summarize(clips, surfacing >= grid[at], weights, kinds)
            curve.append({"target": target, "reduction": reductions[at], **row})
            if target == round(rules_reduction, 3):
                matched.append({"reduction": reductions[at], **row})
                chosen.append(grid[at])
        print(f"  repetição {repeat + 1}/{args.repeats}", flush=True)

    matched = pd.DataFrame(matched)
    aucs = np.array(aucs)

    def span(values, pct=True):
        mean, low, high = np.mean(values), np.min(values), np.max(values)
        return (f"{mean:.0%} ({low:.0%}–{high:.0%})" if pct
                else f"{mean:.2f} ({low:.2f}–{high:.2f})")

    print(f"\nAUC fora da amostra: {span(aucs[:, 0], False)} em todos os clipes, "
          f"{span(aucs[:, 1], False)} na rodada de teste")
    print(f"(entre parênteses, o intervalo entre as {args.repeats} divisões de animais)\n")

    print(f"mesma redução das regras ({rules_reduction:.1%} do vídeo pulado):\n")
    print(f"{'':<34} {'regras':>8} {'modelo':>22}")
    print("-" * 66)
    lines = [("escape ponderado, rodada de teste", "escape_test_weighted"),
             ("escape simples, rodada de teste", "escape_test"),
             ("escape simples, todas as rodadas", "escape_all")]
    lines += [(f"{kind} que chega ao pesquisador", f"seen_{kind}") for kind in kinds]
    for label, key in lines:
        print(f"{label:<34} {rules[key]:>8.0%} {span(matched[key]):>22}")

    curve = pd.DataFrame(curve).groupby("target").mean(numeric_only=True)
    print("\ncurva do modelo — quanto mais vídeo pulado, mais escapa:\n")
    print(f"{'vídeo pulado':>13} {'escape ponderado (teste)':>26} {'escape (todas)':>16}")
    for _, row in curve.iterrows():
        print(f"{row.reduction:>13.0%} {row.escape_test_weighted:>26.0%} {row.escape_all:>16.0%}")

    print("\ncurva de aprendizado — AUC com parte dos clipes de treino:\n")
    for fraction in (0.25, 0.5, 0.75, 1.0):
        values = [roc_auc_score(clips.notable, cross_validate(
            clips, X, sessions, args.folds, seed=r, fraction=fraction, score_frames=False)[0])
            for r in range(args.repeats)]
        print(f"  {fraction:>4.0%} dos clipes ({int(fraction * len(clips) * (args.folds - 1) / args.folds):>3})"
              f"  AUC {np.mean(values):.3f}")

    if args.save:
        import joblib
        final = make_model(0).fit(X, clips.notable.to_numpy())
        args.save.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": final,
            "features": list(X.columns),
            "window_sec": args.window_sec,
            "step_sec": args.step_sec,
            # Limiar que, na validação cruzada, pula a mesma fração de vídeo que as regras.
            "threshold": float(np.mean(chosen)),
            "sniffing_is_notable": args.sniffing_is_notable,
            "n_clips": len(clips),
        }, args.save)
        print(f"\nmodelo treinado com {len(clips)} clipes salvo em {args.save} "
              f"(limiar {np.mean(chosen):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
