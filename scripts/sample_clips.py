"""Sorteia clipes estratificados pelas faixas da triagem, para rotulagem às cegas.

A amostra precisa cobrir **todas** as faixas, não só a de revisão: taxa de escape
é comportamento interessante que caiu numa faixa de rotina, e rotular apenas o
que já foi para revisão o tornaria invisível por construção.

A saída não carrega o rótulo da triagem — ele fica num arquivo à parte, para que
a rotulagem seja cega. Ver o palpite da máquina leva o anotador a confirmar em
vez de julgar, e a medição vira circular.

Uso:
    python scripts/sample_clips.py --pose data/pose --objects data/objects.csv \
                                   --videos .../arenas --output data/clips.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.detectors import BEHAVIOR_ORDER, Thresholds, triage  # noqa: E402
from mindtrace_ml.kinematics import frame_kinematics  # noqa: E402
from mindtrace_ml.triage_model import prepare_session, score_session, surfacing_score  # noqa: E402

KEYPOINTS = ("nose", "ear_left", "ear_right", "neck", "body", "tail_base")
STRATA = (*BEHAVIOR_ORDER, "review")


def sample_from(mask: np.ndarray, frames: np.ndarray, clip_frames: int,
                count: int, rng, max_per_run: int = 1) -> list[int]:
    """Início de clipes inteiramente contidos num trecho da faixa.

    Exigir que o clipe todo caia dentro da faixa evita amostrar transições, que
    seriam ambíguas para o anotador e não dizem nada sobre a faixa em si.

    **Mas torna a amostra não representativa.** Só 5% das janelas de 2 s caem
    inteiras numa faixa; os outros 95% são justamente transições, e nelas 30%
    dos clipes têm algo notável. Ponderar esses clipes pelo tamanho da faixa
    tratou 5% do vídeo como se fosse o todo, e o escape das regras saiu 51%
    quando o sorteio do vídeo inteiro mediu 23%. Para medir, use `--model`, que
    ladrilha a sessão toda.

    Cada trecho contínuo é dividido em janelas que não se sobrepõem, e contribui
    com no máximo `max_per_run` delas. A versão anterior sorteava entre todos os
    inícios válidos: num congelamento de 10 s há ~200, e quatro sorteios caíam
    quase sempre encavalados — na primeira rodada, 125 clipes eram 81 trechos
    independentes, e os 25 de revisão vinham de apenas 3. O anotador via o mesmo
    episódio várias vezes, e a medida contava como amostras o que era uma só.
    """
    runs, start = [], None
    for index, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= clip_frames:
                runs.append((start, index))
            start = None

    starts = []
    for run_start, run_end in runs:
        length = run_end - run_start
        tiles = length // clip_frames
        # Deslocamento aleatório para não amostrar sempre o começo do trecho,
        # que é a borda de uma transição e tende a ser atípico.
        offset = int(rng.integers(0, length - tiles * clip_frames + 1))
        windows = [run_start + offset + k * clip_frames for k in range(tiles)]
        if len(windows) > max_per_run:
            windows = list(rng.choice(windows, size=max_per_run, replace=False))
        starts += windows

    if not starts:
        return []
    picks = rng.choice(len(starts), size=min(count, len(starts)), replace=False)
    return [int(frames[starts[p]]) for p in sorted(picks)]


def overlaps(windows, start: int, end: int, margin: int = 0) -> bool:
    return any(start <= taken_end + margin and end >= taken_start - margin
               for taken_start, taken_end in windows)


def sample_by_score(args, objects, taken, rng) -> pd.DataFrame | None:
    """Estratos por faixa de nota do modelo, em vez de faixa das regras.

    Cada faixa é o trecho de vídeo que passa a ser pulado entre dois pontos de
    operação — `faixa_70_80` é o que o modelo manda para revisão pulando 70% do
    vídeo e pula quando o corte sobe para 80%. Com o clipe sorteado por faixa, o
    escape sai medido em todos esses pontos de uma vez, e a escolha do corte fica
    para depois de ver o resultado.

    O vídeo é ladrilhado em janelas do tamanho do clipe, e a nota de cada uma é a
    mesma que decide se ela chega ao pesquisador (`surfacing_score`). A parcela de
    ladrilhos em cada faixa é o peso do estrato, gravado no gabarito.
    """
    import joblib

    bundle = joblib.load(args.model)
    # O clipe tem o tamanho da janela do modelo: é a unidade que ele julga.
    window = clip_frames = int(round(bundle["window_sec"] * args.fps))
    step = int(round(bundle["step_sec"] * args.fps))

    # Janelas vizinhas a um clipe já rotulado incluem parte dele nas features: o
    # modelo treinou ali. A margem mantém a rodada fora do alcance do treino.
    margin = window
    cuts = [float(c) for c in args.bands.split(",")]
    tiles, pooled = [], []
    for path in sorted(args.pose.glob("*.csv")):
        session_objects = objects[objects.session_id == path.stem]
        if session_objects.empty:
            continue
        pose = pd.read_csv(path, encoding="utf-8-sig")
        session = prepare_session(pose, session_objects, args.fps, window, step)
        if list(session["windows"].columns) != bundle["features"]:
            print("as features do modelo salvo não batem com as atuais: treine de novo")
            return None
        scores = score_session(bundle["model"], session)
        missing = session["unscorable"]
        pooled.append(np.where(missing, np.inf, scores))

        frames = session["frames"]
        offset = int(rng.integers(0, clip_frames))
        for first in range(offset, len(frames) - clip_frames + 1, clip_frames):
            last = first + clip_frames
            start, end = int(frames[first]), int(frames[last - 1])
            if overlaps(taken.get(path.stem, []), start, end, margin):
                continue
            score = surfacing_score(scores[first:last], missing[first:last], 0.10)
            if np.isfinite(score):
                tiles.append((path.stem, start, end, score))

    # Corte de cada ponto de operação: a nota acima da qual fica a fração de
    # vídeo que vai para revisão. Pose ausente conta como revisada, como na triagem.
    thresholds = np.quantile(np.concatenate(pooled), cuts)
    edges = [0.0, *cuts, 1.0]
    names = [f"faixa_{round(a * 100):02d}_{round(b * 100)}" for a, b in zip(edges, edges[1:])]

    table = pd.DataFrame(tiles, columns=["session_id", "start_frame", "end_frame", "score"])
    table["band"] = np.searchsorted(thresholds, table.score.to_numpy(), side="right")
    shares = table.band.value_counts(normalize=True)

    rows = []
    print(f"\n{'faixa':<16} {'% do vídeo':>11} {'ladrilhos':>10} {'sorteados':>10}")
    print("-" * 50)
    for band, name in enumerate(names):
        pool = table[table.band == band]
        chosen = pool.sample(n=min(args.per_stratum, len(pool)), random_state=args.seed)
        for clip in chosen.itertuples():
            rows.append({"clip_id": f"{clip.session_id}_{clip.start_frame}",
                         "session_id": clip.session_id,
                         "start_frame": clip.start_frame, "end_frame": clip.end_frame,
                         "triage_label": name, "score": round(clip.score, 4),
                         "band_share": round(float(shares.get(band, 0.0)), 4)})
        print(f"{name:<16} {shares.get(band, 0.0):>11.0%} {len(pool):>10} {len(chosen):>10}")
    return pd.DataFrame(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip-sec", type=float, default=3.0)
    # 25 por faixa dá 100 clipes de rotina, que é onde a estimativa de escape
    # deixa de ser ruído: em torno de 5%, a margem cai de ±6 para ±4 pontos.
    # Escape só é observável nos clipes classificados como rotina, então é esse
    # número — não o total — que governa a precisão.
    parser.add_argument("--per-stratum", type=int, default=25)
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--strata", default=None,
                        help="ex.: low_activity:50,walking:30 — padrão: todos, --per-stratum cada")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[],
                        help="CSVs de clipes já sorteados, para não repeti-los")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", type=Path, default=None,
                        help="modelo de train_triage_model: estratos por faixa de nota, clipe do tamanho da janela")
    parser.add_argument("--bands", default="0.6,0.7,0.8,0.9",
                        help="frações de vídeo pulado que delimitam as faixas de nota")
    args = parser.parse_args()

    objects = pd.read_csv(args.objects, encoding="utf-8-sig")
    # Limiares padrão do módulo: o sorteio precisa usar os mesmos que a triagem
    # avaliada, senão o estrato do clipe não corresponde ao que se mede.
    thresholds = Thresholds()

    quotas = {stratum: args.per_stratum for stratum in STRATA}
    if args.strata:
        quotas = {}
        for item in args.strata.split(","):
            name, count = item.split(":")
            if name not in STRATA:
                print(f"estrato desconhecido: {name}. Válidos: {', '.join(STRATA)}")
                return 1
            quotas[name] = int(count)

    # Janelas já rotuladas, por sessão, para excluir sobreposição e não só
    # clip_id idêntico — um clipe deslocado de poucos quadros seria o mesmo trecho.
    taken = {}
    for path in args.exclude:
        previous = pd.read_csv(path, encoding="utf-8-sig")
        for row in previous.itertuples():
            taken.setdefault(row.session_id, []).append((row.start_frame, row.end_frame))
    clip_frames = int(round(args.clip_sec * args.fps))
    rng = np.random.default_rng(args.seed)

    sessions = sorted(args.pose.glob("*.csv"))
    if not sessions:
        print(f"nenhum CSV de pose em {args.pose}")
        return 1

    if args.model:
        clips = sample_by_score(args, objects, taken, rng)
        if clips is None:
            return 1
        return write_round(clips, args, clips.end_frame.iloc[0] - clips.start_frame.iloc[0] + 1)

    # Junta todos os candidatos por faixa e só depois sorteia, para que o número
    # de clipes por faixa não dependa de quantas sessões cada uma domina.
    candidates = {stratum: [] for stratum in quotas}

    for path in sessions:
        session = path.stem
        pose = pd.read_csv(path, encoding="utf-8-sig")
        session_objects = objects[objects.session_id == session]
        if session_objects.empty:
            print(f"  {session}: sem objetos marcados, pulado")
            continue

        kinematics = frame_kinematics(pose, KEYPOINTS, args.fps, min_confidence=0.25)
        triaged = triage(pose, kinematics, session_objects, args.fps, thresholds)
        frames = triaged["frame"].to_numpy()

        for stratum, quota in quotas.items():
            starts = sample_from(triaged[stratum].to_numpy(bool), frames,
                                 clip_frames, quota * 3, rng)
            candidates[stratum] += [(session, start) for start in starts]

    # A seleção final percorre cada estrato em ordem aleatória e recusa janelas
    # que cruzem qualquer outra já escolhida — inclusive de outro estrato. A
    # triagem é multi-etiqueta, então o mesmo trecho pode ser candidato em duas
    # faixas, e sem isso apareceria duas vezes na rodada.
    rows = []
    print(f"\n{'faixa':<20} {'candidatos':>11} {'sorteados':>10}")
    print("-" * 44)
    for stratum, quota in quotas.items():
        pool = candidates[stratum]
        chosen = 0
        for p in (rng.permutation(len(pool)) if pool else []):
            if chosen >= quota:
                break
            session, start = pool[p]
            end = start + clip_frames - 1
            if overlaps(taken.get(session, []), start, end):
                continue
            taken.setdefault(session, []).append((start, end))
            chosen += 1
            rows.append({
                "clip_id": f"{session}_{start}",
                "session_id": session,
                "start_frame": start,
                "end_frame": start + clip_frames - 1,
                "triage_label": stratum,
            })
        print(f"{stratum:<20} {len(pool):>11} {chosen:>10}")

    return write_round(pd.DataFrame(rows), args, clip_frames)


def write_round(clips: pd.DataFrame, args, clip_frames: int) -> int:
    clips = clips.sample(frac=1.0, random_state=args.seed, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Ordem embaralhada e só com o necessário para achar o trecho: o arquivo que o
    # anotador abre não pode revelar o que a triagem achou.
    clips[["clip_id", "session_id", "start_frame", "end_frame"]].to_csv(
        args.output, index=False, encoding="utf-8-sig")

    key = args.output.with_name(args.output.stem + "_gabarito.csv")
    clips.to_csv(key, index=False, encoding="utf-8-sig")

    print(f"\n{len(clips)} clipes de {clip_frames / args.fps:.1f}s em {args.output}")
    print(f"gabarito (NÃO abrir antes de rotular): {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
