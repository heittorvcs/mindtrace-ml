"""Camada de pose: detectores de grooming e rearing, um por comportamento.

Complementa a camada de movimento. Aquela diz onde o rato está e se ele se move;
esta diz o que ele está fazendo — e os dois podem acontecer juntos: um rearing
no meio de uma caminhada é movendo *e* rearing.

Diferente do modelo da triagem, este aprende das **sessões rotuladas por
completo**, com início e fim de cada comportamento. A pergunta muda de "há algo
neste clipe?" para "o rato está fazendo isto *agora*?", que é o que dá duração e
contagem de episódios — os números que o laboratório reporta.

Cada quadro é descrito por duas janelas centradas nele: uma curta (0,5 s), que
localiza o comportamento no tempo, e uma longa (2 s), que dá o contexto. Só a
longa diria que houve grooming por perto, sem dizer quando começou; só a curta
confundiria um puxão de cabeça com um grooming.

Sniffing ficou de fora por decisão do laboratório: difícil de distinguir, sutil,
muitas vezes misturado a outros comportamentos, e de pouco valor para o NOR.
"""

import numpy as np
import pandas as pd

from .arena import floor_distance
from .detectors import Thresholds, _sustained, triage
from .kinematics import frame_kinematics
from .labels import bouts_for, intervals_to_mask
from .schema import CONFIDENCE_THRESHOLD
from .triage_model import KEYPOINTS, SPEC, describe, frame_signals
from .video_features import pose_rise_signals

BEHAVIORS = ("grooming", "rearing")

# Sinais além dos da triagem: pixels do vídeo e a geometria do levantar
# (video_features.py). Entram só quando o cache de vídeo existe.
EXTRA_SPEC = {
    "motion_head": ("median", "p90"),
    "motion_body": ("median", "p90"),
    "motion_ratio": ("median", "p90"),
    "blob_area": ("median", "p10", "p90"),
    "blob_height": ("median", "p10", "p90"),
    "blob_width": ("median", "p10", "p90"),
    "blob_aspect": ("median", "p10", "p90"),
    "blob_dy": ("median", "p10", "p90"),
    "neck_rel_y": ("median", "p10", "p90"),
    "nose_rel_y": ("median", "p10", "p90"),
    "length_local": ("median", "p10", "std"),
    # Há quanto tempo o rato está parado e encostado na parede (dwell_signals).
    # Saem da pose, então entram sempre.
    "still_sec": ("median", "p10", "p90"),
    "wall_sec": ("median", "p10", "p90"),
    # Focinho até a borda do chão marcado: negativo quando a cabeça aparece sobre a
    # parede, o rato em pé na câmera oblíqua. Só com o chão (session_features).
    "nose_floor": ("median", "p10", "p90"),
}
# Quadros: ~0,5 s, 2 s, 4 s e 10 s a 30 fps. A de 10 s separa o rearing, que sobe
# e desce em 1 a 3 s, do rato que fica encostado na parede: para as janelas curtas
# os dois são iguais. Deixando um animal de fora, em 23 sessões, ela levou o F1
# um para um de 0,751 para 0,777 e, com dwell_signals, para 0,784. Os alarmes
# falsos foram de 3,5 para 2,4 por sessão: na TT_28, de 30 para 16 detecções
# (4 marcadas); na TT_26, rato imóvel, de 21 para 2.
SCALES = (15, 60, 120, 300)
STEP = 8                 # uma janela a cada ~0,27 s
MIN_BOUT_SEC = 0.3

# Quanto tempo o rato pode tocar o chão sem que o rearing termine. É o critério
# de quem conta os rearings no laboratório: nas ratas F6 e F2 da habituação, o
# anotador marcou cada descida (20 e 54 rearings em 5 min) e a planilha tinha 15
# e 37; juntando as descidas de menos de 0,5 s, os rótulos dão 16 e 36. Descartar
# os rearings curtos não explicava as duas ratas com um mesmo corte.
MERGE_GAP_SEC = {"rearing": 0.5}


# Sinais em que importa a direção da mudança dentro da janela, não só o nível.
# Um rearing tem forma — sobe, fica, desce —, e o resumo por mediana e quantis
# não distingue o corpo encurtando do corpo voltando ao normal.
TREND_SIGNALS = ("length_local", "blob_height", "blob_width", "blob_aspect", "blob_area",
                 "blob_dy", "neck_rel_y", "nose_rel_y", "full_length", "body_length",
                 "conf_nose", "motion_body", "body_speed", "wall_distance", "nose_floor")


def behavior_spec(signals: pd.DataFrame) -> dict:
    """Sinais da triagem, mais os extras presentes, com tendência onde ela importa."""
    spec = {**SPEC, **{k: v for k, v in EXTRA_SPEC.items() if k in signals}}
    return {k: (*v, "trend") if k in TREND_SIGNALS else v for k, v in spec.items()}


def centered_features(signals: pd.DataFrame, centers: np.ndarray, scales=SCALES,
                      spec: dict | None = None) -> pd.DataFrame:
    """Features de janelas centradas em cada quadro de `centers` (índices de linha).

    A janela de 4 s dá o contexto do episódio inteiro: um rearing curto é uma
    subida e uma descida dentro dela, que a janela de 2 s mal abrange.
    """
    spec = spec or behavior_spec(signals)
    matrix = signals[list(spec)].to_numpy(float)
    parts = []
    for size in scales:
        half = size // 2
        # Bordas completadas com NaN: a janela perto do início do vídeo simplesmente
        # tem menos quadros, como qualquer trecho sem pose.
        padded = np.vstack([np.full((half, matrix.shape[1]), np.nan), matrix,
                            np.full((size, matrix.shape[1]), np.nan)])
        index = centers[:, None] + np.arange(size)[None, :]
        block = describe(padded[index], spec)
        block.columns = [f"w{size}_{c}" for c in block.columns]
        parts.append(block)
    return pd.concat(parts, axis=1)


def session_features(pose: pd.DataFrame, objects: pd.DataFrame, fps: float,
                     pixels: pd.DataFrame | None = None, floor: np.ndarray | None = None) -> dict:
    """Tudo o que os detectores precisam de uma sessão.

    `pixels` é o cache de extract_video_features; sem ele, só sinais de pose.

    `floor` é o chão da arena (arena.floor_for). A distância à parede passa a ser
    a do pescoço até a borda dele, e entra o focinho até a borda. Sem ele, a parede
    é o retângulo dos percentis do dorso, como até o v8. Deixando um animal de
    fora, em 23 sessões, com dois estágios e limiar 0,40, o chão levou o F1 um
    para um de 0,765 para 0,811, a precisão de 76% para 81% e o erro de contagem
    por sessão de 41% para 22% (models/experimentos/floor3.py). Na TT_28, rato
    parado na parede, foram de 15 para 9 detecções (4 marcadas).
    """
    kinematics = frame_kinematics(pose, KEYPOINTS, fps)
    triaged = triage(pose, kinematics, objects, fps, Thresholds())
    signals = frame_signals(pose, kinematics, triaged, objects, fps)
    if pixels is not None:
        signals = pd.concat([signals, pixels.drop(columns="frame", errors="ignore").reset_index(drop=True),
                             pose_rise_signals(pose, kinematics, fps)], axis=1)
    if floor is not None:
        signals = with_floor(signals, pose, floor)
    signals = pd.concat([signals, dwell_signals(signals, fps)], axis=1)
    centers = window_centers(len(pose))
    return {"frames": pose["frame"].to_numpy(), "triaged": triaged, "centers": centers,
            "freezing": triaged["freezing"].to_numpy(bool),
            "X": centered_features(signals, centers)}


def with_floor(signals: pd.DataFrame, pose: pd.DataFrame, floor: np.ndarray) -> pd.DataFrame:
    """Parede medida até a borda do chão dado, mais o focinho até ela."""
    def point(name):
        ok = pose[f"{name}_p"].to_numpy() >= CONFIDENCE_THRESHOLD
        return np.where(ok, pose[f"{name}_x"], np.nan), np.where(ok, pose[f"{name}_y"], np.nan)

    out = signals.copy()
    out["wall_distance"] = floor_distance(*point("neck"), floor)
    out["nose_floor"] = floor_distance(*point("nose"), floor)
    return out


DWELL_CAP_SEC = 60.0
WALL_PX = 25.0


def run_seconds(flag: np.ndarray, fps: float, cap: float = DWELL_CAP_SEC) -> np.ndarray:
    """Segundos seguidos em que `flag` vale, até cada quadro; 0 onde não vale."""
    out = np.zeros(len(flag))
    run = 0
    for index, value in enumerate(flag):
        run = run + 1 if value else 0
        out[index] = run
    return np.minimum(out / fps, cap)


def dwell_signals(signals: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Há quanto tempo o rato está parado e há quanto tempo está encostado na parede.

    Parado é abaixo da velocidade de caminhada da camada de movimento, com a
    velocidade suavizada em 0,5 s. Quadro sem pose não interrompe a contagem: o
    rato não sumiu andando.
    """
    speed = signals["body_speed"].rolling(int(round(0.5 * fps)), center=True, min_periods=1).median()
    moving = np.nan_to_num(speed.to_numpy(), nan=0.0) > Thresholds().walking_speed
    at_wall = np.nan_to_num(signals["wall_distance"].to_numpy(), nan=np.inf) < WALL_PX
    return pd.DataFrame({"still_sec": run_seconds(~moving, fps), "wall_sec": run_seconds(at_wall, fps)})


def detect(bundle: dict, behavior: str, session: dict, fps: float) -> np.ndarray:
    """Episódios por quadro de um comportamento, com o modelo e o limiar salvos."""
    scores = bundle["models"][behavior].predict_proba(session["X"][bundle["features"]])[:, 1]
    mask = frame_mask(scores, session["centers"], len(session["frames"]),
                      bundle["thresholds"][behavior], fps, low=bundle.get("low_thresholds", {}).get(behavior))
    return gate(behavior, mask, session["freezing"], fps)


def window_centers(n_frames: int, step: int = STEP) -> np.ndarray:
    return np.arange(0, n_frames, step)


def merge_gap(behavior: str, fps: float) -> int:
    """Descida máxima, em quadros, que não encerra um episódio do comportamento."""
    return int(round(MERGE_GAP_SEC.get(behavior, 0.0) * fps))


def label_mask(labels: pd.DataFrame, behavior: str, frames: np.ndarray, fps: float) -> np.ndarray:
    """Quadros marcados, com a mesma definição de episódio usada na detecção."""
    return intervals_to_mask(bouts_for(labels, behavior, gap=merge_gap(behavior, fps)), frames)


def close_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """Preenche as interrupções de até `gap` quadros entre dois trechos marcados."""
    if gap <= 0 or not mask.any():
        return mask
    out = mask.copy()
    found = bouts(mask)
    for (_, previous_end), (start, _) in zip(found, found[1:]):
        if start - previous_end <= gap:
            out[previous_end:start] = True
    return out


def targets(labels: pd.DataFrame, frames: np.ndarray, centers: np.ndarray,
            fps: float = 29.97) -> pd.DataFrame:
    """Para cada centro, se cada comportamento estava marcado naquele quadro."""
    return pd.DataFrame({behavior: label_mask(labels, behavior, frames, fps)[centers]
                         for behavior in BEHAVIORS})


def frame_scores(window_scores: np.ndarray, centers: np.ndarray, n_frames: int) -> np.ndarray:
    """Nota de cada quadro: a da janela de centro mais próximo."""
    nearest = np.clip(np.rint(np.arange(n_frames) / (centers[1] - centers[0])).astype(int),
                      0, len(centers) - 1) if len(centers) > 1 else np.zeros(n_frames, int)
    return window_scores[nearest]


def frame_mask(window_scores: np.ndarray, centers: np.ndarray, n_frames: int,
               threshold: float, fps: float, low: float | None = None) -> np.ndarray:
    """Nota por janela → episódios por quadro, sem trechos curtos demais.

    Com `low`, histerese: o episódio nasce onde a nota passa de `threshold`, mas
    só termina quando ela cai abaixo de `low`. A nota sobe e desce mais devagar
    que o rato, e com um corte único o episódio começa tarde e acaba cedo.

    Medida e não adotada para contar rearings. Deixando um animal de fora, nas
    23 sessões avaliáveis, com os limiares escolhidos nas outras sessões por
    pareamento um para um, o corte único (0,45) acha 75% com viés de contagem
    de 0%, e a histerese (0,6/0,25) acha 70% com viés de -17%. Ela troca
    rearings achados por precisão (84% contra 75%) e funde rearings vizinhos,
    porque entre dois deles a nota raramente cai até o limiar baixo.
    """
    scores = frame_scores(window_scores, centers, n_frames)
    mask = scores >= threshold
    if low is not None and low < threshold:
        mask = _grow(mask, scores >= low)
    return _sustained(mask, int(round(MIN_BOUT_SEC * fps)))


def _grow(seeds: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Os trechos contínuos de `allowed` que contêm ao menos uma semente."""
    out = np.zeros_like(allowed)
    for start, end in bouts(allowed):
        if seeds[start:end + 1].any():
            out[start:end + 1] = True
    return out


def triage_episodes(window_scores: np.ndarray, centers: np.ndarray, n_frames: int,
                    freezing: np.ndarray, fps: float, low: float, high: float,
                    cut: float | None = None, behavior: str = "rearing") -> list[tuple[int, int, float, bool]]:
    """Episódios em três classes: aceitos, a revisar e, por omissão, descartados.

    Devolve (início, fim, nota máxima, aceito) em índices de linha. Aceito é o
    episódio cuja nota máxima chega a `high`; a revisar, o que fica entre `low` e
    `high`; abaixo de `low`, nada é apresentado.

    Os limites saem de `cut`, o limiar de operação do modelo, e `low` só pesca os
    candidatos fracos, que não chegam a `cut` em nenhum ponto. Recortar tudo pelo
    limiar baixo junta dois rearings vizinhos num episódio só, porque entre eles
    a nota raramente cai abaixo de 0,1. Sem `cut`, o recorte é pelo limiar baixo.
    """
    scores = frame_scores(window_scores, centers, n_frames)

    def found(threshold):
        return bouts(gate(behavior, frame_mask(window_scores, centers, n_frames, threshold, fps), freezing, fps))

    def classify(segments):
        return [(int(a), int(b), float(scores[a:b + 1].max()), bool(scores[a:b + 1].max() >= high))
                for a, b in segments]

    if cut is None:
        return classify(found(low))
    main = found(cut)
    covered = np.zeros(n_frames, dtype=bool)
    for a, b in main:
        covered[a:b + 1] = True
    weak = [(a, b) for a, b in found(low) if not covered[a:b + 1].any()]
    return sorted(classify(main) + [(a, b, peak, False) for a, b, peak, _ in classify(weak)])


def gate(behavior: str, mask: np.ndarray, freezing: np.ndarray, fps: float) -> np.ndarray:
    """Regras finais de cada comportamento sobre os episódios detectados.

    Rearing: descidas de menos de MERGE_GAP_SEC não encerram o episódio — a
    mesma definição aplicada aos rótulos, para contagem e tempo serem comparáveis.

    Grooming: a camada de movimento veta o que é incompatível com ela. Grooming é
    movimento — cabeça e patas trabalhando —, e freezing é a ausência dele. Sem
    esse veto, o detector confundia rato imóvel com grooming: nos clipes de
    outros animais, 69 de 106 alarmes falsos eram "parado" ou "freezing", 44 deles
    dos dois animais mais imóveis. Custo medido nas sessões: 7% do grooming
    marcado cai no que a camada chama de freezing.
    """
    mask = close_gaps(mask, merge_gap(behavior, fps))
    if behavior == "grooming":
        mask = mask & ~freezing
    return _sustained(mask, int(round(MIN_BOUT_SEC * fps)))


def bouts(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def compare(marked: np.ndarray, predicted: np.ndarray, fps: float) -> dict:
    """Concordância quadro a quadro e por episódio entre a marcação e o detector."""
    manual, found = bouts(marked), bouts(predicted)
    hit = sum(predicted[s:e + 1].any() for s, e in manual)
    real = sum(marked[s:e + 1].any() for s, e in found)
    both = (marked & predicted).sum()
    return {
        "manual_s": marked.sum() / fps,
        "detected_s": predicted.sum() / fps,
        "manual_bouts": len(manual),
        "detected_bouts": len(found),
        "frames_found": both / marked.sum() if marked.any() else np.nan,
        "frames_right": both / predicted.sum() if predicted.any() else np.nan,
        "bouts_found": hit / len(manual) if manual else np.nan,
        "bouts_right": real / len(found) if found else np.nan,
        "matched_bouts": matched_episodes(manual, found),
    }


def matched_episodes(manual: list[tuple[int, int]], found: list[tuple[int, int]]) -> int:
    """Pares (marcado, detectado) que se cruzam, com cada episódio usado uma vez só.

    É o critério para contar rearings. Por sobreposição, uma detecção longa que
    encosta em dois rearings vale pelos dois, e duas detecções dentro de um
    rearing são duas certas. Um para um, fundir vira um acerto e uma perda, e
    partir vira um acerto e um alarme falso: é o erro que a contagem sente.
    """
    used, pairs = set(), 0
    for start, end in found:
        for index, (first, last) in enumerate(manual):
            if index not in used and first <= end and start <= last:
                used.add(index)
                pairs += 1
                break
    return pairs


def make_model(seed: int = 0):
    """Árvores rasas, parâmetros fixados antes de ver resultado.

    Sem peso balanceado de classes, ao contrário do modelo da triagem: aqui o
    limiar é escolhido numa validação interna, e é ele que compensa o
    comportamento ser raro. O peso ainda deixava cada treino 6 vezes mais lento —
    com pesos, o scikit-learn calcula as faixas de cada feature por quantil
    ponderado.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
        l2_regularization=1.0, random_state=seed)


# Contexto de notas, em janelas (STEP quadros): de ±0,27 s a ±3,2 s, mais o
# máximo e a média em volta (±1,2 s e ±3,3 s).
CONTEXT_OFFSETS = (1, 2, 3, 4, 6, 8, 12)
CONTEXT_SPANS = (9, 25)


def score_context(scores: np.ndarray) -> pd.DataFrame:
    """As notas do primeiro estágio em volta de cada janela, na ordem do vídeo."""
    s = pd.Series(np.asarray(scores, dtype=float))
    columns = {"ctx_self": s}
    for k in CONTEXT_OFFSETS:
        columns[f"ctx_prev{k}"], columns[f"ctx_next{k}"] = s.shift(k), s.shift(-k)
    for span in CONTEXT_SPANS:
        rolling = s.rolling(span, center=True, min_periods=1)
        columns[f"ctx_max{span}"], columns[f"ctx_mean{span}"] = rolling.max(), rolling.mean()
    return pd.DataFrame(columns)


class ContextModel:
    """Dois estágios: o segundo vê as features e as notas do primeiro em volta.

    O primeiro estágio julga cada janela sozinha. O segundo aprende com a
    vizinhança: uma janela de nota mediana entre vizinhas altas faz parte do
    rearing; um pico isolado é ruído. Deixando um animal de fora, em 23 sessões,
    o F1 um para um foi de 0,772 para 0,796, com 31 dos 49 rearings curtos achados
    contra 25 (models/experimentos/modelos.py). Estágios mais profundos não
    ganharam: ficaram mais precisos, mas contaram de 8% a 13% a menos.

    As notas que o segundo estágio vê no treino vêm de validação cruzada por
    animal: notas de sessões que o primeiro estágio já viu seriam boas demais, e
    ele aprenderia a confiar demais nelas.

    `predict_proba` recebe as janelas de **uma sessão inteira, na ordem do vídeo**:
    o contexto é o das janelas vizinhas na tabela.
    """

    def __init__(self, folds: int = 5, seed: int = 0):
        self.folds, self.seed = folds, seed

    def fit_sessions(self, Xs: list, ys: list, groups: list, seen: list | None = None) -> "ContextModel":
        from sklearn.model_selection import GroupKFold

        Xs = [x.reset_index(drop=True) for x in Xs]
        seen = [np.ones(len(x), bool) for x in Xs] if seen is None else list(seen)
        n_groups = len(set(groups))
        if n_groups < 2:
            raise ValueError("o contexto precisa de pelo menos 2 animais para a validação cruzada")

        def fit(index):
            return make_model(self.seed).fit(pd.concat([Xs[i][seen[i]] for i in index]),
                                             np.concatenate([np.asarray(ys[i])[seen[i]] for i in index]))

        held_out = [None] * len(Xs)
        for train, test in GroupKFold(n_splits=min(self.folds, n_groups)).split(Xs, groups=groups):
            model = fit(train)
            for i in test:
                held_out[i] = model.predict_proba(Xs[i])[:, 1]
        self.first = fit(range(len(Xs)))
        Z = pd.concat([self._stack(x, s)[m] for x, s, m in zip(Xs, held_out, seen)])
        self.second = make_model(self.seed).fit(Z, np.concatenate([np.asarray(y)[m] for y, m in zip(ys, seen)]))
        return self

    @staticmethod
    def _stack(X: pd.DataFrame, first_scores: np.ndarray) -> pd.DataFrame:
        return pd.concat([X.reset_index(drop=True), score_context(first_scores)], axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.second.predict_proba(self._stack(X, self.first.predict_proba(X)[:, 1]))
