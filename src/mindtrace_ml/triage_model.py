"""Decisão da triagem aprendida dos rótulos: rotina ou "outros".

As regras geométricas chegaram ao limite na rodada de teste: cada ajuste
recuperava pontos nos clipes usados para calibrá-lo e perdia parte deles nos
clipes novos — o grooming que chegava ao pesquisador caiu de 83% para 50%. Um
limiar escolhido olhando 17 exemplos se ajusta àqueles animais.

O modelo aprende a mesma pergunta — há algo aqui que a rotina não descreve? — a
partir dos clipes rotulados, com três cuidados:

1. **Janelas, não quadros.** O rótulo humano é de presença num clipe de 2–3 s, e
   o modelo vê o clipe inteiro resumido em estatísticas. Na aplicação, a mesma
   janela desliza sobre a sessão.
2. **As regras entram como features.** A fração da janela em cada faixa da
   triagem geométrica é informação que o modelo pode usar ou ignorar: ele parte
   de onde as regras pararam, em vez de começar do zero.
3. **Avaliação por animal.** O modelo é medido sempre em animais que não viu no
   treino — TR e TT do mesmo rato nunca ficam em lados opostos.

As faixas de rotina continuam vindo das regras. O modelo só decide o que vai
para revisão.
"""

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .detectors import BEHAVIOR_ORDER, REVIEW_FLAGS, Thresholds, path_straightness, triage
from .kinematics import frame_kinematics

KEYPOINTS = ("nose", "ear_left", "ear_right", "neck", "body", "tail_base")

# Rótulos finos que a triagem funde em "outros". Sniffing fora do objeto fica de
# fora por padrão: não é desfecho do NOR.
NOTABLE = frozenset({"grooming", "rearing", "other"})
STILL = frozenset({"still", "freezing", "low_activity"})

RULE_COLUMNS = (*BEHAVIOR_ORDER, *REVIEW_FLAGS, "unscorable")

# Estatísticas por sinal. Quantis em vez de mínimo e máximo: os clipes das
# primeiras rodadas têm 3 s e os da de teste 2 s, e o máximo cresce com o tamanho
# da janela — o p90 muito menos.
SPEC = {
    "body_speed": ("median", "p90"),
    "typical_speed": ("median", "p90"),
    "head_speed": ("median", "p90"),
    "nose_speed": ("median", "p90"),
    "head_turn": ("median", "p90"),
    "body_length": ("median", "p10", "std"),
    "full_length": ("median", "p10", "std"),
    "ear_span": ("median", "std"),
    "bend": ("median", "p90"),
    "straightness": ("median", "p10"),
    "body_x": ("std",),
    "body_y": ("std",),
    "object_distance": ("p10", "median"),
    "wall_distance": ("p10", "median"),
    **{f"conf_{name}": ("mean", "p10") for name in KEYPOINTS},
    **{f"rule_{name}": ("mean",) for name in RULE_COLUMNS},
}

_STATS = {
    "mean": lambda v: np.nanmean(v, axis=1),
    "median": lambda v: np.nanmedian(v, axis=1),
    "p10": lambda v: np.nanpercentile(v, 10, axis=1),
    "p90": lambda v: np.nanpercentile(v, 90, axis=1),
    "std": lambda v: np.nanstd(v, axis=1),
}


def label_sets(labels: pd.DataFrame) -> pd.Series:
    """Conjunto de rótulos por clipe, qualquer que seja a rodada.

    Rodadas antigas têm um rótulo em `label` e, em dois casos, extras em `also`;
    a de teste tem todos em `labels`, separados por ponto e vírgula.
    """
    def one(row):
        if isinstance(row.get("labels"), str) and row["labels"]:
            return set(row["labels"].split(";"))
        names = {row["label"]}
        if isinstance(row.get("also"), str) and row["also"]:
            names |= set(row["also"].split(";"))
        return names
    return labels.apply(one, axis=1)


def notable_set(sniffing_is_notable: bool = False) -> frozenset:
    return NOTABLE | ({"sniffing"} if sniffing_is_notable else frozenset())


def animal_of(session_id: str) -> int:
    """TR_22_cam1 e TT_22_cam1 são o mesmo animal, em dias diferentes."""
    return int(session_id.split("_")[1])


def _position(kinematics: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    # Posições de frame_kinematics: já vêm NaN abaixo do limiar de confiança.
    return kinematics[f"posx_{name}"].to_numpy(float), kinematics[f"posy_{name}"].to_numpy(float)


def _speed(x: np.ndarray, y: np.ndarray, fps: float) -> np.ndarray:
    return np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * fps


def _relative_to_session(values: np.ndarray) -> np.ndarray:
    """Divide pela mediana da sessão: tamanho do animal e zoom da câmera saem da conta."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        typical = np.nanmedian(values)
    return values / typical if typical > 0 else np.full_like(values, np.nan)


def frame_signals(pose: pd.DataFrame, kinematics: pd.DataFrame, triaged: pd.DataFrame,
                  objects: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Sinais por quadro que as janelas resumem.

    Cada um tem um motivo etológico: a cabeça relativa ao dorso é o que o grooming
    mexe com o corpo parado; o comprimento aparente do corpo encurta quando o
    animal se levanta; a confiança do focinho cai quando as patas o cobrem; a
    distância à parede separa o rearing apoiado do resto.
    """
    nose, neck, body, tail = (_position(kinematics, n) for n in ("nose", "neck", "body", "tail_base"))
    ear_left, ear_right = _position(kinematics, "ear_left"), _position(kinematics, "ear_right")
    ear_mid = ((ear_left[0] + ear_right[0]) / 2, (ear_left[1] + ear_right[1]) / 2)

    out = pd.DataFrame({"frame": pose["frame"].to_numpy()})
    out["body_speed"] = kinematics["speed_body"].to_numpy(float)

    speeds = kinematics[[f"speed_{n}" for n in KEYPOINTS]].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["typical_speed"] = np.nanmedian(speeds, axis=1)

    out["head_speed"] = _speed(neck[0] - body[0], neck[1] - body[1], fps)
    out["nose_speed"] = _speed(nose[0] - neck[0], nose[1] - neck[1], fps)

    heading = np.arctan2(nose[1] - ear_mid[1], nose[0] - ear_mid[0])
    turn = np.angle(np.exp(1j * np.diff(heading, prepend=np.nan)))
    out["head_turn"] = np.degrees(np.abs(turn)) * fps

    out["body_length"] = _relative_to_session(np.hypot(neck[0] - tail[0], neck[1] - tail[1]))
    out["full_length"] = _relative_to_session(np.hypot(nose[0] - tail[0], nose[1] - tail[1]))
    out["ear_span"] = _relative_to_session(
        np.hypot(ear_left[0] - ear_right[0], ear_left[1] - ear_right[1]))

    # Curvatura: ângulo entre cauda→dorso e dorso→pescoço. O grooming dobra o corpo.
    ax, ay = body[0] - tail[0], body[1] - tail[1]
    bx, by = neck[0] - body[0], neck[1] - body[1]
    out["bend"] = np.degrees(np.abs(np.arctan2(ax * by - ay * bx, ax * bx + ay * by)))

    out["straightness"] = path_straightness(kinematics, fps, 1.0)
    out["body_x"], out["body_y"] = body

    distances = [np.hypot(o.center_x - nose[0], o.center_y - nose[1]) - o.radius
                 for o in objects.itertuples()]
    out["object_distance"] = np.min(distances, axis=0) if distances else np.nan

    # A arena não está marcada; o animal a percorre inteira ao longo da sessão,
    # então os percentis extremos da posição do dorso aproximam as paredes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low_x, high_x = np.nanpercentile(body[0], [1, 99])
        low_y, high_y = np.nanpercentile(body[1], [1, 99])
    out["wall_distance"] = np.minimum.reduce(
        [neck[0] - low_x, high_x - neck[0], neck[1] - low_y, high_y - neck[1]])

    for name in KEYPOINTS:
        out[f"conf_{name}"] = pose[f"{name}_p"].to_numpy(float)
    for name in RULE_COLUMNS:
        out[f"rule_{name}"] = triaged[name].to_numpy(float)
    return out


def describe(block: np.ndarray) -> pd.DataFrame:
    """Resume janelas em features. `block` tem forma (janelas, quadros, sinais)."""
    columns = {}
    with warnings.catch_warnings():
        # Janela inteira sem pose num sinal dá NaN, que o modelo trata como ausente.
        warnings.simplefilter("ignore", RuntimeWarning)
        for index, (name, stats) in enumerate(SPEC.items()):
            values = block[:, :, index]
            for stat in stats:
                columns[f"{name}_{stat}"] = _STATS[stat](values)
    return pd.DataFrame(columns)


def _matrix(signals: pd.DataFrame) -> np.ndarray:
    return signals[list(SPEC)].to_numpy(float)


def clip_features(signals: pd.DataFrame, spans) -> pd.DataFrame:
    """Features de trechos [início, fim] em quadros do vídeo, inclusive."""
    matrix = _matrix(signals)
    frames = signals["frame"].to_numpy()
    rows = []
    for start, end in spans:
        first, last = np.searchsorted(frames, start), np.searchsorted(frames, end, side="right")
        rows.append(describe(matrix[first:last][None]))
    return pd.concat(rows, ignore_index=True)


def sliding_features(signals: pd.DataFrame, window: int, step: int):
    """Features de janelas deslizantes; devolve também o quadro central de cada uma."""
    matrix = _matrix(signals)
    if len(matrix) < window:
        return describe(matrix[None]), np.array([len(matrix) // 2])
    view = sliding_window_view(matrix, window, axis=0)[::step]   # (janelas, sinais, quadros)
    starts = np.arange(len(view)) * step
    return describe(view.transpose(0, 2, 1)), starts + (window - 1) / 2


def frame_scores(window_scores: np.ndarray, centers: np.ndarray, n_frames: int) -> np.ndarray:
    """Cada quadro recebe a nota da janela cujo centro está mais perto dele."""
    if len(centers) == 1:
        return np.full(n_frames, window_scores[0])
    step = centers[1] - centers[0]
    nearest = np.rint((np.arange(n_frames) - centers[0]) / step).astype(int)
    return window_scores[np.clip(nearest, 0, len(centers) - 1)]


def prepare_session(pose: pd.DataFrame, objects: pd.DataFrame, fps: float,
                    window: int, step: int, thresholds: Thresholds | None = None) -> dict:
    """Tudo o que o modelo precisa de uma sessão: triagem geométrica, sinais e janelas."""
    kinematics = frame_kinematics(pose, KEYPOINTS, fps)
    triaged = triage(pose, kinematics, objects, fps, thresholds or Thresholds())
    signals = frame_signals(pose, kinematics, triaged, objects, fps)
    windows, centers = sliding_features(signals, window, step)
    return {"frames": signals["frame"].to_numpy(),
            "unscorable": triaged["unscorable"].to_numpy(bool),
            "triaged": triaged, "signals": signals,
            "windows": windows, "centers": centers}


def score_session(model, session: dict) -> np.ndarray:
    """Nota por quadro: probabilidade de haver algo que a rotina não descreve."""
    window_scores = model.predict_proba(session["windows"])[:, 1]
    return frame_scores(window_scores, session["centers"], len(session["frames"]))


def surfacing_score(scores: np.ndarray, missing: np.ndarray, min_presence: float) -> float:
    """Maior limiar em que o trecho ainda chega ao pesquisador.

    Mesma regra de evaluate_triage — revisão ou pose ausente em ao menos
    `min_presence` do trecho —, reescrita como uma nota: o trecho é visto num
    limiar t se a k-ésima maior nota entre seus quadros válidos é ≥ t. Com isso a
    curva inteira sai de uma ordenação, sem reavaliar trecho por trecho a cada
    limiar.
    """
    if missing.mean() >= min_presence:
        return np.inf
    needed = int(np.ceil(min_presence * len(scores) - 1e-9))
    valid = np.sort(scores[~missing])[::-1]
    return float(valid[needed - 1]) if len(valid) >= needed else -np.inf


def make_model(seed: int = 0):
    """Árvores rasas e regularizadas, com parâmetros fixados antes de ver resultado.

    Com algumas centenas de clipes, escolher hiperparâmetros pela própria
    avaliação a contaminaria. Gradient boosting porque aceita NaN — pose ausente é
    informação, não algo a imputar — e porque as interações que importam são do
    tipo "cabeça rápida *e* corpo parado", que um modelo linear não expressa.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=10,
        l2_regularization=1.0, class_weight="balanced", random_state=seed)
