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

from .detectors import _sustained
from .labels import bouts_for, intervals_to_mask
from .triage_model import SPEC, describe

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
}
SCALES = (15, 60)        # quadros: ~0,5 s e 2 s a 30 fps
STEP = 8                 # uma janela a cada ~0,27 s
MIN_BOUT_SEC = 0.3


def centered_features(signals: pd.DataFrame, centers: np.ndarray, scales=SCALES,
                      spec: dict | None = None) -> pd.DataFrame:
    """Features de janelas centradas em cada quadro de `centers` (índices de linha).

    Sem `spec`, usa os sinais da triagem e mais os extras que estiverem em `signals`.
    """
    if spec is None:
        spec = {**SPEC, **{k: v for k, v in EXTRA_SPEC.items() if k in signals}}
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


def window_centers(n_frames: int, step: int = STEP) -> np.ndarray:
    return np.arange(0, n_frames, step)


def targets(labels: pd.DataFrame, frames: np.ndarray, centers: np.ndarray) -> pd.DataFrame:
    """Para cada centro, se cada comportamento estava marcado naquele quadro."""
    out = {}
    for behavior in BEHAVIORS:
        mask = intervals_to_mask(bouts_for(labels, behavior), frames)
        out[behavior] = mask[centers]
    return pd.DataFrame(out)


def frame_mask(window_scores: np.ndarray, centers: np.ndarray, n_frames: int,
               threshold: float, fps: float) -> np.ndarray:
    """Nota por janela → episódios por quadro, sem trechos curtos demais."""
    nearest = np.clip(np.rint(np.arange(n_frames) / (centers[1] - centers[0])).astype(int),
                      0, len(centers) - 1) if len(centers) > 1 else np.zeros(n_frames, int)
    return _sustained(window_scores[nearest] >= threshold, int(round(MIN_BOUT_SEC * fps)))


def gate(behavior: str, mask: np.ndarray, freezing: np.ndarray, fps: float) -> np.ndarray:
    """A camada de movimento veta o que é incompatível com ela.

    Grooming é movimento — cabeça e patas trabalhando —, e freezing é a ausência
    dele. Sem esse veto, o detector confundia rato imóvel com grooming: nos
    clipes de outros animais, 69 de 106 alarmes falsos eram "parado" ou
    "freezing", 44 deles dos dois animais mais imóveis. Custo medido nas
    sessões: 7% do grooming marcado cai no que a camada chama de freezing.
    """
    if behavior != "grooming":
        return mask
    return _sustained(mask & ~freezing, int(round(MIN_BOUT_SEC * fps)))


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
    }


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
