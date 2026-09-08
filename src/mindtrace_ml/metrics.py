"""Métricas em dois níveis: por quadro e por bout.

Para etologia o nível de bout costuma importar mais — o desfecho experimental é
contagem de episódios e tempo total, não acerto quadro a quadro. Um modelo pode
ter 85% de acerto por quadro e ainda assim fragmentar bouts a ponto de inutilizar
a contagem.
"""

import numpy as np
import pandas as pd

from .schema import BEHAVIORS, DEFAULT_MAX_GAP


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray, scorable: np.ndarray | None = None):
    """Precisão, revocação e F1 binários para cada comportamento."""
    if scorable is None:
        scorable = np.ones(len(y_true), dtype=bool)

    rows = []
    for index, behavior in enumerate(BEHAVIORS):
        truth = y_true[scorable, index]
        prediction = y_pred[scorable, index]

        true_positive = int(np.sum(truth & prediction))
        false_positive = int(np.sum(~truth & prediction))
        false_negative = int(np.sum(truth & ~prediction))
        true_negative = int(np.sum(~truth & ~prediction))

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append({
            "behavior": behavior,
            "prevalence": float(np.mean(truth)) if len(truth) else 0.0,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "tn": true_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows)


def mask_to_bouts(mask: np.ndarray, frames: np.ndarray, max_gap: int = DEFAULT_MAX_GAP):
    """Converte máscara booleana em intervalos, quebrando em lacunas de quadro.

    Trabalha sobre os quadros realmente registrados: uma lacuna maior que max_gap
    significa que o MindTrace não gravou aqueles quadros, e o bout não pode ser
    assumido contínuo através dela.
    """
    if not mask.any():
        return []

    positions = np.flatnonzero(mask)
    bouts = []
    start = frames[positions[0]]
    previous = frames[positions[0]]

    for position in positions[1:]:
        current = frames[position]
        if current - previous > max_gap:
            bouts.append((int(start), int(previous)))
            start = current
        previous = current

    bouts.append((int(start), int(previous)))
    return bouts


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    intersection = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if intersection <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    return intersection / union


def match_bouts(true_bouts, pred_bouts, iou_threshold: float = 0.5):
    """Pareamento guloso por IoU decrescente. Retorna os pares casados."""
    candidates = []
    for true_index, true_bout in enumerate(true_bouts):
        for pred_index, pred_bout in enumerate(pred_bouts):
            iou = interval_iou(true_bout, pred_bout)
            if iou >= iou_threshold:
                candidates.append((iou, true_index, pred_index))
    candidates.sort(reverse=True)

    used_true, used_pred, matches = set(), set(), []
    for iou, true_index, pred_index in candidates:
        if true_index in used_true or pred_index in used_pred:
            continue
        used_true.add(true_index)
        used_pred.add(pred_index)
        matches.append((true_index, pred_index, iou))
    return matches


def _total_frames(bouts):
    return sum(end - start + 1 for start, end in bouts)


def bout_metrics(y_true, y_pred, frames, scorable=None, iou_threshold: float = 0.5,
                 max_gap: int = DEFAULT_MAX_GAP):
    """Detecção de episódios, erro de contagem e erro de tempo total."""
    if scorable is None:
        scorable = np.ones(len(frames), dtype=bool)

    rows = []
    for index, behavior in enumerate(BEHAVIORS):
        true_bouts = mask_to_bouts(y_true[:, index] & scorable, frames, max_gap)
        pred_bouts = mask_to_bouts(y_pred[:, index] & scorable, frames, max_gap)
        matches = match_bouts(true_bouts, pred_bouts, iou_threshold)

        true_time = _total_frames(true_bouts)
        pred_time = _total_frames(pred_bouts)

        rows.append({
            "behavior": behavior,
            "n_true": len(true_bouts),
            "n_pred": len(pred_bouts),
            "matched": len(matches),
            "bout_recall": len(matches) / len(true_bouts) if true_bouts else 0.0,
            "bout_precision": len(matches) / len(pred_bouts) if pred_bouts else 0.0,
            "count_error": len(pred_bouts) - len(true_bouts),
            "time_true": true_time,
            "time_pred": pred_time,
            "time_error_pct": (pred_time - true_time) / true_time * 100 if true_time else np.nan,
        })
    return pd.DataFrame(rows)


def cooccurrence(y_true: np.ndarray, scorable: np.ndarray | None = None) -> pd.DataFrame:
    """Fração de quadros em que cada par de comportamentos ocorre junto.

    Estabelece o teto do classificador atual: as regras são mutuamente exclusivas
    e emitem um rótulo por quadro, então toda coocorrência aqui é revocação que
    elas não têm como alcançar.
    """
    if scorable is None:
        scorable = np.ones(len(y_true), dtype=bool)
    observed = y_true[scorable]

    size = len(BEHAVIORS)
    matrix = np.zeros((size, size))
    total = len(observed) or 1
    for i in range(size):
        for j in range(size):
            matrix[i, j] = np.sum(observed[:, i] & observed[:, j]) / total

    return pd.DataFrame(matrix, index=list(BEHAVIORS), columns=list(BEHAVIORS))


def multilabel_rate(y_true: np.ndarray, scorable: np.ndarray | None = None) -> float:
    """Fração de quadros pontuáveis com mais de um comportamento simultâneo."""
    if scorable is None:
        scorable = np.ones(len(y_true), dtype=bool)
    observed = y_true[scorable]
    if len(observed) == 0:
        return 0.0
    return float(np.mean(observed.sum(axis=1) > 1))
