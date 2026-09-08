"""Rótulos multi-etiqueta: cada comportamento é uma trilha independente de bouts.

Um mesmo intervalo de quadros pode aparecer sob mais de um comportamento — é
exatamente essa coocorrência que a cadeia de regras do MindTrace não consegue
representar, e o motivo de o conjunto ser multi-etiqueta.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import BEHAVIORS, LABEL_COLUMNS, UNSCORABLE


def load_labels(path: str | Path) -> pd.DataFrame:
    """Carrega e valida um CSV de bouts anotados."""
    frame = pd.read_csv(path, encoding="utf-8-sig")

    missing = [column for column in LABEL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: colunas ausentes no CSV de rótulos: {missing}")

    valid = set(BEHAVIORS) | {UNSCORABLE}
    unknown = sorted(set(frame["behavior"]) - valid)
    if unknown:
        raise ValueError(f"{path}: comportamentos desconhecidos: {unknown}")

    inverted = frame["start_frame"] > frame["end_frame"]
    if inverted.any():
        rows = frame.index[inverted].tolist()
        raise ValueError(f"{path}: start_frame > end_frame nas linhas {rows}")

    return frame


def overlapping_within_behavior(labels: pd.DataFrame) -> pd.DataFrame:
    """Bouts do mesmo comportamento e anotador que se sobrepõem.

    Coocorrência entre comportamentos diferentes é esperada; dentro do mesmo
    comportamento indica bouts que deveriam ter sido fundidos na anotação.
    """
    group_columns = ["session_id", "field", "behavior", "annotator"]
    offenders = []
    for key, group in labels.groupby(group_columns):
        ordered = group.sort_values("start_frame")
        previous_end = None
        for row in ordered.itertuples():
            if previous_end is not None and row.start_frame <= previous_end:
                offenders.append({**dict(zip(group_columns, key)), "start_frame": row.start_frame})
            previous_end = row.end_frame if previous_end is None else max(previous_end, row.end_frame)
    return pd.DataFrame(offenders)


def bouts_for(labels: pd.DataFrame, behavior: str, field: int | None = None) -> list[tuple[int, int]]:
    """Intervalos (início, fim) inclusivos de um comportamento, fundidos."""
    selected = labels[labels["behavior"] == behavior]
    if field is not None:
        selected = selected[selected["field"] == field]
    if selected.empty:
        return []

    intervals = sorted(zip(selected["start_frame"], selected["end_frame"]))
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def intervals_to_mask(intervals, frames: np.ndarray) -> np.ndarray:
    """Projeta intervalos de quadros sobre os quadros realmente registrados."""
    mask = np.zeros(len(frames), dtype=bool)
    for start, end in intervals:
        low = np.searchsorted(frames, start, side="left")
        high = np.searchsorted(frames, end, side="right")
        mask[low:high] = True
    return mask


def label_matrix(labels: pd.DataFrame, frames: np.ndarray, field: int | None = None):
    """Matriz booleana (quadros × comportamentos) mais a máscara de pontuáveis.

    Retorna (y, scorable) onde y[i, j] indica se o comportamento j ocorre no
    quadro i, e scorable[i] é False onde a anotação marcou 'unscorable'.
    """
    y = np.zeros((len(frames), len(BEHAVIORS)), dtype=bool)
    for index, behavior in enumerate(BEHAVIORS):
        y[:, index] = intervals_to_mask(bouts_for(labels, behavior, field), frames)

    unscorable = intervals_to_mask(bouts_for(labels, UNSCORABLE, field), frames)
    return y, ~unscorable
