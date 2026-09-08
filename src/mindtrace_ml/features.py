"""Leitura dos CSVs de features exportados pelo MindTrace."""

from pathlib import Path

import pandas as pd

from .schema import FEATURE_COLUMNS, FRAME_COLUMN, RULE_LABEL_COLUMN


def load_features(path: str | Path) -> pd.DataFrame:
    """Carrega um CSV de InferenceController::exportBehaviorFeatures().

    O arquivo é gravado com BOM UTF-8 para compatibilidade com Excel, daí o
    utf-8-sig. Retorna as colunas na ordem do contrato, ordenadas por quadro.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")

    expected = (FRAME_COLUMN, *FEATURE_COLUMNS, RULE_LABEL_COLUMN)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: colunas ausentes no CSV de features: {missing}")

    frame = frame.loc[:, list(expected)].sort_values(FRAME_COLUMN, ignore_index=True)

    duplicates = frame[FRAME_COLUMN].duplicated().sum()
    if duplicates:
        raise ValueError(f"{path}: {duplicates} quadros duplicados na coluna 'frame'")

    return frame


def frame_index(features: pd.DataFrame):
    """Vetor dos quadros efetivamente presentes — não é um range contíguo."""
    return features[FRAME_COLUMN].to_numpy()


def coverage(features: pd.DataFrame) -> dict:
    """Quanto da sessão sobreviveu ao limiar de confiança de pose de 0,75."""
    frames = frame_index(features)
    if len(frames) == 0:
        return {"recorded": 0, "span": 0, "coverage": 0.0, "largest_gap": 0}

    span = int(frames[-1] - frames[0] + 1)
    gaps = frames[1:] - frames[:-1]
    return {
        "recorded": int(len(frames)),
        "span": span,
        "coverage": float(len(frames) / span) if span else 0.0,
        "largest_gap": int(gaps.max()) if len(gaps) else 0,
    }
