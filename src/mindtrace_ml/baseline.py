"""A cadeia de regras do MindTrace como linha de base a ser superada.

classifySimple() emite exatamente um rótulo por quadro. Projetada no espaço
multi-etiqueta, ela vira uma matriz one-hot — e por construção nunca acerta um
quadro em que dois comportamentos coocorrem. Esse déficit é mensurável e é o
primeiro número que este repositório deve produzir.
"""

import numpy as np
import pandas as pd

from .schema import BEHAVIORS, RULE_LABEL_COLUMN, RULE_LABEL_TO_BEHAVIOR


def rule_predictions(features: pd.DataFrame) -> np.ndarray:
    """Converte a coluna rule_label em matriz booleana (quadros × comportamentos)."""
    behavior_index = {behavior: index for index, behavior in enumerate(BEHAVIORS)}
    predictions = np.zeros((len(features), len(BEHAVIORS)), dtype=bool)

    for row_position, rule_label in enumerate(features[RULE_LABEL_COLUMN].to_numpy()):
        behavior = RULE_LABEL_TO_BEHAVIOR.get(int(rule_label))
        if behavior is not None:
            predictions[row_position, behavior_index[behavior]] = True

    return predictions


def unreachable_recall(y_true: np.ndarray, scorable: np.ndarray | None = None) -> pd.DataFrame:
    """Revocação que a exclusividade mútua torna inatingível, por comportamento.

    Um quadro em que 'sniffing' e 'rearing' coocorrem só pode receber um dos dois
    rótulos das regras. O outro é um falso negativo estrutural, não um erro de
    calibração de limiar.
    """
    if scorable is None:
        scorable = np.ones(len(y_true), dtype=bool)
    observed = y_true[scorable]

    rows = []
    for index, behavior in enumerate(BEHAVIORS):
        present = observed[:, index]
        shared = present & (observed.sum(axis=1) > 1)
        rows.append({
            "behavior": behavior,
            "frames": int(present.sum()),
            "frames_shared": int(shared.sum()),
            "unreachable_pct": float(shared.sum() / present.sum() * 100) if present.sum() else 0.0,
        })
    return pd.DataFrame(rows)
