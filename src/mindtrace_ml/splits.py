"""Divisão treino/validação/teste agrupada por animal.

Quadros vizinhos da mesma sessão são quase idênticos: uma divisão aleatória por
quadro entrega acurácia altíssima que é puro vazamento. A única divisão honesta
agrupa por animal — nenhum animal aparece em mais de uma partição.
"""

import numpy as np
import pandas as pd


def split_by_animal(manifest: pd.DataFrame, fractions=(0.6, 0.2, 0.2), seed: int = 42):
    """Sorteia animais (não sessões, não quadros) para treino, validação e teste."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"as frações devem somar 1.0, receberam {fractions}")

    animals = np.array(sorted(manifest["animal_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(animals)

    n_train = int(round(len(animals) * fractions[0]))
    n_validation = int(round(len(animals) * fractions[1]))

    partitions = {
        "train": set(animals[:n_train]),
        "validation": set(animals[n_train:n_train + n_validation]),
        "test": set(animals[n_train + n_validation:]),
    }

    assignment = manifest.copy()
    assignment["split"] = assignment["animal_id"].map(
        lambda animal: next(name for name, members in partitions.items() if animal in members)
    )
    return assignment


def check_leakage(assignment: pd.DataFrame) -> list[str]:
    """Animais presentes em mais de uma partição. Deve retornar lista vazia."""
    per_animal = assignment.groupby("animal_id")["split"].nunique()
    return sorted(per_animal.index[per_animal > 1].tolist())
