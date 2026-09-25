"""Camada de movimento: o que se mede da posição, sem modelo e sem rotulagem.

Mover-se não é um comportamento que compete com os outros — o rato anda
cheirando, anda e levanta, anda até o objeto. Na rodada sorteada do vídeo
inteiro, 37 de 150 clipes tinham mais de um rótulo, quase todos "andando + algo".
Então o movimento sai da classificação e vira medida contínua, calculada em todo
quadro, sobre a qual a camada de pose acrescenta o que o animal está fazendo.

Três coisas saem daqui:

- **movendo ou parado**, e as grandezas de locomoção — distância, velocidade;
- **freezing**, que é definido por imobilidade total prolongada e por isso é
  medida de movimento, não de postura;
- **exploração de cada objeto**, definida pela geometria — focinho na zona e
  cabeça voltada para ele. É o resultado principal do NOR, e por isso fica numa
  regra que se pode auditar, e não num modelo.

A locomoção é também o controle do NOR: mostra se uma diferença de exploração
não vem só de um animal que se move menos.

Distâncias em pixels. Todas as sessões usam a mesma montagem de câmera, então
comparam entre si; para centímetros basta a escala da arena.
"""

import numpy as np
import pandas as pd

from .detectors import Thresholds, _smooth, _sustained, detect_freezing, object_exploration, pose_is_valid

# Passos maiores que isso num quadro (~900 px/s) são salto de detecção, não corrida.
MAX_STEP_PX = 30.0
MOVING_MIN_SEC = 0.3


def movement_layer(pose: pd.DataFrame, kinematics: pd.DataFrame, objects: pd.DataFrame,
                   fps: float, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Uma linha por quadro: posição, velocidade, movendo, freezing e exploração."""
    thresholds = thresholds or Thresholds()
    valid = pose_is_valid(kinematics)

    x = _smooth(kinematics["posx_body"].to_numpy(float), fps, thresholds.smooth_sec)
    y = _smooth(kinematics["posy_body"].to_numpy(float), fps, thresholds.smooth_sec)
    speed = _smooth(kinematics["speed_body"].to_numpy(float), fps, thresholds.smooth_sec)

    moving = _sustained(valid & (np.nan_to_num(speed, nan=0.0) >= thresholds.walking_speed),
                        int(round(MOVING_MIN_SEC * fps)))

    out = pd.DataFrame({
        "frame": pose["frame"].to_numpy(),
        "time_ms": pose["time_ms"].to_numpy(),
        "x": x,
        "y": y,
        "speed": speed,
        "moving": moving,
        "still": valid & ~moving,
        "freezing": detect_freezing(kinematics, fps, thresholds),
        "unscorable": ~valid,
    })

    # Distância só enquanto se move: parado, o ruído da detecção soma alguns
    # pixels por quadro e, em 5 minutos, vira uma caminhada que não existiu.
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    step = np.where(moving & (step <= MAX_STEP_PX), step, 0.0)
    out["distance"] = np.nancumsum(step)

    exploring = np.zeros(len(pose), dtype=bool)
    for object_id, mask in object_exploration(pose, objects, fps, thresholds).items():
        out[f"exploring_{object_id}"] = mask
        exploring |= mask
    out["exploring"] = exploring
    return out


def session_summary(layer: pd.DataFrame, fps: float) -> dict:
    """Números de uma sessão, no formato de uma linha de planilha."""
    seconds = len(layer) / fps
    moving_speed = layer.loc[layer.moving, "speed"]
    summary = {
        "duration_s": round(seconds, 1),
        "moving_pct": round(100 * layer.moving.mean(), 1),
        "still_pct": round(100 * layer.still.mean(), 1),
        "freezing_pct": round(100 * layer.freezing.mean(), 1),
        "unscorable_pct": round(100 * layer.unscorable.mean(), 1),
        "distance_px": round(float(layer.distance.iloc[-1]), 0) if len(layer) else 0.0,
        "speed_moving_px_s": round(float(moving_speed.median()), 1) if len(moving_speed) else 0.0,
    }
    total = 0.0
    for column in [c for c in layer.columns if c.startswith("exploring_")]:
        explored = layer[column].sum() / fps
        summary[f"exploration_{column.split('_')[1]}_s"] = round(float(explored), 1)
        summary[f"exploration_{column.split('_')[1]}_bouts"] = _count_bouts(layer[column].to_numpy())
        total += explored
    summary["exploration_total_s"] = round(float(total), 1)
    return summary


def _count_bouts(mask: np.ndarray) -> int:
    return int(np.sum(np.diff(np.concatenate([[0], mask.astype(int)])) == 1))
