"""Geometria da arena: o chão marcado à mão, um quadrilátero por câmera.

As câmeras do DVR são fixas (o fundo se desloca no máximo 0,4 px entre sessões
da mesma câmera), então o chão é marcado uma vez por câmera e vale para todas as
sessões dela. NOR e habituação usam caixas diferentes nas mesmas posições do
DVR, por isso a arena é identificada pela fase e pela câmera.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def arena_of(session_id: str) -> str:
    """'TR_19_cam1' e 'TT_5_cam1' -> 'NOR_cam1'; 'HD3_F6_cam2' -> 'HD_cam2'."""
    camera = session_id.rsplit("_cam", 1)[1]
    return f"{'HD' if session_id.startswith('HD') else 'NOR'}_cam{camera}"


def order_corners(points) -> np.ndarray:
    """Cantos em volta do centro, começando pelo de cima à esquerda, no sentido horário da imagem."""
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.argmin(ordered.sum(axis=1))   # menor x + y: o canto de cima à esquerda
    return np.roll(ordered, -start, axis=0)


def load_floors(path: str | Path) -> dict[str, np.ndarray]:
    """Arena -> cantos do chão (4 x 2), na ordem gravada por mark_arena."""
    path = Path(path)
    if not path.exists():
        return {}
    table = pd.read_csv(path, encoding="utf-8-sig").sort_values(["arena", "corner"])
    return {arena: group[["x", "y"]].to_numpy(float) for arena, group in table.groupby("arena")}


def floor_for(session_id: str, pose: pd.DataFrame, floors: dict, threshold: float = 0.25) -> np.ndarray:
    """O chão marcado da arena da sessão ou, sem marcação, o retângulo dos percentis do dorso.

    O retângulo é a parede que as features usavam antes do chão marcado: os
    percentis 1 e 99 da posição do dorso. Fica para arenas sem marcação.
    """
    floor = floors.get(arena_of(session_id))
    if floor is not None:
        return floor
    ok = pose["body_p"].to_numpy() >= threshold
    x, y = pose["body_x"].to_numpy()[ok], pose["body_y"].to_numpy()[ok]
    low_x, high_x = np.nanpercentile(x, [1, 99])
    low_y, high_y = np.nanpercentile(y, [1, 99])
    return np.array([[low_x, low_y], [high_x, low_y], [high_x, high_y], [low_x, high_y]])


def floor_distance(x: np.ndarray, y: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Distância com sinal até a borda do chão: positiva dentro, negativa fora, NaN sem ponto.

    Fora do chão quer dizer que o ponto foi projetado sobre a parede, o que, na
    câmera oblíqua, é o focinho de um rato em pé encostado nela.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    starts, ends = floor, np.roll(floor, -1, axis=0)
    distances = []
    for (ax, ay), (bx, by) in zip(starts, ends):
        dx, dy = bx - ax, by - ay
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0, 1)
        distances.append(np.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    nearest = np.min(distances, axis=0)
    return np.where(inside(x, y, floor), nearest, -nearest)


def inside(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ponto dentro do polígono (contagem de cruzamentos); NaN conta como fora."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    result = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(polygon, np.roll(polygon, -1, axis=0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            crosses = ((ay > y) != (by > y)) & (x < (bx - ax) * (y - ay) / (by - ay) + ax)
        result ^= np.nan_to_num(crosses, nan=False).astype(bool)
    return result


_FLOORS: dict = {}


def model_floor(bundle: dict | None, session_id: str, pose: pd.DataFrame, path: str | Path) -> np.ndarray | None:
    """O chão que as features de um modelo salvo precisam: o da sessão, se ele foi
    treinado com chão (bundle["floor"]); None nos modelos até o v8."""
    if not bundle or not bundle.get("floor"):
        return None
    key = str(path)
    if key not in _FLOORS:
        _FLOORS[key] = load_floors(path)
    return floor_for(session_id, pose, _FLOORS[key])
