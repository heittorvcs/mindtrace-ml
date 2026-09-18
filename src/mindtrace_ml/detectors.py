"""Detectores geométricos das faixas de rotina da triagem.

As três faixas "confiantes" — caminhando, congelamento e interação com objeto —
são definíveis por regras sobre coordenadas, sem modelo treinado e sem anotação
de comportamento. É isso que permite a primeira triagem existir antes de qualquer
rotulagem: o humano passa a validar e calibrar em vez de gerar dados do zero.

"Indeterminado" não é detectado — é o que resta quando nenhum detector dispara.
Ele nunca é treinado, e é essa ausência de definição própria que permite
subdividi-lo depois, acrescentando um detector de cada vez, sem invalidar nada
do que já foi anotado.

Os limiares são parâmetros porque são **decisão de política**: apertá-los reduz o
que escapa da revisão e reduz também o ganho de tempo. O ponto de operação é do
laboratório, não um ótimo matemático.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import mask_to_bouts


@dataclass
class Thresholds:
    """Limiares em unidades físicas — px/s e segundos, nunca px/quadro.

    Os valores padrão são ponto de partida e precisam ser calibrados contra os
    vídeos do laboratório; `suggest_thresholds` ajuda a ancorá-los na distribuição
    observada em vez de no chute.
    """

    walking_speed: float = 40.0       # px/s do centro do corpo
    walking_min_sec: float = 0.5

    freezing_speed: float = 8.0       # px/s máximo entre todos os pontos
    freezing_min_sec: float = 1.0

    object_margin: float = 12.0       # px além da borda do objeto
    object_angle: float = 60.0        # graus entre a direção da cabeça e o objeto
    object_min_sec: float = 0.3


def _sustained(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Zera trechos contíguos mais curtos que min_frames.

    Sem isso a saída oscila a cada quadro e a fila de revisão vira milhares de
    fragmentos de 33 ms, inúteis para quem vai assistir.
    """
    if min_frames <= 1 or not mask.any():
        return mask

    out = mask.copy()
    start = None
    for index, value in enumerate(np.append(mask, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start < min_frames:
                out[start:index] = False
            start = None
    return out


def head_direction(pose: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Vetor do ponto médio das orelhas até o focinho — para onde a cabeça aponta.

    Só é calculável com os 6 pontos; com focinho e corpo apenas, não há como
    distinguir cheirar o objeto de apenas estar perto dele.
    """
    ear_x = (pose["ear_left_x"].to_numpy(float) + pose["ear_right_x"].to_numpy(float)) / 2
    ear_y = (pose["ear_left_y"].to_numpy(float) + pose["ear_right_y"].to_numpy(float)) / 2
    return pose["nose_x"].to_numpy(float) - ear_x, pose["nose_y"].to_numpy(float) - ear_y


def detect_walking(kinematics: pd.DataFrame, fps: float, thresholds: Thresholds) -> np.ndarray:
    speed = kinematics["speed_body"].to_numpy(float)
    mask = np.nan_to_num(speed, nan=0.0) >= thresholds.walking_speed
    return _sustained(mask, int(round(thresholds.walking_min_sec * fps)))


def detect_freezing(kinematics: pd.DataFrame, fps: float, thresholds: Thresholds) -> np.ndarray:
    columns = [c for c in kinematics.columns if c.startswith("speed_")]
    speeds = kinematics[columns].to_numpy(float)

    # Quadro sem pose não é imobilidade — é ausência de informação.
    known = ~np.isnan(speeds).all(axis=1)
    fastest = np.nanmax(np.where(np.isnan(speeds), -np.inf, speeds), axis=1)

    mask = known & (fastest <= thresholds.freezing_speed)
    return _sustained(mask, int(round(thresholds.freezing_min_sec * fps)))


def detect_object_interaction(pose: pd.DataFrame, objects: pd.DataFrame, fps: float,
                              thresholds: Thresholds) -> np.ndarray:
    """Focinho dentro da zona do objeto **e** cabeça orientada para ele.

    A exigência de orientação é o que separa explorar de apenas passar perto — e
    é a razão de o conjunto de pontos incluir as orelhas.
    """
    nose_x = pose["nose_x"].to_numpy(float)
    nose_y = pose["nose_y"].to_numpy(float)
    head_dx, head_dy = head_direction(pose)

    head_norm = np.hypot(head_dx, head_dy)
    cos_limit = np.cos(np.deg2rad(thresholds.object_angle))

    mask = np.zeros(len(pose), dtype=bool)
    for obj in objects.itertuples():
        to_x = obj.center_x - nose_x
        to_y = obj.center_y - nose_y
        distance = np.hypot(to_x, to_y)

        near = distance <= obj.radius + thresholds.object_margin

        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = (head_dx * to_x + head_dy * to_y) / (head_norm * distance)
        facing = np.nan_to_num(cosine, nan=-1.0) >= cos_limit

        mask |= near & facing

    return _sustained(mask, int(round(thresholds.object_min_sec * fps)))


BEHAVIOR_ORDER = ("walking", "freezing", "object_interaction")


def triage(pose: pd.DataFrame, kinematics: pd.DataFrame, objects: pd.DataFrame,
           fps: float, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Aplica os detectores e marca como 'review' o que nenhum reconheceu."""
    thresholds = thresholds or Thresholds()

    result = pd.DataFrame({
        "frame": pose["frame"].to_numpy(),
        "time_ms": pose["time_ms"].to_numpy(),
        "walking": detect_walking(kinematics, fps, thresholds),
        "freezing": detect_freezing(kinematics, fps, thresholds),
        "object_interaction": detect_object_interaction(pose, objects, fps, thresholds),
    })
    result["review"] = ~result[list(BEHAVIOR_ORDER)].any(axis=1)
    return result


def review_segments(triaged: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """Trechos que vão para o revisor, do mais longo para o mais curto.

    Ordenar por duração põe primeiro o que rende mais informação por minuto
    assistido — um trecho de 20 s tende a conter um comportamento inteiro, um de
    meio segundo raramente contém algo nomeável.
    """
    frames = triaged["frame"].to_numpy()
    bouts = mask_to_bouts(triaged["review"].to_numpy(bool), frames, max_gap)

    time_by_frame = dict(zip(frames, triaged["time_ms"].to_numpy()))
    rows = [{
        "start_frame": start,
        "end_frame": end,
        "start_ms": time_by_frame.get(start),
        "end_ms": time_by_frame.get(end),
        "n_frames": end - start + 1,
    } for start, end in bouts]

    segments = pd.DataFrame(rows)
    return segments.sort_values("n_frames", ascending=False, ignore_index=True)


def reduction_rate(triaged: pd.DataFrame) -> float:
    """Fração do vídeo que não vai para revisão — o benefício da triagem."""
    return 1.0 - float(triaged["review"].mean())


def suggest_thresholds(kinematics: pd.DataFrame) -> dict:
    """Percentis da velocidade observada, para ancorar os limiares nos dados.

    Substitui o chute inicial: a velocidade de caminhada de um animal depende da
    escala do vídeo, e um limiar em px/s só faz sentido contra a distribuição
    daquela gravação.
    """
    speed = kinematics["speed_body"].to_numpy(float)
    speed = speed[~np.isnan(speed)]
    if not len(speed):
        return {}
    return {f"p{p}": round(float(np.percentile(speed, p)), 1)
            for p in (10, 25, 50, 75, 90, 95)}
