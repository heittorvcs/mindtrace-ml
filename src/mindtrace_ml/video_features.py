"""Sinais tirados dos pixels do vídeo, guiados pela pose.

Os 6 pontos dizem onde estão focinho, orelhas e dorso, mas não enxergam duas
coisas que definem os comportamentos que mais importam aqui:

- **as patas no grooming.** O que distingue grooming de um rato parado mexendo a
  cabeça é a pata esfregando o rosto. Sem ponto nas patas, o que sobra é a
  agitação dos pixels em volta da cabeça — forte e rítmica no grooming, pequena
  no sniffing, nula no rato imóvel.
- **o levantar no rearing.** O modelo só com pose achava 63% dos rearings na
  parede e nenhum no objeto ou no meio da arena: tinha aprendido "rearing =
  estar perto da parede". A silhueta do rato muda de forma quando ele se
  levanta, em qualquer lugar — é o sinal do próprio movimento.

Silhueta por subtração de fundo: o fundo é a mediana de quadros espalhados pela
sessão, em que o rato, sempre em movimento, some. Num animal parado a maior
parte da sessão o fundo o absorve e a silhueta falha — nesses animais ela
importa pouco, porque eles quase não levantam.

Tudo é relativo à própria sessão (dividido pela mediana), para que iluminação e
tamanho do animal não virem sinal.
"""

import cv2
import numpy as np
import pandas as pd

BACKGROUND_SAMPLES = 60
FOREGROUND_THRESHOLD = 30      # níveis de cinza de diferença para o fundo
HEAD_BOX = 0.35                # meia largura da caixa da cabeça, em comprimentos de corpo
BODY_BOX = 0.6
SILHOUETTE_BOX = 1.0

COLUMNS = ("motion_head", "motion_body", "motion_ratio", "blob_area", "blob_height",
           "blob_width", "blob_aspect", "blob_dy")


def background(capture, n_frames: int, samples: int = BACKGROUND_SAMPLES) -> np.ndarray:
    frames = []
    for index in np.linspace(0, n_frames - 1, samples).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(frames), axis=0).astype(np.int16)


def _box(image: np.ndarray, x: float, y: float, half: int) -> np.ndarray:
    if not (np.isfinite(x) and np.isfinite(y)):
        return image[0:0, 0:0]
    top, left = max(0, int(y) - half), max(0, int(x) - half)
    return image[top:int(y) + half + 1, left:int(x) + half + 1]


def _point(pose: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    # Coordenada mesmo com confiança baixa — o palpite do modelo costuma cair no
    # lugar certo —, mas nunca o -1 de ponto não detectado.
    x, y = pose[f"{name}_x"].to_numpy(float), pose[f"{name}_y"].to_numpy(float)
    missing = (x < 0) | (y < 0)
    return np.where(missing, np.nan, x), np.where(missing, np.nan, y)


def video_signals(video_path, pose: pd.DataFrame) -> pd.DataFrame:
    """Um valor por quadro para cada coluna de COLUMNS, alinhado às linhas da pose."""
    capture = cv2.VideoCapture(str(video_path))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    plate = background(capture, n_frames)

    nose, body, tail = _point(pose, "nose"), _point(pose, "body"), _point(pose, "tail_base")
    left, right = _point(pose, "ear_left"), _point(pose, "ear_right")
    head_x = np.nanmean(np.stack([nose[0], left[0], right[0]]), axis=0)
    head_y = np.nanmean(np.stack([nose[1], left[1], right[1]]), axis=0)

    length = np.nanmedian(np.hypot(nose[0] - tail[0], nose[1] - tail[1]))
    head_half = max(4, int(round(HEAD_BOX * length)))
    body_half = max(6, int(round(BODY_BOX * length)))
    silhouette_half = max(8, int(round(SILHOUETTE_BOX * length)))
    kernel = np.ones((3, 3), np.uint8)

    out = np.full((len(pose), len(COLUMNS)), np.nan)
    previous = None
    for row in range(len(pose)):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)

        if previous is not None:
            motion = np.abs(gray - previous)
            head = _box(motion, head_x[row], head_y[row], head_half)
            trunk = _box(motion, body[0][row], body[1][row], body_half)
            if head.size and trunk.size:
                out[row, 0] = head.mean()
                out[row, 1] = trunk.mean()
                out[row, 2] = head.mean() / (trunk.mean() + 1.0)
        previous = gray

        foreground = (np.abs(gray - plate) > FOREGROUND_THRESHOLD).astype(np.uint8)
        crop = _box(foreground, body[0][row], body[1][row], silhouette_half)
        if crop.size:
            crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)
            count, _, stats, centroids = cv2.connectedComponentsWithStats(crop)
            if count > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                width = stats[largest, cv2.CC_STAT_WIDTH]
                height = stats[largest, cv2.CC_STAT_HEIGHT]
                top = max(0, int(body[1][row]) - silhouette_half)
                out[row, 3] = stats[largest, cv2.CC_STAT_AREA]
                out[row, 4] = height
                out[row, 5] = width
                out[row, 6] = height / max(1, width)
                out[row, 7] = (top + centroids[largest][1] - body[1][row]) / length
    capture.release()

    signals = pd.DataFrame(out, columns=list(COLUMNS))
    # Relativo à sessão: iluminação e tamanho do animal saem da conta.
    for column in ("motion_head", "motion_body", "blob_area", "blob_height", "blob_width"):
        typical = np.nanmedian(signals[column])
        signals[column] = signals[column] / typical if typical > 0 else np.nan
    return signals


def pose_rise_signals(pose: pd.DataFrame, kinematics: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Geometria do levantar que a pose já tem, mas o modelo não via.

    As distâncias entre pontos não têm sinal nem direção. Levantar sobe a cabeça
    na imagem em relação ao dorso — uma direção, não uma distância — e encurta o
    corpo aparente por um instante, em relação ao próprio animal alguns segundos
    antes, e não à mediana da sessão inteira.
    """
    neck_x, neck_y = kinematics["posx_neck"].to_numpy(float), kinematics["posy_neck"].to_numpy(float)
    nose_y = kinematics["posy_nose"].to_numpy(float)
    body_y = kinematics["posy_body"].to_numpy(float)
    tail_x, tail_y = kinematics["posx_tail_base"].to_numpy(float), kinematics["posy_tail_base"].to_numpy(float)

    length = np.hypot(neck_x - tail_x, neck_y - tail_y)
    typical = np.nanmedian(length)
    local = pd.Series(length).rolling(int(round(5 * fps)), center=True, min_periods=int(fps)).median()
    return pd.DataFrame({
        "neck_rel_y": (neck_y - body_y) / typical,
        "nose_rel_y": (nose_y - body_y) / typical,
        "length_local": length / local.to_numpy(),
    })
