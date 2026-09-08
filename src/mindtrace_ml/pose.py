"""Inferência de pose direto do vídeo, sem passar pelo aplicativo Qt.

Reimplementa a decodificação de InferenceEngine::inferCrop() — argmax no mapa de
confiança mais refinamento locref — mas lendo o vídeo com OpenCV, quadro a quadro.
A diferença que importa: aqui `frame` é o índice real do quadro no vídeo e nada é
descartado, então uma anotação humana feita em tempo de vídeo alinha com a linha
correta do CSV. No MindTrace o índice conta quadros processados, que dependem da
velocidade da máquina.

Genérico no número de pontos: lê a contagem da forma de saída do modelo, então
serve tanto ao modelo atual de 2 pontos quanto a um retreino com mais.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .schema import (
    CONFIDENCE_THRESHOLD,
    LOCREF_STD,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    PEAK_FLOOR,
    STRIDE,
)

# Deslocamentos dos quadrantes do mosaico 2×2, na ordem usada por
# InferenceEngine::processJob(): superior-esquerdo, superior-direito,
# inferior-esquerdo, inferior-direito.
QUADRANT_ORDER = ((0, 0), (1, 0), (0, 1), (1, 1))


@dataclass
class Keypoint:
    x: float = -1.0
    y: float = -1.0
    p: float = 0.0

    @property
    def valid(self) -> bool:
        return self.p >= CONFIDENCE_THRESHOLD


class PoseModel:
    """Sessão ONNX do modelo DeepLabCut exportado."""

    def __init__(self, model_path: str | Path, providers=None):
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]

        scoremap_shape = outputs[0].shape
        self.n_keypoints = int(scoremap_shape[3])
        self.heat_rows = int(scoremap_shape[1])
        self.heat_cols = int(scoremap_shape[2])
        self.has_locref = len(outputs) >= 2

    def infer(self, crop_bgr: np.ndarray) -> list[Keypoint]:
        """Recebe um recorte BGR do OpenCV e devolve um ponto por canal."""
        if crop_bgr.shape[:2] != (MODEL_HEIGHT, MODEL_WIDTH):
            crop_bgr = cv2.resize(
                crop_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR
            )

        # O grafo faz a subtração de média internamente (nó Sub), então a entrada
        # é RGB bruto em 0–255, exatamente como o C++ alimenta.
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = rgb[np.newaxis, ...]

        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        scoremap = outputs[0][0]
        locref = outputs[1][0] if self.has_locref and len(outputs) >= 2 else None

        points = []
        for channel in range(self.n_keypoints):
            plane = scoremap[:, :, channel]
            peak_row, peak_col = np.unravel_index(int(np.argmax(plane)), plane.shape)
            peak_score = float(plane[peak_row, peak_col])

            if peak_score < PEAK_FLOOR:
                points.append(Keypoint())
                continue

            offset_x = offset_y = 0.0
            if locref is not None:
                offset_x = float(locref[peak_row, peak_col, channel * 2]) * LOCREF_STD
                offset_y = float(locref[peak_row, peak_col, channel * 2 + 1]) * LOCREF_STD

            points.append(Keypoint(
                x=(peak_col + 0.5) * STRIDE + offset_x,
                y=(peak_row + 0.5) * STRIDE + offset_y,
                p=peak_score,
            ))
        return points


def equalize(frame_bgr: np.ndarray) -> np.ndarray:
    """Equalização de histograma sobre a luminância.

    Medido em vídeos monocromáticos de baixo contraste: leva a detecção válida
    de 28% para 65% e a confiança mediana de 0,20 para 0,95. O modelo generaliza
    para outros laboratórios muito melhor quando as estatísticas de imagem são
    normalizadas antes da inferência.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)


def crop_roi(frame_bgr: np.ndarray, roi) -> np.ndarray:
    """Recorta a região da arena. roi = (x0, y0, x1, y1) ou None."""
    if roi is None:
        return frame_bgr
    x0, y0, x1, y1 = roi
    return frame_bgr[y0:y1, x0:x1]


def crop_field(frame_bgr: np.ndarray, field: int, layout: str) -> np.ndarray:
    """Recorta o campo do quadro. layout 'full' usa o quadro inteiro."""
    if layout == "full":
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    half_width, half_height = width // 2, height // 2
    column, row = QUADRANT_ORDER[field]
    return frame_bgr[
        row * half_height:(row + 1) * half_height,
        column * half_width:(column + 1) * half_width,
    ]


def extract_video(video_path, model: PoseModel, fields=(0,), layout="mosaic2x2",
                  every: int = 1, max_frames: int | None = None, progress=None,
                  roi=None, use_equalize: bool = False, keypoints=None):
    """Percorre o vídeo e devolve uma linha por (quadro, campo).

    `every` amostra 1 a cada N quadros — útil para uma verificação rápida antes
    de rodar a sessão inteira. Os índices continuam sendo os do vídeo.
    """
    names = tuple(keypoints) if keypoints else tuple(f"kp{i}" for i in range(model.n_keypoints))
    if len(names) != model.n_keypoints:
        raise ValueError(
            f"o modelo tem {model.n_keypoints} pontos, mas {len(names)} nomes foram dados: {names}"
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"não foi possível abrir o vídeo: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rows = []
    frame_index = 0
    processed = 0

    while True:
        time_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame_bgr = capture.read()
        if not ok:
            break

        if frame_index % every == 0:
            prepared = crop_roi(frame_bgr, roi)
            if use_equalize:
                prepared = equalize(prepared)

            for field in fields:
                points = model.infer(crop_field(prepared, field, layout))
                row = {"frame": frame_index, "time_ms": round(time_ms, 2), "field": field}
                for name, point in zip(names, points):
                    row[f"{name}_x"] = round(point.x, 3) if point.valid else np.nan
                    row[f"{name}_y"] = round(point.y, 3) if point.valid else np.nan
                    row[f"{name}_p"] = round(point.p, 4)
                rows.append(row)

            processed += 1
            if progress and processed % 50 == 0:
                progress(frame_index, total)

            if max_frames is not None and processed >= max_frames:
                break

        frame_index += 1

    capture.release()
    return rows, {"fps": fps, "total_frames": total, "read_frames": frame_index}
