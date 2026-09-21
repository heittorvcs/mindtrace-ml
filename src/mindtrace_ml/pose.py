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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PoseModel:
    """Sessão ONNX de um modelo DeepLabCut, em qualquer um dos dois formatos.

    O modelo original do MindTrace veio de TensorFlow via tf2onnx: saída **NHWC**
    `(1, H, W, K)` e entrada RGB bruta em 0-255, com a subtração de média embutida
    no grafo. Os modelos retreinados no DLC 3.x vêm de PyTorch: saída **NCHW**
    `(1, K, H, W)` e entrada normalizada pelas estatísticas do ImageNet, que ficam
    *fora* do grafo. Alimentar um com a convenção do outro não gera erro — gera
    coordenadas erradas em silêncio.

    O layout é detectado pela forma da saída; a normalização segue dele, já que as
    duas convenções vêm emparelhadas na prática.
    """

    def __init__(self, model_path: str | Path, providers=None, normalize: bool | None = None):
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]
        self.has_locref = len(outputs) >= 2

        shape = [int(d) for d in outputs[0].shape]
        # O eixo dos pontos é sempre o menor: são 2 a 8 pontos contra dezenas de
        # células de mapa em cada dimensão espacial.
        self.channels_first = shape[1] < shape[3]
        if self.channels_first:
            _, self.n_keypoints, self.heat_rows, self.heat_cols = shape
        else:
            _, self.heat_rows, self.heat_cols, self.n_keypoints = shape

        self.normalize = self.channels_first if normalize is None else normalize

        # O mapa de confiança NÃO cobre a imagem inteira: a rede preenche embaixo
        # e à direita até um múltiplo do stride, e a origem segue no canto
        # superior esquerdo. Então a conversão é `(célula + 0,5) * stride`, sem
        # deslocamento — e não `tamanho / número de células`, que estica o mapa
        # sobre a imagem e desloca os pontos progressivamente em direção à borda.
        #
        # Medido contra o centro do animal segmentado por subtração de fundo: a
        # escala por divisão errava 26,7 px no cspnext_s e 10,1 px no resnet_50;
        # o stride erra 4,1 e 4,5 px, que é o resíduo esperado entre o centroide
        # da silhueta e o ponto do dorso.
        self.stride = 2 ** round(np.log2(MODEL_WIDTH / self.heat_cols))
        stride_y = 2 ** round(np.log2(MODEL_HEIGHT / self.heat_rows))
        if stride_y != self.stride:
            raise ValueError(
                f"stride inconsistente entre eixos ({self.stride} x {stride_y}) "
                f"para mapa {self.heat_rows}x{self.heat_cols} — modelo inesperado"
            )

    def _prepare(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.shape[:2] != (MODEL_HEIGHT, MODEL_WIDTH):
            crop_bgr = cv2.resize(
                crop_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR
            )

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.normalize:
            rgb = (rgb / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        if self.channels_first:
            return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis, ...])
        return rgb[np.newaxis, ...]

    def infer(self, crop_bgr: np.ndarray) -> list[Keypoint]:
        """Recebe um recorte BGR do OpenCV e devolve um ponto por canal."""
        outputs = self.session.run(self.output_names, {self.input_name: self._prepare(crop_bgr)})

        scoremap = outputs[0][0]
        locref = outputs[1][0] if self.has_locref else None
        if self.channels_first:
            scoremap = scoremap.transpose(1, 2, 0)
            if locref is not None:
                locref = locref.transpose(1, 2, 0)

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
                x=(peak_col + 0.5) * self.stride + offset_x,
                y=(peak_row + 0.5) * self.stride + offset_y,
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
                    # Grava coordenada e confiança sempre. O limiar é decisão de
                    # análise, e sua escala depende do modelo: o antigo saturava
                    # perto de 1,0 por sigmoide, enquanto os do DLC 3.x regridem
                    # uma gaussiana e ficam na faixa de 0,4-0,8. Aplicar o corte
                    # aqui congelaria essa escolha dentro de 34 CSVs.
                    row[f"{name}_x"] = round(point.x, 3)
                    row[f"{name}_y"] = round(point.y, 3)
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
