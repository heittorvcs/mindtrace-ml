"""Detectores geométricos das faixas de rotina da triagem.

As faixas de rotina — caminhando, movimento lento, congelamento e interação com
objeto — são definíveis por regras sobre coordenadas, sem modelo treinado e sem
anotação de comportamento. É isso que permite a primeira triagem existir antes de
qualquer rotulagem: o humano passa a validar e calibrar em vez de gerar dados do
zero.

Medido numa sessão real de 5,1 min: as quatro faixas cobrem 79,7% dos quadros,
deixando 1,0 min para revisão. A quarta faixa foi acrescentada depois de medir
que quase metade do que sobrava caía entre os limiares de congelamento e
caminhada — e acrescentá-la custou um detector, sem tocar nos demais.

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

    Os padrões saem da distribuição medida em 33 sessões do laboratório, com a
    decodificação corrigida: velocidade do corpo com mediana de 13,9 px/s e p90
    de 57,4.

    Medir mostrou que a **redução é insensível a estes limiares** — de 15/8 a
    20/12 ela fica entre 87,4% e 88,6%, porque caminhada, movimento lento e
    congelamento são todas rotina e mexer nos cortes apenas redistribui quadros
    entre elas. Então os limiares não devem ser escolhidos para maximizar
    redução: devem ser escolhidos para ficarem etologicamente corretos, e é a
    rotulagem de clipes que responde isso.
    """

    walking_speed: float = 17.0       # px/s do centro do corpo, já suavizado
    walking_min_sec: float = 0.5

    freezing_speed: float = 9.0       # px/s mediano entre os pontos, já suavizado
    freezing_min_sec: float = 1.0

    object_margin: float = 12.0       # px além da borda do objeto
    object_angle: float = 60.0        # graus entre a direção da cabeça e o objeto
    object_min_sec: float = 0.3
    low_activity_min_sec: float = 0.5

    # Cabeça ativa: pescoço se mexendo em relação ao dorso, com o dorso parado.
    active_head_speed: float = 12.0   # px/s do pescoço relativo ao dorso
    active_head_body_speed: float = 12.0   # px/s máximo do dorso
    active_head_min_sec: float = 0.5
    active_head_smooth_sec: float = 0.5

    # Suavização antes de limiarizar. Sem ela, a velocidade instantânea oscila e
    # quase nunca permanece acima do corte pelos quadros seguidos que a duração
    # mínima exige — o detector some mesmo com o limiar correto. Mediana em vez
    # de média porque rejeita picos de detecção sem arrastar a borda do bout.
    smooth_sec: float = 0.3


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


def pose_is_valid(kinematics: pd.DataFrame) -> np.ndarray:
    """Quadros em que ao menos um ponto foi detectado com confiança suficiente."""
    columns = [c for c in kinematics.columns if c.startswith("speed_")]
    return ~kinematics[columns].isna().all(axis=1).to_numpy()


def _smooth(values: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    """Mediana móvel centrada, ignorando NaN."""
    window = max(1, int(round(seconds * fps)))
    if window <= 1:
        return values
    return (pd.Series(values)
            .rolling(window, center=True, min_periods=1)
            .median()
            .to_numpy())


def detect_walking(kinematics: pd.DataFrame, fps: float, thresholds: Thresholds) -> np.ndarray:
    speed = _smooth(kinematics["speed_body"].to_numpy(float), fps, thresholds.smooth_sec)
    mask = np.nan_to_num(speed, nan=0.0) >= thresholds.walking_speed
    return _sustained(mask, int(round(thresholds.walking_min_sec * fps)))


def detect_freezing(kinematics: pd.DataFrame, fps: float, thresholds: Thresholds) -> np.ndarray:
    columns = [c for c in kinematics.columns if c.startswith("speed_")]
    speeds = kinematics[columns].to_numpy(float)

    # Quadro sem pose não é imobilidade — é ausência de informação.
    known = ~np.isnan(speeds).all(axis=1)

    # Mediana entre os pontos, não o máximo: o máximo é decidido pelo ponto mais
    # ruidoso — tipicamente o focinho — e basta um tremor nele para negar uma
    # imobilidade que todos os outros pontos confirmam.
    typical = np.full(len(speeds), np.nan)
    typical[known] = np.nanmedian(speeds[known], axis=1)
    typical = _smooth(typical, fps, thresholds.smooth_sec)

    mask = known & (np.nan_to_num(typical, nan=np.inf) <= thresholds.freezing_speed)
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


def detect_low_activity(kinematics: pd.DataFrame, fps: float, thresholds: Thresholds,
                        freezing: np.ndarray) -> np.ndarray:
    """Movimento lento no lugar: devagar demais para caminhar, ativo demais para congelar.

    Existe porque a medição mostrou que quase metade do que sobrava para revisão
    caía na faixa entre os dois limiares — ajustes posturais, farejar o chão,
    pausas breves. É rotina, mas **não é congelamento**: alargar o limiar de
    congelamento até cobri-la produziria um detector que reporta congelamento
    demais, e congelamento é variável com significado científico próprio.

    Absorve também a imobilidade curta que a duração mínima de 1 s rejeita — uma
    pausa de meio segundo não é freezing, mas também não é nada digno de revisão.
    """
    columns = [c for c in kinematics.columns if c.startswith("speed_")]
    speeds = kinematics[columns].to_numpy(float)
    known = ~np.isnan(speeds).all(axis=1)

    speed = _smooth(kinematics["speed_body"].to_numpy(float), fps, thresholds.smooth_sec)
    slow = np.nan_to_num(speed, nan=np.inf) < thresholds.walking_speed

    mask = known & slow & ~freezing
    return _sustained(mask, int(round(thresholds.low_activity_min_sec * fps)))


def detect_active_head(pose: pd.DataFrame, kinematics: pd.DataFrame, fps: float,
                    thresholds: Thresholds, object_interaction: np.ndarray) -> np.ndarray:
    """Cabeça trabalhando sobre um corpo parado, longe dos objetos.

    Não é um classificador de comportamento: é o critério que **desqualifica
    "parado"**. A primeira medição mostrou que 44% dos clipes que a faixa de
    movimento lento reivindicava continham algo notável — sobretudo grooming —, e
    o que eles tinham em comum era a cabeça se mexendo com o corpo parado. Esses
    trechos deixam de ser rotina e vão para "outros".

    Nos 181 clipes rotulados, a velocidade do pescoço relativa ao dorso tem
    mediana de 13,1 px/s no grooming contra 6,0 no animal parado (AUC 0,91). Usa o
    pescoço, não o focinho, porque o focinho some justamente nesses trechos —
    visível em 70% dos quadros contra 100% — enquanto o pescoço tem 97% de
    cobertura.

    Dispara também em parte dos clipes marcados como parado, muito provavelmente
    sniffing. É o custo aceito: mandar a mais para revisão custa segundos de vídeo;
    mandar a menos perde o evento.
    """
    # As posições de frame_kinematics já vêm filtradas por confiança (NaN abaixo
    # do limiar), então a posição relativa herda a validade dos dois pontos.
    rel_x = kinematics["posx_neck"].to_numpy(float) - kinematics["posx_body"].to_numpy(float)
    rel_y = kinematics["posy_neck"].to_numpy(float) - kinematics["posy_body"].to_numpy(float)
    head = np.hypot(np.diff(rel_x, prepend=np.nan), np.diff(rel_y, prepend=np.nan)) * fps

    head = _smooth(head, fps, thresholds.active_head_smooth_sec)
    body = _smooth(kinematics["speed_body"].to_numpy(float), fps, thresholds.active_head_smooth_sec)

    mask = ((np.nan_to_num(head, nan=0.0) >= thresholds.active_head_speed)
            & (np.nan_to_num(body, nan=np.inf) < thresholds.active_head_body_speed)
            & ~object_interaction)
    return _sustained(mask, int(round(thresholds.active_head_min_sec * fps)))


# Faixas de rotina: o que elas reconhecem pode ser pulado.
BEHAVIOR_ORDER = ("walking", "freezing", "low_activity", "object_interaction")

# Critérios que mandam o trecho para "outros" mesmo que uma faixa de rotina o
# reivindique — sinais de que ali acontece algo que a rotina não descreve.
REVIEW_FLAGS = ("active_head",)


def triage(pose: pd.DataFrame, kinematics: pd.DataFrame, objects: pd.DataFrame,
           fps: float, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Aplica os detectores e marca como 'review' o que nenhum reconheceu."""
    thresholds = thresholds or Thresholds()

    freezing = detect_freezing(kinematics, fps, thresholds)
    object_interaction = detect_object_interaction(pose, objects, fps, thresholds)
    active_head = detect_active_head(pose, kinematics, fps, thresholds, object_interaction)

    result = pd.DataFrame({
        "frame": pose["frame"].to_numpy(),
        "time_ms": pose["time_ms"].to_numpy(),
        "walking": detect_walking(kinematics, fps, thresholds),
        "freezing": freezing,
        # Cabeça ativa desqualifica "parado": esses trechos vão para "outros".
        "low_activity": detect_low_activity(kinematics, fps, thresholds, freezing) & ~active_head,
        "object_interaction": object_interaction,
        "active_head": active_head,
    })

    # Quadro sem pose não é comportamento indeterminado — é dado faltante, e
    # mandá-lo ao revisor lhe entrega um trecho onde não há o que ver. Separar os
    # dois também mantém honesta a taxa de redução: cobertura de pose ruim
    # deixaria de se disfarçar de triagem eficiente.
    result["unscorable"] = ~pose_is_valid(kinematics)
    unclaimed = ~result[list(BEHAVIOR_ORDER)].any(axis=1)
    flagged = result[list(REVIEW_FLAGS)].any(axis=1)
    result["review"] = (unclaimed | flagged) & ~result["unscorable"]
    return result


def review_segments(triaged: pd.DataFrame, max_gap: int = 2,
                    merge_gap_sec: float = 1.0, fps: float = 30.0) -> pd.DataFrame:
    """Trechos que vão para o revisor, do mais longo para o mais curto.

    Ordenar por duração põe primeiro o que rende mais informação por minuto
    assistido — um trecho de 20 s tende a conter um comportamento inteiro, um de
    meio segundo raramente contém algo nomeável.

    `merge_gap_sec` funde trechos separados por um intervalo curto de rotina.
    Sem isso a fila fica com dezenas de fragmentos de pouco mais de um segundo, e
    o revisor gasta mais tempo navegando entre clipes do que assistindo. Fundir
    aumenta um pouco o tempo total revisado e reduz muito o esforço real.
    """
    frames = triaged["frame"].to_numpy()
    bouts = mask_to_bouts(triaged["review"].to_numpy(bool), frames, max_gap)

    merge_gap = int(round(merge_gap_sec * fps))
    if merge_gap > 0 and bouts:
        merged = [list(bouts[0])]
        for start, end in bouts[1:]:
            if start - merged[-1][1] <= merge_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        bouts = [tuple(b) for b in merged]

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


def composition(triaged: pd.DataFrame) -> dict:
    """Como a sessão se reparte entre as faixas, a revisão e o dado faltante.

    Reportar `unscorable` à parte impede que cobertura de pose ruim seja lida
    como triagem eficiente: as duas reduzem o tempo de revisão, mas só uma delas
    é um bom sinal.
    """
    parts = {name: float(triaged[name].mean()) for name in BEHAVIOR_ORDER}
    parts["review"] = float(triaged["review"].mean())
    if "unscorable" in triaged:
        parts["unscorable"] = float(triaged["unscorable"].mean())
    parts["reduction"] = reduction_rate(triaged)
    return parts


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
