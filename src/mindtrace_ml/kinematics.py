"""Features cinemáticas calculadas a partir da pose, em unidades de tempo real.

Três diferenças em relação ao BehaviorScanner do MindTrace:

1. **Unidades por segundo, não por quadro.** Os limiares do C++ são em px/quadro
   com 30 fps fixo em código. Os vídeos do arquivo NOR são 25 fps, então lá cada
   quadro cobre 20% mais deslocamento e todo limiar fica enviesado.

2. **Variância dentro da janela.** O C++ só calcula média e soma. Mas grooming é,
   por definição, movimento de focinho de alta variância com corpo parado — a
   grandeza discriminante não estava sendo computada.

3. **Janelas centradas.** Processamento offline pode olhar para frente; o modo ao
   vivo não pode. Uma janela centrada em t usa contexto dos dois lados e é
   estritamente mais informativa que a janela acumulada do C++.

Genérico no número de pontos: funciona com os 2 atuais e com o conjunto de 6 do
retreino, sem alteração.
"""

from itertools import combinations

import numpy as np
import pandas as pd

DEFAULT_WINDOWS_SEC = (0.2, 0.4, 1.0, 2.0)
DEFAULT_STATS = ("mean", "std", "min", "max")


def frame_kinematics(pose: pd.DataFrame, keypoints, fps: float) -> pd.DataFrame:
    """Velocidade de cada ponto e distância entre cada par, por quadro.

    Velocidade em px/s. Quadros com pose inválida entram como NaN e propagam —
    nunca como zero, que o modelo leria como "animal parado".
    """
    if fps <= 0:
        raise ValueError(f"fps deve ser positivo, recebeu {fps}")

    out = pd.DataFrame(index=pose.index)
    out["frame"] = pose["frame"].to_numpy()
    out["time_ms"] = pose["time_ms"].to_numpy()

    for name in keypoints:
        x = pose[f"{name}_x"].to_numpy(dtype=float)
        y = pose[f"{name}_y"].to_numpy(dtype=float)

        dx = np.diff(x, prepend=np.nan)
        dy = np.diff(y, prepend=np.nan)
        out[f"speed_{name}"] = np.hypot(dx, dy) * fps

        # A posição entra como feature para que o desvio-padrão móvel meça a
        # dispersão espacial. Velocidade sozinha não separa grooming de walking:
        # focinho oscilando no lugar e focinho avançando em linha reta têm o
        # mesmo módulo de velocidade. O que difere é a área percorrida.
        out[f"posx_{name}"] = x
        out[f"posy_{name}"] = y

    for first, second in combinations(keypoints, 2):
        dx = pose[f"{first}_x"].to_numpy(dtype=float) - pose[f"{second}_x"].to_numpy(dtype=float)
        dy = pose[f"{first}_y"].to_numpy(dtype=float) - pose[f"{second}_y"].to_numpy(dtype=float)
        out[f"dist_{first}_{second}"] = np.hypot(dx, dy)

    return out


def window_features(base: pd.DataFrame, fps: float,
                    windows_sec=DEFAULT_WINDOWS_SEC, stats=DEFAULT_STATS) -> pd.DataFrame:
    """Estatísticas móveis centradas sobre cada feature de quadro."""
    value_columns = [c for c in base.columns if c not in ("frame", "time_ms")]
    out = base[["frame", "time_ms"]].copy()

    for seconds in windows_sec:
        size = max(1, int(round(seconds * fps)))
        rolling = base[value_columns].rolling(window=size, center=True, min_periods=1)
        label = str(seconds).replace(".", "_")

        for statistic in stats:
            computed = getattr(rolling, statistic)()
            computed.columns = [f"{c}_w{label}_{statistic}" for c in value_columns]
            out = pd.concat([out, computed], axis=1)

    return out


def build_features(pose: pd.DataFrame, keypoints, fps: float,
                   windows_sec=DEFAULT_WINDOWS_SEC, stats=DEFAULT_STATS) -> pd.DataFrame:
    """Pipeline completo: pose bruta → features prontas para o classificador."""
    return window_features(frame_kinematics(pose, keypoints, fps), fps, windows_sec, stats)


def coverage_by_keypoint(pose: pd.DataFrame, keypoints, threshold: float = 0.75) -> pd.DataFrame:
    """Fração de quadros em que cada ponto foi detectado com confiança suficiente.

    É o primeiro diagnóstico a rodar num vídeo novo: um ponto com cobertura baixa
    não sustenta nenhuma feature derivada dele.
    """
    rows = []
    for name in keypoints:
        probability = pose[f"{name}_p"].to_numpy(dtype=float)
        rows.append({
            "keypoint": name,
            "coverage": float(np.mean(probability >= threshold)),
            "median_p": float(np.median(probability)),
            "frames_lost": int(np.sum(probability < threshold)),
        })
    return pd.DataFrame(rows)
