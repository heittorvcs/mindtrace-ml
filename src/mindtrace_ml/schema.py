"""Contrato de dados entre o MindTrace (Qt/C++) e este repositório."""

BEHAVIORS = ("walking", "sniffing", "grooming", "resting", "rearing")

UNSCORABLE = "unscorable"

# Índices emitidos por BehaviorScanner::classifySimple(), espelhados em
# behaviorNames no QML. A ordem é parte do contrato — não reordenar.
RULE_LABEL_TO_BEHAVIOR = {
    0: "walking",
    1: "sniffing",
    2: "grooming",
    3: "resting",
    4: "rearing",
}

# Cabeçalho de InferenceController::exportBehaviorFeatures(), na ordem exata.
FEATURE_COLUMNS = (
    "move_nose",
    "move_body",
    "bp_sum",
    "bp_mean",
    "bp_min",
    "bp_max",
    "roll2s_mean",
    "roll2s_sum",
    "roll5s_mean",
    "roll5s_sum",
    "roll6s_mean",
    "roll6s_sum",
    "roll7_5s_mean",
    "roll7_5s_sum",
    "roll15s_mean",
    "roll15s_sum",
    "prob_sum",
    "prob_mean",
    "low_prob_01",
    "low_prob_05",
    "low_prob_075",
)

FRAME_COLUMN = "frame"
RULE_LABEL_COLUMN = "rule_label"

LABEL_COLUMNS = (
    "session_id",
    "animal_id",
    "field",
    "behavior",
    "start_frame",
    "end_frame",
    "annotator",
)

MANIFEST_COLUMNS = (
    "session_id",
    "animal_id",
    "apparatus",
    "fps",
    "pose_model",
    "lighting",
)

# BehaviorScanner só grava o quadro quando ao menos um ponto tem p >= 0.75,
# então `frame` é descontínuo. Uma lacuna maior que isto encerra o bout.
DEFAULT_MAX_GAP = 2

# ── Pose extraída neste repositório ──────────────────────────────────────────
# Ao contrário do CSV do MindTrace, `frame` aqui é o índice real do quadro no
# vídeo e `time_ms` é o timestamp de apresentação: nenhum quadro é descartado e
# o resultado independe da velocidade da máquina.
POSE_INDEX_COLUMNS = ("frame", "time_ms", "field")

# Modelo atual (Network-MemoryLab-v2.onnx): canal 0 = focinho, canal 1 = corpo.
CURRENT_KEYPOINTS = ("nose", "body")

# Conjunto proposto para o retreino. Cada ponto justifica seu custo de anotação:
# as orelhas dão orientação da cabeça, a base da cauda dá o eixo do corpo (é o
# sinal de rearing, que encurta a projeção focinho–cauda), o pescoço dá curvatura.
PROPOSED_KEYPOINTS = ("nose", "ear_left", "ear_right", "neck", "body", "tail_base")


def pose_columns(keypoints):
    """Colunas do CSV de pose para um conjunto arbitrário de pontos."""
    columns = list(POSE_INDEX_COLUMNS)
    for name in keypoints:
        columns += [f"{name}_x", f"{name}_y", f"{name}_p"]
    return tuple(columns)

# Constantes de decodificação do modelo DeepLabCut, espelhadas de
# InferenceEngine (inference_engine.h). Alterar aqui exige alterar lá.
MODEL_WIDTH = 360
MODEL_HEIGHT = 240
STRIDE = 8.0
LOCREF_STD = 7.2801
PEAK_FLOOR = 0.05
CONFIDENCE_THRESHOLD = 0.75
