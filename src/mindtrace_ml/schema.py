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
