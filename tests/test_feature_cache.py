import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_behavior_model import feature_key  # noqa: E402

OBJECTS = pd.DataFrame([{"session_id": "TT_1_cam1", "object_id": 1, "center_x": 100.0, "center_y": 100.0,
                         "radius": 10.0}])
FLOOR = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)


def test_key_changes_with_every_input(tmp_path):
    pose = tmp_path / "TT_1_cam1.csv"
    pose.write_text("frame,nose_x\n0,1\n")
    base = feature_key([pose], OBJECTS, FLOOR, 29.97)

    assert feature_key([pose], OBJECTS, FLOOR, 29.97) == base
    assert feature_key([pose], OBJECTS.assign(radius=12.0), FLOOR, 29.97) != base
    assert feature_key([pose], OBJECTS, FLOOR + 1, 29.97) != base
    assert feature_key([pose], OBJECTS, None, 29.97) != base
    assert feature_key([pose], OBJECTS, FLOOR, 30.0) != base
    pose.write_text("frame,nose_x\n0,2\n")
    assert feature_key([pose], OBJECTS, FLOOR, 29.97) != base
