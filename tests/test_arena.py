import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.arena import arena_of, floor_distance, order_corners  # noqa: E402

# Um trapézio como o chão visto pela câmera oblíqua: a borda de cima, mais longe, é mais estreita.
FLOOR = np.array([[110.0, 20.0], [310.0, 20.0], [330.0, 200.0], [90.0, 200.0]])


class TestArena:
    def test_nor_and_habituation_are_different_arenas_on_the_same_camera(self):
        assert arena_of("TR_19_cam1") == arena_of("TT_5_cam1") == "NOR_cam1"
        assert arena_of("HD3_F6_cam2") == "HD_cam2"

    def test_corners_come_out_in_order_whatever_the_click_order(self):
        clicked = FLOOR[[2, 0, 3, 1]]

        assert np.array_equal(order_corners(clicked), FLOOR)


class TestFloorDistance:
    def test_positive_inside_negative_over_the_wall(self):
        d = floor_distance(np.array([210.0, 210.0]), np.array([110.0, 5.0]), FLOOR)

        assert d[0] == 90.0          # meio do chão: 90 px até a borda de cima e a de baixo
        assert d[1] == -15.0         # acima da borda de cima: projetado sobre a parede

    def test_slanted_side_wall_is_measured_square_to_it(self):
        # Na parede lateral inclinada do trapézio, a distância é perpendicular a ela, não em x.
        d = floor_distance(np.array([120.0]), np.array([110.0]), FLOOR)

        assert 19 < d[0] < 20

    def test_missing_point_stays_missing(self):
        assert np.isnan(floor_distance(np.array([np.nan]), np.array([50.0]), FLOOR)[0])


class TestFloorFor:
    def test_marked_floor_wins_and_habituation_falls_back_to_the_body_rectangle(self):
        import pandas as pd
        from mindtrace_ml.arena import floor_for

        pose = pd.DataFrame({"body_x": np.linspace(100, 200, 101), "body_y": np.linspace(50, 150, 101),
                             "body_p": np.full(101, 0.9)})

        assert np.array_equal(floor_for("TT_5_cam1", pose, {"NOR_cam1": FLOOR}), FLOOR)
        rectangle = floor_for("HD3_F6_cam2", pose, {"NOR_cam1": FLOOR})
        assert np.allclose(rectangle, [[101, 51], [199, 51], [199, 149], [101, 149]])


class TestWithFloor:
    def test_nose_over_the_wall_is_negative(self):
        import pandas as pd
        from mindtrace_ml.behavior_model import with_floor

        pose = pd.DataFrame({"nose_x": [210.0, 210.0], "nose_y": [110.0, 5.0], "nose_p": [0.9, 0.9],
                             "neck_x": [210.0, 210.0], "neck_y": [120.0, 30.0], "neck_p": [0.9, 0.1]})

        out = with_floor(pd.DataFrame({"wall_distance": [0.0, 0.0]}), pose, FLOOR)

        assert out.nose_floor.tolist() == [90.0, -15.0]
        assert out.wall_distance.iloc[0] == 80.0 and np.isnan(out.wall_distance.iloc[1])   # pescoço incerto: sem ponto


class TestModelFloor:
    def test_old_models_get_no_floor_new_ones_get_theirs(self, tmp_path):
        import pandas as pd
        from mindtrace_ml.arena import model_floor

        path = tmp_path / "arena_floor.csv"
        pd.DataFrame([{"arena": "NOR_cam1", "corner": i + 1, "x": x, "y": y} for i, (x, y) in enumerate(FLOOR)]).to_csv(path, index=False)
        pose = pd.DataFrame({"body_x": [150.0], "body_y": [100.0], "body_p": [0.9]})

        assert model_floor({"models": {}}, "TT_5_cam1", pose, path) is None        # v1–v8: sem a chave
        assert model_floor(None, "TT_5_cam1", pose, path) is None
        assert np.array_equal(model_floor({"floor": True}, "TT_5_cam1", pose, path), FLOOR)
