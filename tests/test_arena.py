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
