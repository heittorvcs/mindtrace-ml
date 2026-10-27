import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindtrace_ml.behavior_model import ContextModel, score_context  # noqa: E402


def session(seed: int, n: int = 400):
    """Uma sessão sintética: o comportamento vem em blocos, e a feature é ruidosa."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, bool)
    for start in rng.choice(n - 20, 8, replace=False):
        y[start:start + 12] = True
    X = pd.DataFrame({"a": y + rng.normal(0, 0.8, n), "b": rng.normal(0, 1, n)})
    return X, y


class TestScoreContext:
    def test_neighbours_are_shifted_scores(self):
        out = score_context(np.arange(30, dtype=float))

        assert out.ctx_self[10] == 10 and out.ctx_prev1[10] == 9 and out.ctx_next12[10] == 22
        assert np.isnan(out.ctx_prev1[0]) and np.isnan(out.ctx_next1[29])

    def test_rolling_summaries_are_centered(self):
        scores = np.zeros(30)
        scores[15] = 1.0

        out = score_context(scores)

        # Um pico na janela 15 aparece no máximo de ±4 janelas, e não além.
        assert out.ctx_max9[11] == 1 and out.ctx_max9[19] == 1 and out.ctx_max9[10] == 0
        assert out.ctx_mean25[15] == pytest.approx(1 / 25)


class TestContextModel:
    def fitted(self):
        data = [session(seed) for seed in range(6)]
        return ContextModel(folds=3).fit_sessions([x for x, _ in data], [y for _, y in data],
                                                  groups=[0, 0, 1, 1, 2, 3])

    def test_one_score_per_window_in_video_order(self):
        model = self.fitted()
        X, y = session(99)
        X.index = X.index + 1000   # índice qualquer, como o de uma fatia

        proba = model.predict_proba(X)[:, 1]

        assert proba.shape == (len(X),)
        assert proba[y].mean() > proba[~y].mean()

    def test_unseen_windows_do_not_train(self):
        data = [session(seed) for seed in range(4)]
        seen = [np.zeros(400, bool) for _ in data]
        for mask in seen:
            mask[:200] = True
        # Rótulos trocados na metade não vista: se ela treinasse, o modelo inverteria.
        ys = [np.r_[y[:200], ~y[200:]] for _, y in data]

        model = ContextModel(folds=2).fit_sessions([x for x, _ in data], ys, [0, 1, 2, 3], seen)
        X, y = session(99)

        proba = model.predict_proba(X)[:, 1]
        assert proba[y].mean() > proba[~y].mean()

    def test_survives_pickling(self):
        model = self.fitted()
        X, _ = session(7)

        again = pickle.loads(pickle.dumps(model))

        assert np.allclose(again.predict_proba(X), model.predict_proba(X))

    def test_needs_two_animals(self):
        data = [session(seed) for seed in range(2)]
        with pytest.raises(ValueError):
            ContextModel().fit_sessions([x for x, _ in data], [y for _, y in data], groups=[5, 5])
