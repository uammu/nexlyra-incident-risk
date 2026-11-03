import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratedModel:
    """Wraps a fitted model and fixes its probabilities using held-out validation data."""

    def __init__(self, base, method: str = "sigmoid"):
        self.base, self.method = base, method

    def _raw(self, X):
        return self.base.predict_proba(X)[:, 1]

    def fit_calibrator(self, X_cal, y_cal):
        raw = self._raw(X_cal)
        if self.method == "isotonic":
            self.cal_ = IsotonicRegression(out_of_bounds="clip").fit(raw, y_cal)
        else:  # Platt scaling on logit scores (strictly monotonic -> ranking preserved)
            z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
            self.cal_ = LogisticRegression().fit(z.reshape(-1, 1), y_cal)
        return self

    def predict_proba(self, X):
        raw = self._raw(X)
        if self.method == "isotonic":
            p = self.cal_.predict(raw)
        else:
            r = np.clip(raw, 1e-6, 1 - 1e-6)
            p = self.cal_.predict_proba(np.log(r / (1 - r)).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])
