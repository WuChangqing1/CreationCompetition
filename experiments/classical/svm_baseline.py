"""SVM baseline with leakage-safe scaling inside a sklearn Pipeline."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_svm(config=None):
    config = dict(config or {})
    classifier = SVC(
        C=float(config.get("C", 1.0)),
        kernel=config.get("kernel", "rbf"),
        gamma=config.get("gamma", "scale"),
        class_weight=config.get("class_weight"),
        probability=True,
        random_state=int(config.get("seed", 3407)),
    )
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
