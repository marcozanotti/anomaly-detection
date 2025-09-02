import pandas as pd
from sklearn.metrics import roc_auc_score

def accuracy(
    df: pd.DataFrame,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Accuracy (per series).

    (TP + TN) / (TP + TN + FP + FN)
    """

    def compute_accuracy(g):
        out = {}
        for m in models:
            y_true = g[target_col]
            y_pred = g[m]
            correct = (y_true == y_pred).sum()
            total = len(g)
            out[m] = correct / total if total > 0 else 0.0
        return pd.Series(out)

    res = df.groupby(id_col, observed=True).apply(compute_accuracy, include_groups=False).reset_index()

    return res

def recall(
    df: pd.DataFrame,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    positive_label: int = 1,
) -> pd.DataFrame:
    """Recall (per series).

    TP / (TP + FN)
    """

    def compute_recall(g):
        out = {}
        for m in models:
            y_true = g[target_col]
            y_pred = g[m]
            tp = ((y_true == positive_label) & (y_pred == positive_label)).sum()
            fn = ((y_true == positive_label) & (y_pred != positive_label)).sum()
            out[m] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return pd.Series(out)

    res = df.groupby(id_col, observed=True).apply(compute_recall, include_groups=False).reset_index()

    return res

def precision(
    df: pd.DataFrame,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    positive_label: int = 1,
) -> pd.DataFrame:
    """Precision (per series).

    TP / (TP + FP)
    """

    def compute_precision(g):
        out = {}
        for m in models:
            y_true = g[target_col]
            y_pred = g[m]
            tp = ((y_true == positive_label) & (y_pred == positive_label)).sum()
            fp = ((y_true != positive_label) & (y_pred == positive_label)).sum()
            out[m] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return pd.Series(out)

    res = df.groupby(id_col, observed=True).apply(compute_precision, include_groups=False).reset_index()

    return res

def f1(
    df: pd.DataFrame,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    positive_label: int = 1,
) -> pd.DataFrame:
    """F1-score (per series).

    2 * (precision * recall) / (precision + recall)
    """

    def compute_f1(g):
        out = {}
        for m in models:
            y_true = g[target_col]
            y_pred = g[m]
            tp = ((y_true == positive_label) & (y_pred == positive_label)).sum()
            fp = ((y_true != positive_label) & (y_pred == positive_label)).sum()
            fn = ((y_true == positive_label) & (y_pred != positive_label)).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            out[m] = (
                2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            )
        return pd.Series(out)

    res = df.groupby(id_col, observed=True).apply(compute_f1, include_groups=False).reset_index()

    return res

def auc(
    df: pd.DataFrame,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """AUC (per series).

    Area Under the ROC Curve.
    """

    def compute_auc(g):
        out = {}
        for m in models:
            try:
                score = roc_auc_score(g[target_col], g[m])
            except ValueError:
                # Happens when only one class is present in y_true
                score = float("nan")
            out[m] = score
        return pd.Series(out)

    res = df.groupby(id_col, observed=True).apply(compute_auc, include_groups=False).reset_index()
    return res
