from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


def fit_tabular_surrogate(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train_frame[list(feature_columns)], train_frame[target_column])
    return model


def export_shap_report(
    model: RandomForestRegressor,
    train_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    output_path: str,
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_frame[list(feature_columns)])
    mean_abs = np.abs(shap_values).mean(axis=0)
    report = pd.DataFrame(
        {"feature": list(feature_columns), "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return report
