import io

import numpy as np
import pandas as pd
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from fastapi import HTTPException, UploadFile
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.models.schema import MitigationResponse, SimulationRequest
from backend.services.bias_detection import build_analysis_response
from backend.utils.preprocessing import normalize_sensitive_features, normalize_target, rebalance_dataframe

if not hasattr(np, "PINF"):
    np.PINF = np.inf


async def apply_mitigation(
    dataset: UploadFile,
    target_column: str,
    sensitive_features: list[str],
    fixes: dict,
) -> MitigationResponse:
    df = pd.read_csv(io.BytesIO(await dataset.read()))
    original_sensitive = list(sensitive_features)
    usable_sensitive = normalize_sensitive_features(df, original_sensitive)
    usable_sensitive = [feature for feature in usable_sensitive if feature != target_column]
    if not usable_sensitive:
        raise HTTPException(
            status_code=400,
            detail="Please select at least one sensitive feature other than the target column.",
        )

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    if fixes.get("rebalance") and usable_sensitive:
        df = rebalance_dataframe(df, usable_sensitive, target_column)

    try:
        y, target_note, target_positive_label = normalize_target(df[target_column])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if y.isna().any():
        raise HTTPException(status_code=400, detail="Target column contains unsupported empty or mixed values.")

    excluded_features = usable_sensitive if fixes.get("sensitive") else []
    removable = [column for column in excluded_features if column in df.columns and column != target_column]
    X = df.drop(columns=[target_column, *removable], errors="ignore")
    sensitive_eval = df[usable_sensitive].copy()
    if X.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="No usable training features remain after preprocessing or mitigation.",
        )

    feature_columns = X.columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    base_estimator = LogisticRegression(max_iter=500, solver="liblinear")
    fairness_model = ExponentiatedGradient(estimator=base_estimator, constraints=DemographicParity())
    baseline_model = LogisticRegression(max_iter=500, solver="liblinear")

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X,
        y,
        sensitive_eval,
        test_size=0.25,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if sparse.issparse(X_train_processed):
        X_train_processed = X_train_processed.toarray()
    if sparse.issparse(X_test_processed):
        X_test_processed = X_test_processed.toarray()

    # Apply mitigation strategy based on selected fixes:
    # - constraint: fairness-constrained training (Fairlearn)
    # - otherwise: standard baseline training
    if fixes.get("constraint"):
        fairness_model.fit(X_train_processed, y_train, sensitive_features=sensitive_train)
        y_pred = fairness_model.predict(X_test_processed)
    else:
        baseline_model.fit(X_train_processed, y_train)
        y_pred = baseline_model.predict(X_test_processed)

    analysis = build_analysis_response(
        dataset_name=dataset.filename or "mitigated.csv",
        target_column=target_column,
        target_positive_label=target_positive_label,
        usable_sensitive=usable_sensitive,
        target_note=target_note,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_eval=sensitive_test,
    )

    return MitigationResponse(**analysis.model_dump())


def simulate_mitigation(request: SimulationRequest):
    has_constraint = bool(request.fixes.get("constraint"))
    has_rebalance = bool(request.fixes.get("rebalance"))
    has_sensitive = bool(request.fixes.get("sensitive"))

    delta = 0.0
    if has_sensitive:
        delta += 0.08
    if has_rebalance:
        delta += 0.06
    if has_constraint:
        delta += 0.12

    if delta == 0:
        bias_score = min(request.before_score + 0.03, 0.95)
    else:
        bias_score = max(request.before_score - delta, 0.01)

    return {
        "bias_score": round(bias_score, 2),
        "message": "Simulation complete",
    }
