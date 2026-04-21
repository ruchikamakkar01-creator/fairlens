import io
import json
import os
import urllib.error
import urllib.request

import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fastapi import HTTPException, UploadFile
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

from backend.models.schema import AnalysisResponse
from backend.utils.preprocessing import (
    build_group_metrics,
    build_insights_and_alerts,
    build_recommendation,
    derive_severity,
    normalize_target,
    normalize_sensitive_features,
)


def gemini_enabled() -> bool:
    if os.getenv("ENABLE_GEMINI_INSIGHTS", "true").lower() in {"0", "false", "no"}:
        return False

    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def call_gemini(prompt: str, max_output_tokens: int = 220) -> str | None:
    if not gemini_enabled():
        return None

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_output_tokens},
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw_response = response.read().decode("utf-8")
            parsed = json.loads(raw_response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    try:
        return parsed["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return None


def generate_gemini_insights(
    sensitive_name: str,
    bias_score: float,
    group_metrics: dict[str, float],
    target_column: str,
    target_positive_label: str | None,
) -> list[str] | None:
    groups_summary = ", ".join(f"{group}: {value:.1f}%" for group, value in group_metrics.items())
    positive_label = target_positive_label or "positive outcome"
    prompt = (
        "You are helping a responsible-AI hackathon team. "
        "Produce exactly 3 short bullet-style insights, each one sentence. "
        f"Target column: {target_column}. Positive outcome label: {positive_label}. "
        f"Sensitive features: {sensitive_name}. Bias score: {bias_score:.2f}. "
        f"Predicted {positive_label} rates: {groups_summary}. "
        "Focus on fairness patterns, practical interpretation, and plain English. Avoid legal advice."
    )
    text = call_gemini(prompt, max_output_tokens=180)
    if not text:
        return None

    lines = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
    cleaned = [line for line in lines if len(line) > 8]
    return cleaned[:3] if cleaned else None


def generate_gemini_recommendation(
    sensitive_name: str,
    bias_score: float,
    group_metrics: dict[str, float],
    target_column: str,
    target_positive_label: str | None,
) -> str | None:
    groups_summary = ", ".join(f"{group}: {value:.1f}%" for group, value in group_metrics.items())
    positive_label = target_positive_label or "positive outcome"
    prompt = (
        "You are advising a team building a fairness dashboard for a Google hackathon. "
        "Write one concise mitigation recommendation, 1 to 2 sentences max. "
        f"Target column: {target_column}. Positive outcome label: {positive_label}. "
        f"Sensitive features: {sensitive_name}. Bias score: {bias_score:.2f}. "
        f"Predicted {positive_label} rates: {groups_summary}. "
        "Recommend the most useful next step among rebalancing, removing sensitive features from inputs where appropriate, "
        "or applying a fairness constraint."
    )
    return call_gemini(prompt, max_output_tokens=120)


def build_analysis_response(
    *,
    dataset_name: str,
    target_column: str,
    target_positive_label: str | None,
    usable_sensitive: list[str],
    target_note: str | None,
    y_true,
    y_pred,
    sensitive_eval,
) -> AnalysisResponse:
    sensitive_label = ", ".join(usable_sensitive)
    group_metrics = build_group_metrics(y_pred, sensitive_eval, y_true)
    dp_diff = float(abs(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_eval)))
    is_binary = y_true.max() == 1
    if is_binary:
        eo_diff = float(abs(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_eval)))
    else:
        eo_diff = 0.0
    bias_score = dp_diff
    group_values = list(group_metrics.values())
    group_gap = max(group_values) - min(group_values) if group_values else 0
    performance = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
    }

    severity, badge = derive_severity(bias_score, group_gap)
    insights, alerts, headline, error_text = build_insights_and_alerts(
        bias_score=bias_score,
        group_metrics=group_metrics,
        sensitive_name=sensitive_label,
        outcome_label=(target_positive_label or target_column).replace("_", " ").lower(),
    )
    gemini_insights = generate_gemini_insights(
        sensitive_label,
        bias_score,
        group_metrics,
        target_column,
        target_positive_label,
    )
    recommendation = build_recommendation(bias_score, usable_sensitive, group_metrics)
    gemini_recommendation = generate_gemini_recommendation(
        sensitive_label,
        bias_score,
        group_metrics,
        target_column,
        target_positive_label,
    )
    insights_source = "gemini" if gemini_insights else "rule-based"
    recommendation_source = "gemini" if gemini_recommendation else "rule-based"

    return AnalysisResponse(
        dataset_name=dataset_name,
        target_column=target_column,
        target_positive_label=target_positive_label,
        sensitive_features=usable_sensitive,
        target_note=target_note,
        fairness_metric="Demographic Parity Difference",
        bias_score=round(bias_score, 2),
        demographic_parity_difference=round(dp_diff, 3),
        equalized_odds_difference=round(eo_diff, 3),
        severity=severity,
        badge=badge,
        performance=performance,
        group_metrics=group_metrics,
        insights=gemini_insights or insights,
        insights_source=insights_source,
        recommended_action=gemini_recommendation or recommendation,
        recommendation_source=recommendation_source,
        alerts=alerts,
        error_headline=headline,
        error_text=error_text,
    )


def analyze_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    sensitive_features: list[str],
    excluded_features: list[str] | None = None,
) -> AnalysisResponse:
    if len(df) < 8:
        raise HTTPException(
            status_code=400,
            detail="Dataset needs at least 8 rows for analysis.",
        )

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    usable_sensitive = normalize_sensitive_features(df, sensitive_features)
    usable_sensitive = [feature for feature in usable_sensitive if feature != target_column]
    if not usable_sensitive:
        raise HTTPException(
            status_code=400,
            detail="Please select at least one sensitive feature other than the target column.",
        )

    try:
        y, target_note, target_positive_label = normalize_target(df[target_column])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if y.isna().any():
        raise HTTPException(
            status_code=400,
            detail="Target column contains unsupported empty or mixed values.",
        )

    if y.nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Target column must have at least two classes.",
        )

    sensitive_eval = df[usable_sensitive].copy()
    excluded_features = excluded_features or []
    removable = [column for column in excluded_features if column in df.columns and column != target_column]
    X = df.drop(columns=[target_column, *removable], errors="ignore")

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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    try:
        X_train, X_test, y_train, y_test, _sensitive_train, sensitive_test = train_test_split(
            X,
            y,
            sensitive_eval,
            test_size=0.25,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_analysis_response(
        dataset_name=dataset_name,
        target_column=target_column,
        target_positive_label=target_positive_label,
        usable_sensitive=usable_sensitive,
        target_note=target_note,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_eval=sensitive_test,
    )


async def analyze_bias(
    dataset: UploadFile,
    target_column: str,
    sensitive_features: list[str],
) -> AnalysisResponse:
    df = pd.read_csv(io.BytesIO(await dataset.read()))
    return analyze_dataframe(df, dataset.filename or "uploaded.csv", target_column, sensitive_features)
