import io
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fastapi import HTTPException, UploadFile
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.models.schema import AnalysisResponse
from backend.utils.preprocessing import (
    build_group_metrics,
    build_insights_and_alerts,
    build_recommendation,
    derive_severity,
    normalize_sensitive_features,
    normalize_target,
)

logger = logging.getLogger("fairlens.gemini")


def gemini_enabled() -> bool:
    if os.getenv("ENABLE_GEMINI_INSIGHTS", "true").lower() in {"0", "false", "no"}:
        return False

    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]


def _clean_leading_label(text: str) -> str:
    cleaned = re.sub(r"^\s*(here are|below are|these are)\b[^:]*:\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(insight|recommendation)s?\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _is_complete_sentence(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 20 and stripped.endswith((".", "!", "?"))


def _normalize_sentence(text: str) -> str:
    cleaned = " ".join(_clean_leading_label(text).split())
    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned = f"{cleaned}."
    return cleaned


def _extract_json_object(text: str) -> dict | None:
    candidate = text.strip()

    # Remove optional markdown fences before parsing.
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()

    # Try direct JSON parsing first.
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            nested = json.loads(parsed)
            return nested if isinstance(nested, dict) else None
    except json.JSONDecodeError:
        pass

    # Fall back to extracting the first JSON object-looking block.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = candidate[start : end + 1]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            nested = json.loads(parsed)
            return nested if isinstance(nested, dict) else None
    except json.JSONDecodeError:
        return None

    return None


def _parse_labeled_narrative(text: str) -> tuple[list[str] | None, str | None]:
    insights: list[str] = []
    recommendation_parts: list[str] = []
    current_label: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith("INSIGHT_") and ":" in line:
            _, value = line.split(":", 1)
            sentence = _normalize_sentence(value)
            if _is_complete_sentence(sentence):
                insights.append(sentence)
            current_label = None
            continue

        if upper.startswith("RECOMMENDATION:"):
            _, value = line.split(":", 1)
            if value.strip():
                recommendation_parts.append(value.strip())
            current_label = "recommendation"
            continue

        if current_label == "recommendation":
            recommendation_parts.append(line)

    recommendation_text = _clean_leading_label(" ".join(recommendation_parts))
    recommendation_sentences = [_normalize_sentence(sentence) for sentence in _split_sentences(recommendation_text)]
    complete_recommendation = [sentence for sentence in recommendation_sentences if _is_complete_sentence(sentence)]

    if len(insights) < 2 or not complete_recommendation:
        return None, None

    return insights[:3], " ".join(complete_recommendation[:2])


def gemini_models() -> list[str]:
    configured = os.getenv("GEMINI_MODELS", "").strip()
    if configured:
        models = [model.strip() for model in configured.split(",") if model.strip()]
        if models:
            return models

    primary = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    fallbacks = ["gemini-2.5-flash-lite"]
    ordered = [primary, *fallbacks]
    seen = set()
    deduped = []
    for model in ordered:
        if model and model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def call_gemini(
    prompt: str,
    *,
    max_output_tokens: int = 220,
    response_schema: dict | None = None,
) -> tuple[str | None, str | None]:
    if not gemini_enabled():
        return None, "Gemini is disabled or no Google API key is configured."

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    models = gemini_models()
    last_error = "Gemini request failed."

    for model in models:
        attempts = 3
        for attempt in range(attempts):
            generation_config = {
                "temperature": 0.2,
                "maxOutputTokens": max_output_tokens,
            }
            if response_schema:
                generation_config["responseMimeType"] = "application/json"
                generation_config["responseJsonSchema"] = response_schema

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": generation_config,
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
                with urllib.request.urlopen(request, timeout=12) as response:
                    raw_response = response.read().decode("utf-8")
                    parsed = json.loads(raw_response)
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8")
                except Exception:
                    error_body = ""
                logger.warning("Gemini HTTP error %s on %s: %s", exc.code, model, error_body or exc.reason)
                last_error = f"Gemini request failed with HTTP {exc.code} on {model}."
                if exc.code in {429, 500, 503} and attempt < attempts - 1:
                    time.sleep(0.6 * (2**attempt))
                    continue
                break
            except urllib.error.URLError as exc:
                logger.warning("Gemini URL error on %s: %s", model, exc.reason)
                last_error = f"Gemini request could not reach Google AI services on {model}."
                if attempt < attempts - 1:
                    time.sleep(0.6 * (2**attempt))
                    continue
                break
            except TimeoutError:
                logger.warning("Gemini request timed out on %s.", model)
                last_error = f"Gemini request timed out on {model}."
                if attempt < attempts - 1:
                    time.sleep(0.6 * (2**attempt))
                    continue
                break
            except json.JSONDecodeError:
                logger.warning("Gemini returned invalid JSON on %s.", model)
                last_error = f"Gemini returned an unreadable response on {model}."
                break

            try:
                return parsed["candidates"][0]["content"]["parts"][0]["text"].strip(), None
            except (KeyError, IndexError, TypeError):
                logger.warning("Gemini response missing expected text on %s: %s", model, parsed)
                last_error = f"Gemini returned no usable text on {model}."
                break

    return None, last_error


def generate_gemini_narrative(
    sensitive_name: str,
    bias_score: float,
    group_metrics: dict[str, float],
    target_column: str,
    target_positive_label: str | None,
) -> tuple[list[str] | None, str | None, str | None]:
    groups_summary = ", ".join(f"{group}: {value:.1f}%" for group, value in group_metrics.items())
    positive_label = target_positive_label or "positive outcome"
    prompt = (
        "You are helping a responsible-AI hackathon team reviewing fairness results. "
        "Return plain text only in this exact format with no extra intro, markdown, code fences, or outro:\n"
        "INSIGHT_1: <complete sentence>\n"
        "INSIGHT_2: <complete sentence>\n"
        "INSIGHT_3: <complete sentence>\n"
        "RECOMMENDATION: <1 to 2 complete sentences>\n"
        f"Target column: {target_column}. Positive outcome label: {positive_label}. "
        f"Sensitive features: {sensitive_name}. Bias score: {bias_score:.2f}. "
        f"Predicted {positive_label} rates: {groups_summary}. "
        "The insights should focus on fairness patterns, practical interpretation, and plain English. "
        "The recommendation should suggest the strongest next step among rebalancing, removing sensitive features from inputs where appropriate, "
        "or applying a fairness constraint. Avoid legal advice."
    )
    text, error = call_gemini(prompt, max_output_tokens=280)
    if not text:
        return None, None, error

    parsed_insights, parsed_recommendation = _parse_labeled_narrative(text)
    if parsed_insights and parsed_recommendation:
        return parsed_insights, parsed_recommendation, None

    logger.warning("Gemini labeled output was invalid, retrying with simpler phrasing: %s", text)
    fallback_prompt = (
        "Respond with exactly four lines and nothing else.\n"
        "INSIGHT_1: one complete sentence.\n"
        "INSIGHT_2: one complete sentence.\n"
        "INSIGHT_3: one complete sentence.\n"
        "RECOMMENDATION: one or two complete sentences.\n"
        f"Target column: {target_column}. Positive outcome label: {positive_label}. "
        f"Sensitive features: {sensitive_name}. Bias score: {bias_score:.2f}. "
        f"Predicted {positive_label} rates: {groups_summary}."
    )
    fallback_text, fallback_error = call_gemini(fallback_prompt, max_output_tokens=220)
    if not fallback_text:
        return None, None, fallback_error or "Gemini returned invalid labeled output."

    parsed_insights, parsed_recommendation = _parse_labeled_narrative(fallback_text)
    if parsed_insights and parsed_recommendation:
        return parsed_insights, parsed_recommendation, None

    logger.warning("Gemini simpler labeled fallback was also invalid: %s", fallback_text)
    return None, None, "Gemini returned invalid labeled output."


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
    gemini_insights, gemini_recommendation, gemini_error = generate_gemini_narrative(
        sensitive_label,
        bias_score,
        group_metrics,
        target_column,
        target_positive_label,
    )
    recommendation = build_recommendation(bias_score, usable_sensitive, group_metrics)
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
        insights_status="ok" if gemini_insights else "fallback",
        insights_error=gemini_error,
        recommended_action=gemini_recommendation or recommendation,
        recommendation_source=recommendation_source,
        recommendation_status="ok" if gemini_recommendation else "fallback",
        recommendation_error=gemini_error,
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
