import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


def normalize_target(series: pd.Series) -> tuple[pd.Series, str | None, str | None]:
    cleaned = series.dropna()
    unique_values = cleaned.unique().tolist()

    if len(unique_values) < 2:
        raise ValueError("Target column must contain at least 2 classes for analysis.")

    if len(unique_values) == 2:
        if pd.api.types.is_numeric_dtype(cleaned):
            ordered = sorted(float(value) for value in unique_values)
            lower, upper = ordered[0], ordered[1]
            normalized = series.apply(
                lambda value: 1
                if pd.notna(value) and float(value) == upper
                else 0
                if pd.notna(value) and float(value) == lower
                else pd.NA
            )
            return normalized, None, str(upper)

        mapping = {}
        normalized_pairs = [
            ("yes", 1),
            ("approved", 1),
            ("true", 1),
            ("1", 1),
            ("no", 0),
            ("rejected", 0),
            ("false", 0),
            ("0", 0),
        ]

        for value in unique_values:
            lowered = str(value).strip().lower()
            mapped = next((target for token, target in normalized_pairs if lowered == token), None)
            if mapped is None:
                mapped = 1 if len(mapping) == 0 else 0
            mapping[value] = mapped

        positive_label = next((str(value) for value, mapped in mapping.items() if mapped == 1), None)
        return series.map(mapping), None, positive_label

    # For multi-class, map to integers 0,1,2,... sorted by unique values
    sorted_unique = sorted(unique_values, key=lambda x: (isinstance(x, str), x))
    mapping = {val: idx for idx, val in enumerate(sorted_unique)}
    normalized = series.map(mapping)
    return (
        normalized,
        f"Multi-class target with {len(unique_values)} classes mapped to 0-{len(unique_values)-1}",
        str(sorted_unique[-1]) if sorted_unique else None,
    )


def normalize_sensitive_features(df: pd.DataFrame, requested: list[str]) -> list[str]:
    lookup = {column.lower(): column for column in df.columns}
    resolved = []

    for feature in requested:
        matched = lookup.get(feature.lower())
        if matched:
            resolved.append(matched)

    return resolved


def bucket_sensitive_series(sensitive) -> pd.Series:
    if isinstance(sensitive, pd.DataFrame):
        binned_df = sensitive.copy()
        for column in binned_df.columns:
            col_data = binned_df[column]
            if pd.api.types.is_numeric_dtype(col_data.dropna()) and col_data.dropna().nunique() > 6:
                binned_df[column] = pd.qcut(
                    col_data.dropna(),
                    q=min(4, col_data.dropna().nunique()),
                    duplicates="drop",
                ).astype(str)
            else:
                binned_df[column] = col_data.fillna("<missing>").astype(str)

        collapsed = binned_df.astype(str).agg(" | ".join, axis=1)
        return collapsed.astype(str)

    cleaned = sensitive.dropna()

    if cleaned.empty:
        return sensitive.astype(str)

    if pd.api.types.is_numeric_dtype(cleaned) and cleaned.nunique() > 6:
        binned = pd.qcut(cleaned, q=min(4, cleaned.nunique()), duplicates="drop")
        labelled = sensitive.copy()
        labelled.loc[cleaned.index] = binned.astype(str)
        return labelled.astype(str)

    return sensitive.astype(str)


def build_group_metrics(predictions, sensitive_series, y_true) -> dict[str, float]:
    bucketed = bucket_sensitive_series(sensitive_series)

    frame = pd.DataFrame({"y_true": y_true, "y_pred": predictions, "group": bucketed})
    raw_rates = frame.groupby("group")["y_pred"].mean().mul(100).round(2).to_dict()
    rates = {str(key): float(value) for key, value in raw_rates.items()}

    return dict(sorted(rates.items(), key=lambda item: item[1], reverse=True))


def _format_feature_list(sensitive_name: str) -> str:
    parts = [part.strip() for part in sensitive_name.split(",") if part.strip()]
    pretty_parts = [part.replace("_", " ").title() for part in parts]

    if not pretty_parts:
        return "Selected groups"
    if len(pretty_parts) == 1:
        return pretty_parts[0]
    if len(pretty_parts) == 2:
        return f"{pretty_parts[0]} and {pretty_parts[1]}"
    return f"{', '.join(pretty_parts[:-1])}, and {pretty_parts[-1]}"


def build_insights_and_alerts(
    bias_score: float,
    group_metrics: dict[str, float],
    sensitive_name: str,
    outcome_label: str,
):
    groups = list(group_metrics.keys())
    first_group = groups[0] if groups else "Group A"
    second_group = groups[1] if len(groups) > 1 else "Group B"
    values = list(group_metrics.values())
    gap = max(values) - min(values) if values else 0
    pretty_sensitive_name = _format_feature_list(sensitive_name)
    is_intersectional = "," in sensitive_name

    if is_intersectional:
        insights = [
            f"{pretty_sensitive_name} appear to influence predictions through correlated feature patterns.",
            f"Among the selected intersectional groups, {first_group} has a higher {outcome_label} rate than {second_group} by about {gap:.1f} percentage points.",
            "Model fairness should be reviewed before deployment, especially for combined demographic groups.",
        ]
        headline = "Model outcomes vary across intersectional groups"
        bias_label = "intersectional group"
    else:
        insights = [
            f"{pretty_sensitive_name} appears to influence predictions through correlated feature patterns.",
            f"{first_group} has a higher {outcome_label} rate than {second_group} by about {gap:.1f} percentage points.",
            "Model fairness should be reviewed before deployment.",
        ]
        headline = f"Model outcomes vary across {pretty_sensitive_name.lower()} groups"
        bias_label = pretty_sensitive_name.lower()

    alerts = []
    if bias_score >= 0.2:
        alerts.append(f"Potential {bias_label} bias detected.")
    if gap >= 10:
        alerts.append(f"{outcome_label.capitalize()} gap exceeds a practical fairness threshold.")
    if not alerts:
        alerts.append("No major fairness alert triggered, but continue monitoring.")

    error_text = (
        f"Predicted {outcome_label} rates differ by {gap:.1f} percentage points across the selected groups."
    )

    return insights, alerts, headline, error_text


def build_recommendation(
    bias_score: float,
    sensitive_features: list[str],
    group_metrics: dict[str, float],
) -> str:
    feature_label = _format_feature_list(", ".join(sensitive_features))
    group_names = list(group_metrics.keys())
    highest_group = group_names[0] if group_names else "the highest-rate group"
    lowest_group = group_names[-1] if len(group_names) > 1 else "the lowest-rate group"

    if bias_score >= 0.25:
        return (
            f"Prioritize fairness-constrained retraining and dataset rebalancing for {feature_label}; "
            f"the widest gap appears between {highest_group} and {lowest_group}."
        )
    if bias_score >= 0.12:
        return (
            f"Start with rebalancing and a targeted review of features correlated with {feature_label}, "
            f"then compare whether the gap between {highest_group} and {lowest_group} narrows."
        )
    return (
        f"Fairness risk is currently lower, but keep monitoring {feature_label} and validate that "
        f"{highest_group} and {lowest_group} remain within your acceptable threshold."
    )


def derive_severity(bias_score: float, group_gap: float) -> tuple[str, str]:
    combined = max(bias_score, group_gap / 100)

    if combined < 0.12:
        return "Low Bias", "Low"
    if combined < 0.25:
        return "Moderate Bias", "Moderate"
    return "High Bias", "High"


def rebalance_dataframe(df: pd.DataFrame, sensitive_columns: list[str] | str, target_column: str) -> pd.DataFrame:
    if isinstance(sensitive_columns, str):
        sensitive_columns = [sensitive_columns]

    grouped = []
    group_key = df[sensitive_columns].astype(str).agg(" | ".join, axis=1)
    max_size = group_key.value_counts().max()

    for _, group in df.groupby(group_key):
        grouped.append(
            resample(
                group,
                replace=True,
                n_samples=max_size,
                random_state=42,
            )
        )

    return pd.concat(grouped).sample(frac=1, random_state=42).reset_index(drop=True)
