from typing import Dict, List

from pydantic import BaseModel


class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float


class AnalysisResponse(BaseModel):
    dataset_name: str
    target_column: str
    target_positive_label: str | None = None
    sensitive_features: List[str]
    target_note: str | None = None
    fairness_metric: str
    bias_score: float
    demographic_parity_difference: float
    equalized_odds_difference: float
    severity: str
    badge: str
    performance: PerformanceMetrics
    group_metrics: Dict[str, float]
    insights: List[str]
    insights_source: str | None = None
    recommended_action: str | None = None
    recommendation_source: str | None = None
    alerts: List[str]
    error_headline: str
    error_text: str


class MitigationResponse(AnalysisResponse):
    pass


class SimulationRequest(BaseModel):
    before_score: float
    fixes: Dict[str, bool]
