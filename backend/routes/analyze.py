import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.models.schema import AnalysisResponse, MitigationResponse, SimulationRequest
from backend.services.bias_detection import analyze_bias
from backend.services.mitigation import apply_mitigation, simulate_mitigation

router = APIRouter()


def _validate_string_list(payload: object, field_name: str) -> list[str]:
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail=f"'{field_name}' must be a JSON array of strings.")

    cleaned = [item.strip() for item in payload if isinstance(item, str) and item.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"'{field_name}' must include at least one value.")

    return cleaned


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_dataset(
    dataset: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_features: str = Form(...),
):
    try:
        sensitive = _validate_string_list(json.loads(sensitive_features), "sensitive_features")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid sensitive_features payload.") from exc

    try:
        return await analyze_bias(dataset, target_column, sensitive)
    except HTTPException:
        raise


@router.post("/mitigate", response_model=MitigationResponse)
async def mitigate_dataset(
    dataset: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_features: str = Form(...),
    fixes: str = Form(...),
):
    try:
        sensitive = _validate_string_list(json.loads(sensitive_features), "sensitive_features")
        selected_fixes = json.loads(fixes)
        if not isinstance(selected_fixes, dict):
            raise HTTPException(status_code=400, detail="'fixes' must be a JSON object.")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid mitigation payload.") from exc

    try:
        return await apply_mitigation(dataset, target_column, sensitive, selected_fixes)
    except HTTPException:
        raise


@router.post("/simulate")
def simulate(request: SimulationRequest):
    return simulate_mitigation(request)
