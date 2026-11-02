"""REST API Endpoints for PlangoGraph."""

from __future__ import annotations
import json
import pathlib
from typing import Dict, List
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from api.schemas import AuditReport, BoundingBoxDetection, PlanogramRef
from core.classifier import MetricSKUClassifier
from core.compliance_engine import PlanogramComplianceEngine
from core.detector import RetailObjectDetector
from core.graph_builder import SpatialGraphBuilder
from core.preprocessing import ShelfPreprocessor
from core.visualizer import PlanogramVisualizer

router = APIRouter(prefix="/api/v1", tags=["PlangoGraph Audit Service"])

PLANOGRAM_STORE: Dict[str, PlanogramRef] = {}
PREPROCESSOR = ShelfPreprocessor()
DETECTOR = RetailObjectDetector()
CLASSIFIER = MetricSKUClassifier()
GRAPH_BUILDER = SpatialGraphBuilder()
COMPLIANCE_ENGINE = PlanogramComplianceEngine()
VISUALIZER = PlanogramVisualizer()

# Load default reference planogram on startup if present
_DEFAULT_PLG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "planograms"
    / "planogram_ref.json"
)
if _DEFAULT_PLG_PATH.exists():
    _default_plg = PlanogramRef.model_validate_json(_DEFAULT_PLG_PATH.read_text())
    PLANOGRAM_STORE[_default_plg.planogram_id] = _default_plg


@router.post("/planograms", response_model=PlanogramRef)
def register_planogram(planogram: PlanogramRef) -> PlanogramRef:
    """Registers or updates a reference planogram blueprint in the system."""
    PLANOGRAM_STORE[planogram.planogram_id] = planogram
    return planogram


@router.get("/planograms/{planogram_id}", response_model=PlanogramRef)
def get_planogram(planogram_id: str) -> PlanogramRef:
    if planogram_id not in PLANOGRAM_STORE:
        raise HTTPException(status_code=404, detail=f"Planogram {planogram_id} not found")
    return PLANOGRAM_STORE[planogram_id]


@router.get("/skus")
def list_skus() -> Dict[str, List[Dict[str, str]]]:
    """Lists registered SKUs in the FAISS/Metric vector database."""
    return {"registered_skus": CLASSIFIER.list_registered_skus()}


@router.post("/audit/analyze", response_model=AuditReport)
async def analyze_shelf(
    planogram_id: str = Form(...),
    image: UploadFile = File(...),
    detections_json: str = Form(default=""),
) -> AuditReport:
    """Runs the 4-stage CV + Spatial Layout Graph compliance pipeline on an uploaded shelf image."""
    if planogram_id not in PLANOGRAM_STORE:
        raise HTTPException(
            status_code=404, detail=f"Reference planogram '{planogram_id}' not registered."
        )
    planogram = PLANOGRAM_STORE[planogram_id]

    raw_bytes = await image.read()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    norm_img, _, _ = PREPROCESSOR.preprocess_image(img_bgr, apply_warp=False)

    precomputed = None
    if detections_json.strip():
        raw_list = json.loads(detections_json)
        precomputed = [BoundingBoxDetection.model_validate(item) for item in raw_list]

    detections = DETECTOR.detect(norm_img, precomputed_boxes=precomputed)
    classified = CLASSIFIER.classify_detections(norm_img, detections)
    spatial_graph = GRAPH_BUILDER.build_graph(
        classified, expected_widths=planogram.expected_widths_map()
    )
    report = COMPLIANCE_ENGINE.evaluate(spatial_graph, planogram)
    annotated = VISUALIZER.render_audit(norm_img, spatial_graph, report)
    report.annotated_image_base64 = VISUALIZER.encode_image_base64(annotated)
    return report
