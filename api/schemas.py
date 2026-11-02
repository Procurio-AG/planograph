"""Pydantic v2 data schemas and API contracts for PlangoGraph."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

class ExpectedSlot(BaseModel):
    slot: int = Field(default=1, ge=1, description="1-based horizontal slot index on the shelf")
    sku_id: str = Field(..., description="Unique SKU identifier (e.g., SKU_001)")
    name: str = Field(default="", description="Human-readable product name")
    facings: int = Field(default=1, ge=1, description="Number of adjacent identical facings expected")
    expected_width_px: Optional[float] = Field(
        default=110.0, description="Reference product width in pixels on normalized plane"
    )

    @property
    def position(self) -> int:
        return self.slot - 1


class VerticalRule(BaseModel):
    upper_sku: str = Field(..., description="SKU expected on the upper shelf")
    lower_sku: str = Field(..., description="SKU expected directly below on the lower shelf")
    rule_type: Literal["ABOVE", "BELOW"] = "ABOVE"

    @property
    def relation(self) -> str:
        return self.rule_type


class ReferenceShelf(BaseModel):
    shelf_index: int = Field(..., ge=0, description="0-based top-to-bottom shelf index")
    shelf_name: str = Field(..., description="Descriptive shelf name (e.g., Top Shelf)")
    expected_sequence: List[ExpectedSlot] = Field(default_factory=list)


class PlanogramRef(BaseModel):
    planogram_id: str = Field(..., description="Unique reference planogram ID")
    bay_number: int = Field(default=1, ge=1)
    total_shelves: int = Field(..., ge=1)
    shelves: List[ReferenceShelf]
    vertical_rules: List[VerticalRule] = Field(default_factory=list)

    def flatten_shelf(self, shelf_index: int) -> List[str]:
        for shelf in self.shelves:
            if shelf.shelf_index == shelf_index:
                seq: List[str] = []
                for item in shelf.expected_sequence:
                    seq.extend([item.sku_id] * item.facings)
                return seq
        return []

    def expected_widths_map(self) -> Dict[str, float]:
        widths: Dict[str, float] = {}
        for shelf in self.shelves:
            for item in shelf.expected_sequence:
                if item.expected_width_px:
                    widths[item.sku_id] = item.expected_width_px
        return widths


class BoundingBoxDetection(BaseModel):
    bbox: Tuple[float, float, float, float] = Field(
        ..., description="(xmin, ymin, xmax, ymax) in pixel coordinates"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_label: Literal["product", "empty_space", "price_tag"] = "product"
    sku_id: Optional[str] = None
    sku_similarity: Optional[float] = None


DetectionItem = BoundingBoxDetection


class Anomaly(BaseModel):
    type: Literal[
        "MISSING_ITEM",
        "MISPLACED_ITEM",
        "EXTRA_ITEM",
        "VERTICAL_RULE_VIOLATION",
    ]
    expected_sku: Optional[str] = None
    detected_sku: Optional[str] = None
    position_index: int = Field(..., ge=0)
    estimated_gap_bbox: Optional[List[int]] = None
    detected_bbox: Optional[List[int]] = None
    details: Optional[str] = None


class ShelfReport(BaseModel):
    shelf_index: int
    shelf_name: Optional[str] = None
    compliance_score: float = Field(..., ge=0.0, le=100.0)
    needleman_wunsch_score: float
    expected_flattened: List[str]
    detected_sequence: List[str]
    aligned_expected: List[str]
    aligned_detected: List[str]
    anomalies: List[Anomaly] = Field(default_factory=list)


class AuditSummary(BaseModel):
    total_expected_facings: int
    total_detected_facings: int
    compliant_count: int
    missing_count: int
    misplaced_count: int
    extra_count: int = 0
    graph_edit_distance: float = 0.0
    vertical_rule_compliance: float = 100.0


class AuditReport(BaseModel):
    audit_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    planogram_id: str
    overall_compliance_score: float
    estimated_tilt_deg: float = 0.0
    summary: AuditSummary
    shelf_reports: List[ShelfReport]
    annotated_image_base64: Optional[str] = None
