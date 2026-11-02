"""Stage 4 Visual Annotation Engine: Color-Coded Bounding Boxes & Topological Scene Graph Overlay."""

from __future__ import annotations
import base64
from typing import Tuple
import cv2
import numpy as np
from api.schemas import AuditReport
from core.graph_builder import SpatialShelfGraph


class PlanogramVisualizer:
    """Renders annotated compliance audit images showing compliant items (green),
    misplaced/extra items (amber/yellow), missing OOS gaps (red dashed), and topological graph edges."""

    COLOR_COMPLIANT = (46, 204, 113)    # Green (BGR)
    COLOR_MISPLACED = (0, 191, 255)     # Amber/Gold (BGR)
    COLOR_MISSING = (60, 76, 231)       # Red (BGR)
    COLOR_EDGE_H = (235, 206, 135)      # Sky Blue horizontal graph edges
    COLOR_EDGE_V = (219, 112, 147)      # Violet vertical graph edges

    @staticmethod
    def _draw_dashed_rect(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_len: int = 10,
    ) -> None:
        x1, y1 = pt1
        x2, y2 = pt2
        for x in range(x1, x2, dash_len * 2):
            cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
        for y in range(y1, y2, dash_len * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)

    def render_audit(
        self,
        image_bgr: np.ndarray,
        spatial_graph: SpatialShelfGraph,
        audit_report: AuditReport,
        draw_topology: bool = True,
    ) -> np.ndarray:
        canvas = image_bgr.copy()

        # Collect misplaced/extra bounding boxes from audit report
        misplaced_bboxes = set()
        for s_rep in audit_report.shelf_reports:
            for anom in s_rep.anomalies:
                if anom.type in ("MISPLACED_ITEM", "EXTRA_ITEM") and anom.detected_bbox:
                    misplaced_bboxes.add(tuple(anom.detected_bbox))

        # 1. Draw bounding boxes for all detected nodes
        for s_idx, nodes in spatial_graph.shelf_clusters.items():
            for node in nodes:
                b_int = tuple(int(round(v)) for v in node.bbox)
                is_bad = b_int in misplaced_bboxes
                color = self.COLOR_MISPLACED if is_bad else self.COLOR_COMPLIANT
                cv2.rectangle(
                    canvas, (b_int[0], b_int[1]), (b_int[2], b_int[3]), color, 2
                )
                label = f"{node.sku_id} ({'ERR' if is_bad else 'OK'})"
                cv2.rectangle(
                    canvas,
                    (b_int[0], max(0, b_int[1] - 20)),
                    (min(canvas.shape[1], b_int[0] + 115), b_int[1]),
                    color,
                    -1,
                )
                cv2.putText(
                    canvas,
                    label,
                    (b_int[0] + 3, max(14, b_int[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (15, 15, 15),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Draw red dashed boxes for missing / OOS items
        for s_rep in audit_report.shelf_reports:
            for anom in s_rep.anomalies:
                if anom.type == "MISSING_ITEM" and anom.estimated_gap_bbox:
                    gx1, gy1, gx2, gy2 = anom.estimated_gap_bbox
                    self._draw_dashed_rect(
                        canvas, (gx1, gy1), (gx2, gy2), self.COLOR_MISSING, 2
                    )
                    cv2.putText(
                        canvas,
                        f"OOS: {anom.expected_sku}",
                        (gx1 + 8, gy1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        self.COLOR_MISSING,
                        1,
                        cv2.LINE_AA,
                    )

        # 3. Draw topological graph nodes and edges
        if draw_topology and spatial_graph.graph is not None:
            for u, v, data in spatial_graph.graph.edges(data=True):
                u_node = spatial_graph.graph.nodes[u]
                v_node = spatial_graph.graph.nodes[v]
                if u_node.get("is_gap") or v_node.get("is_gap"):
                    continue
                pt_u = (int(round(u_node["xc"])), int(round(u_node["yc"])))
                pt_v = (int(round(v_node["xc"])), int(round(v_node["yc"])))
                rel = data.get("relation", "")
                if rel == "IMMEDIATE_RIGHT_OF":
                    cv2.line(canvas, pt_u, pt_v, self.COLOR_EDGE_H, 2, cv2.LINE_AA)
                elif rel == "ABOVE":
                    cv2.line(canvas, pt_u, pt_v, self.COLOR_EDGE_V, 1, cv2.LINE_AA)

            for _, n_data in spatial_graph.graph.nodes.items():
                if not n_data.get("is_gap"):
                    pt = (int(round(n_data["xc"])), int(round(n_data["yc"])))
                    cv2.circle(canvas, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 3, (30, 41, 59), -1, cv2.LINE_AA)

        # 4. Header banner with overall score
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (30, 41, 59), -1)
        banner_txt = (
            f"PlangoGraph Audit: {audit_report.planogram_id} | "
            f"Compliance: {audit_report.overall_compliance_score:.1f}% | "
            f"OK: {audit_report.summary.compliant_count} "
            f"OOS: {audit_report.summary.missing_count} "
            f"Misplaced: {audit_report.summary.misplaced_count} "
            f"GED: {audit_report.summary.graph_edit_distance:.1f}"
        )
        cv2.putText(
            canvas,
            banner_txt,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (248, 250, 252),
            1,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def encode_image_base64(image_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("utf-8")
