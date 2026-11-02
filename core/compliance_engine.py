"""Stage 4: Graph Alignment, Needleman-Wunsch Sequence Matching & Planogram Compliance Engine."""

from __future__ import annotations
import uuid
from typing import Dict, List, Optional, Tuple
import numpy as np
from api.schemas import (
    Anomaly,
    AuditReport,
    AuditSummary,
    PlanogramRef,
    ShelfReport,
)
from core.graph_builder import GraphNode, SpatialShelfGraph


class PlanogramComplianceEngine:
    """Aligns extracted spatial shelf graphs against a reference planogram blueprint using
    Needleman-Wunsch dynamic programming and 2D cross-shelf topological verification."""

    def __init__(
        self,
        match_reward: float = 2.0,
        mismatch_penalty: float = -1.6,
        gap_penalty: float = -1.0,
    ) -> None:
        self.match_reward = match_reward
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def needleman_wunsch(
        self,
        expected_seq: List[str],
        detected_seq: List[str],
    ) -> Tuple[List[str], List[str], List[Optional[int]], float]:
        """Runs global sequence alignment between expected SKU sequence and detected SKU sequence.
        Returns:
            aligned_expected: list of SKU strings or '-' for insertions
            aligned_detected: list of SKU strings or '-' for deletions (OOS gaps)
            detected_idx_map: index in detected_seq corresponding to each aligned step (or None)
            optimal_score: raw Needleman-Wunsch alignment score
        """
        n, m = len(expected_seq), len(detected_seq)
        dp = np.zeros((n + 1, m + 1), dtype=np.float64)
        ptr = np.zeros((n + 1, m + 1), dtype=int)  # 1: diag, 2: up (del), 3: left (ins)

        for i in range(1, n + 1):
            dp[i, 0] = i * self.gap_penalty
            ptr[i, 0] = 2
        for j in range(1, m + 1):
            dp[0, j] = j * self.gap_penalty
            ptr[0, j] = 3

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if expected_seq[i - 1] == detected_seq[j - 1]:
                    step_score = self.match_reward
                elif detected_seq[j - 1] == "__PHYSICAL_GAP__":
                    # Physical gap detected by Stage 3 aligns with a missing expected SKU
                    step_score = self.gap_penalty * 0.65
                else:
                    step_score = self.mismatch_penalty
                diag = dp[i - 1, j - 1] + step_score
                up = dp[i - 1, j] + self.gap_penalty
                left = dp[i, j - 1] + self.gap_penalty

                best = max(diag, up, left)
                dp[i, j] = best
                if best == diag:
                    ptr[i, j] = 1
                elif best == up:
                    ptr[i, j] = 2
                else:
                    ptr[i, j] = 3

        aligned_exp: List[str] = []
        aligned_det: List[str] = []
        det_indices: List[Optional[int]] = []

        i, j = n, m
        while i > 0 or j > 0:
            move = ptr[i, j]
            if i > 0 and j > 0 and move == 1:
                aligned_exp.append(expected_seq[i - 1])
                aligned_det.append(detected_seq[j - 1])
                det_indices.append(j - 1)
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or move == 2):
                aligned_exp.append(expected_seq[i - 1])
                aligned_det.append("-")
                det_indices.append(None)
                i -= 1
            else:
                aligned_exp.append("-")
                aligned_det.append(detected_seq[j - 1])
                det_indices.append(j - 1)
                j -= 1

        aligned_exp.reverse()
        aligned_det.reverse()
        det_indices.reverse()
        return aligned_exp, aligned_det, det_indices, float(dp[n, m])

    @staticmethod
    def _estimate_missing_bbox(
        aligned_pos: int,
        det_indices: List[Optional[int]],
        shelf_nodes: List[GraphNode],
        shelf_gaps: List[GraphNode],
        default_width: float = 110.0,
    ) -> List[int]:
        """Estimates the pixel bounding box [xmin, ymin, xmax, ymax] for an OOS / missing SKU."""
        if not shelf_nodes:
            x0 = int(80 + aligned_pos * default_width)
            return [x0, 100, int(x0 + default_width), 280]

        # Find nearest left and right detected neighbor in the aligned sequence
        left_det_idx: Optional[int] = None
        right_det_idx: Optional[int] = None
        for k in range(aligned_pos - 1, -1, -1):
            if det_indices[k] is not None:
                left_det_idx = det_indices[k]
                break
        for k in range(aligned_pos + 1, len(det_indices)):
            if det_indices[k] is not None:
                right_det_idx = det_indices[k]
                break

        # Check if there is an explicit physical gap node on this shelf
        if shelf_gaps:
            if left_det_idx is not None:
                for g_node in shelf_gaps:
                    if g_node.slot_index == left_det_idx:
                        return [int(round(v)) for v in g_node.bbox]
            # Otherwise pick the nearest physical gap node on the shelf
            best_g = min(shelf_gaps, key=lambda g: abs(g.slot_index - (left_det_idx or 0)))
            return [int(round(v)) for v in best_g.bbox]

        if left_det_idx is not None and right_det_idx is not None:
            ln = shelf_nodes[left_det_idx]
            rn = shelf_nodes[right_det_idx]
            xmin = int(round(ln.bbox[2]))
            xmax = int(round(max(ln.bbox[2] + 40.0, rn.bbox[0])))
            ymin = int(round(min(ln.bbox[1], rn.bbox[1])))
            ymax = int(round(max(ln.bbox[3], rn.bbox[3])))
            return [xmin, ymin, xmax, ymax]

        if left_det_idx is not None:
            ln = shelf_nodes[left_det_idx]
            xmin = int(round(ln.bbox[2] + 10.0))
            return [
                xmin,
                int(round(ln.bbox[1])),
                int(round(xmin + ln.width)),
                int(round(ln.bbox[3])),
            ]

        rn = shelf_nodes[right_det_idx if right_det_idx is not None else 0]
        xmax = int(round(max(20.0, rn.bbox[0] - 10.0)))
        xmin = int(round(max(0.0, xmax - rn.width)))
        return [xmin, int(round(rn.bbox[1])), xmax, int(round(rn.bbox[3]))]

    def evaluate(
        self,
        spatial_graph: SpatialShelfGraph,
        planogram: PlanogramRef,
        audit_id: Optional[str] = None,
    ) -> AuditReport:
        """Runs full planogram compliance evaluation across all shelves and vertical rules."""
        audit_id = audit_id or f"AUD_{uuid.uuid4().hex[:6].upper()}"
        shelf_reports: List[ShelfReport] = []

        total_expected = 0
        total_detected = 0
        total_compliant = 0
        total_missing = 0
        total_misplaced = 0
        total_extra = 0
        ged_cost = 0.0

        for ref_shelf in planogram.shelves:
            s_idx = ref_shelf.shelf_index
            expected_flat = planogram.flatten_shelf(s_idx)
            shelf_nodes = spatial_graph.shelf_clusters.get(s_idx, [])
            shelf_gaps = spatial_graph.gap_nodes.get(s_idx, [])
            detected_seq = [n.sku_id for n in shelf_nodes]

            # Build spatially-augmented sequence incorporating Stage 3 physical GAP nodes
            gap_map = {g.slot_index: g for g in shelf_gaps}
            aug_seq: List[str] = []
            aug_to_det_idx: List[Optional[int]] = []
            aug_gap_bbox: Dict[int, List[int]] = {}
            for d_i, node in enumerate(shelf_nodes):
                aug_seq.append(node.sku_id)
                aug_to_det_idx.append(d_i)
                if d_i in gap_map:
                    g_node = gap_map[d_i]
                    n_miss = max(1, min(3, g_node.estimated_missing_facings))
                    gx1, gy1, gx2, gy2 = g_node.bbox
                    gw = (gx2 - gx1) / n_miss
                    for m_i in range(n_miss):
                        aug_idx = len(aug_seq)
                        aug_seq.append("__PHYSICAL_GAP__")
                        aug_to_det_idx.append(None)
                        aug_gap_bbox[aug_idx] = [
                            int(round(gx1 + m_i * gw)),
                            int(round(gy1)),
                            int(round(gx1 + (m_i + 1) * gw)),
                            int(round(gy2)),
                        ]

            aligned_exp, aligned_aug, aug_indices, nw_score = self.needleman_wunsch(
                expected_flat, aug_seq
            )
            aligned_det = ["-" if s == "__PHYSICAL_GAP__" else s for s in aligned_aug]
            det_indices = [
                aug_to_det_idx[a_i] if a_i is not None else None for a_i in aug_indices
            ]

            anomalies: List[Anomaly] = []
            shelf_compliant = 0

            for pos_idx, (exp_sku, det_sku, d_idx, a_idx) in enumerate(
                zip(aligned_exp, aligned_det, det_indices, aug_indices)
            ):
                if exp_sku == "-" and det_sku == "-":
                    continue
                if exp_sku == det_sku and exp_sku != "-":
                    shelf_compliant += 1
                elif det_sku == "-":
                    total_missing += 1
                    ged_cost += 1.0
                    est_bbox = (
                        aug_gap_bbox[a_idx]
                        if (a_idx is not None and a_idx in aug_gap_bbox)
                        else self._estimate_missing_bbox(
                            pos_idx, det_indices, shelf_nodes, shelf_gaps
                        )
                    )
                    anomalies.append(
                        Anomaly(
                            type="MISSING_ITEM",
                            expected_sku=exp_sku,
                            detected_sku=None,
                            position_index=pos_idx,
                            estimated_gap_bbox=est_bbox,
                            details=f"Expected {exp_sku} at aligned slot {pos_idx} is missing (OOS).",
                        )
                    )
                elif exp_sku == "-":
                    total_extra += 1
                    ged_cost += 1.0
                    det_bbox = (
                        [int(round(v)) for v in shelf_nodes[d_idx].bbox]
                        if d_idx is not None
                        else None
                    )
                    anomalies.append(
                        Anomaly(
                            type="EXTRA_ITEM",
                            expected_sku=None,
                            detected_sku=det_sku,
                            position_index=pos_idx,
                            detected_bbox=det_bbox,
                            details=f"Unauthorized extra SKU {det_sku} inserted at slot {pos_idx}.",
                        )
                    )
                else:
                    total_misplaced += 1
                    ged_cost += 1.0
                    det_bbox = (
                        [int(round(v)) for v in shelf_nodes[d_idx].bbox]
                        if d_idx is not None
                        else None
                    )
                    anomalies.append(
                        Anomaly(
                            type="MISPLACED_ITEM",
                            expected_sku=exp_sku,
                            detected_sku=det_sku,
                            position_index=pos_idx,
                            detected_bbox=det_bbox,
                            details=f"Expected {exp_sku} but found {det_sku} at slot {pos_idx}.",
                        )
                    )

            denom = max(1, len(expected_flat) + sum(1 for e in aligned_exp if e == "-"))
            shelf_score = round((shelf_compliant / denom) * 100.0, 2)

            total_expected += len(expected_flat)
            total_detected += len(detected_seq)
            total_compliant += shelf_compliant

            shelf_reports.append(
                ShelfReport(
                    shelf_index=s_idx,
                    shelf_name=ref_shelf.shelf_name,
                    compliance_score=shelf_score,
                    needleman_wunsch_score=round(nw_score, 2),
                    expected_flattened=expected_flat,
                    detected_sequence=detected_seq,
                    aligned_expected=aligned_exp,
                    aligned_detected=aligned_det,
                    anomalies=anomalies,
                )
            )

        # Verify 2D cross-shelf vertical rules (ABOVE / BELOW)
        vertical_violations = 0
        if planogram.vertical_rules and spatial_graph.graph is not None:
            above_pairs = set()
            for u, v, data in spatial_graph.graph.edges(data=True):
                if data.get("relation") == "ABOVE":
                    u_sku = spatial_graph.graph.nodes[u].get("sku_id")
                    v_sku = spatial_graph.graph.nodes[v].get("sku_id")
                    above_pairs.add((u_sku, v_sku))

            for rule in planogram.vertical_rules:
                if (rule.upper_sku, rule.lower_sku) not in above_pairs:
                    vertical_violations += 1
                    ged_cost += 1.0
                    if shelf_reports:
                        shelf_reports[0].anomalies.append(
                            Anomaly(
                                type="VERTICAL_RULE_VIOLATION",
                                expected_sku=rule.upper_sku,
                                detected_sku=rule.lower_sku,
                                position_index=0,
                                details=f"Vertical topological rule violated: {rule.upper_sku} must be ABOVE {rule.lower_sku}.",
                            )
                        )

        v_compliance = (
            round(
                (1.0 - vertical_violations / len(planogram.vertical_rules)) * 100.0, 2
            )
            if planogram.vertical_rules
            else 100.0
        )

        overall_denom = max(1, total_expected + total_extra)
        overall_score = round((total_compliant / overall_denom) * 100.0, 2)

        return AuditReport(
            audit_id=audit_id,
            planogram_id=planogram.planogram_id,
            overall_compliance_score=overall_score,
            estimated_tilt_deg=round(float(spatial_graph.estimated_tilt_deg), 2),
            summary=AuditSummary(
                total_expected_facings=total_expected,
                total_detected_facings=total_detected,
                compliant_count=total_compliant,
                missing_count=total_missing,
                misplaced_count=total_misplaced,
                extra_count=total_extra,
                graph_edit_distance=round(ged_cost, 2),
                vertical_rule_compliance=v_compliance,
            ),
            shelf_reports=shelf_reports,
        )
