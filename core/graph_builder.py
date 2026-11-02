"""Stage 3: Topological Spatial Graph Construction Engine."""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import networkx as nx
import numpy as np
from api.schemas import (
    BoundingBoxDetection,
    ExpectedSlot,
    PlanogramRef,
    ReferenceShelf,
    VerticalRule,
)
from core.preprocessing import ShelfPreprocessor, rectify_quadratic_shelf_curve


@dataclass
class GraphNode:
    node_id: str
    shelf_index: int
    slot_index: int
    sku_id: str
    bbox: Tuple[float, float, float, float]
    xc: float
    yc: float
    width: float
    height: float
    ymax: float
    confidence: float = 1.0
    is_gap: bool = False
    estimated_missing_facings: int = 0


@dataclass
class SpatialShelfGraph:
    shelf_clusters: Dict[int, List[GraphNode]] = field(default_factory=dict)
    gap_nodes: Dict[int, List[GraphNode]] = field(default_factory=dict)
    graph: Any = None
    estimated_tilt_deg: float = 0.0


class SpatialGraphBuilder:
    """Converts 2D bounding boxes into a structured topological scene graph G_extracted = (V, E)."""

    def __init__(
        self,
        tau_y_ratio: float = 0.26,
        gap_width_multiplier: float = 0.75,
        vertical_overlap_min_ratio: float = 0.20,
        enable_ransac_deskew: bool = True,
    ) -> None:
        self.tau_y_ratio = tau_y_ratio
        self.gap_width_multiplier = gap_width_multiplier
        self.vertical_overlap_min_ratio = vertical_overlap_min_ratio
        self.enable_ransac_deskew = enable_ransac_deskew

    @staticmethod
    def _cluster_1d_dbscan(values: np.ndarray, eps: float) -> np.ndarray:
        """Exact 1D DBSCAN / connected-components clustering with min_samples=1 on sorted coordinates.
        Guarantees identical output whether scikit-learn is installed or not."""
        if len(values) == 0:
            return np.array([], dtype=int)
        order = np.argsort(values)
        sorted_vals = values[order]
        labels_sorted = np.zeros(len(values), dtype=int)
        cluster_id = 0
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] - sorted_vals[i - 1] > eps:
                cluster_id += 1
            labels_sorted[i] = cluster_id

        # Sort cluster IDs top-to-bottom by mean y_max
        labels = np.zeros(len(values), dtype=int)
        labels[order] = labels_sorted
        unique_ids = np.unique(labels)
        means = [np.mean(values[labels == cid]) for cid in unique_ids]
        rank_map = {old_id: new_id for new_id, old_id in enumerate(np.argsort(means))}
        return np.array([rank_map[cid] for cid in labels], dtype=int)

    @staticmethod
    def _merge_overlapping_shelf_clusters(
        bboxes: List[Tuple[float, float, float, float]],
        shelf_labels: np.ndarray,
        overlap_ratio_thresh: float = 0.45,
    ) -> np.ndarray:
        """Merges smaller double-stacked or vertically overlapping sub-clusters into adjacent dominant
        shelf clusters when vertical interval overlap >= overlap_ratio_thresh * height_dominant."""
        if len(bboxes) == 0 or len(shelf_labels) == 0:
            return shelf_labels
        labels = shelf_labels.copy().astype(int)
        while True:
            unique_ids = sorted(int(c) for c in np.unique(labels))
            if len(unique_ids) <= 1:
                break
            stats = {
                cid: (
                    int(np.sum(labels == cid)),
                    float(np.median([bboxes[i][1] for i, lbl in enumerate(labels) if lbl == cid])),
                    float(np.median([bboxes[i][3] for i, lbl in enumerate(labels) if lbl == cid])),
                )
                for cid in unique_ids
            }
            merged = False
            for i_a, ca in enumerate(unique_ids):
                for cb in unique_ids[i_a + 1 :]:
                    (cnt_a, ymin_a, ymax_a), (cnt_b, ymin_b, ymax_b) = stats[ca], stats[cb]
                    dom_id, sub_id, dom_ymin, dom_ymax = (
                        (ca, cb, ymin_a, ymax_a) if cnt_a >= cnt_b else (cb, ca, ymin_b, ymax_b)
                    )
                    overlap_y = max(0.0, min(ymax_a, ymax_b) - max(ymin_a, ymin_b))
                    if overlap_y >= overlap_ratio_thresh * max(1.0, dom_ymax - dom_ymin):
                        labels[labels == sub_id] = dom_id
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break

        unique_final = np.unique(labels)
        y_maxs = np.array([b[3] for b in bboxes], dtype=np.float64)
        means = [float(np.mean(y_maxs[labels == cid])) for cid in unique_final]
        rank_map = {int(unique_final[idx]): new_id for new_id, idx in enumerate(np.argsort(means))}
        return np.array([rank_map[int(cid)] for cid in labels], dtype=int)

    def build_graph(
        self,
        detections: List[BoundingBoxDetection],
        expected_widths: Optional[Dict[str, float]] = None,
    ) -> SpatialShelfGraph:
        """Clusters detections into horizontal shelves, orders left-to-right, infers width-normalized
        gap nodes, and constructs directed topological edges."""
        product_dets = [d for d in detections if d.class_label == "product"]
        if not product_dets:
            return SpatialShelfGraph(graph=nx.DiGraph())

        raw_bboxes = [d.bbox for d in product_dets]
        tilt_deg = 0.0
        if self.enable_ransac_deskew and len(raw_bboxes) >= 3:
            tilt_deg = ShelfPreprocessor.estimate_ransac_bbox_tilt(raw_bboxes)
            bboxes = ShelfPreprocessor.deskew_bboxes(raw_bboxes, tilt_deg)
        else:
            bboxes = list(raw_bboxes)

        heights = np.array([b[3] - b[1] for b in bboxes], dtype=np.float64)
        widths = np.array([b[2] - b[0] for b in bboxes], dtype=np.float64)
        y_maxs = np.array([b[3] for b in bboxes], dtype=np.float64)

        median_h = float(np.median(heights)) if len(heights) > 0 else 100.0
        median_w = float(np.median(widths)) if len(widths) > 0 else 80.0
        tau_y = max(4.0, self.tau_y_ratio * median_h)

        shelf_labels = self._cluster_1d_dbscan(y_maxs, eps=tau_y)
        if self.enable_ransac_deskew and len(bboxes) >= 3:
            bboxes = rectify_quadratic_shelf_curve(bboxes, shelf_labels)
            y_maxs = np.array([b[3] for b in bboxes], dtype=np.float64)
            shelf_labels = self._cluster_1d_dbscan(y_maxs, eps=tau_y)
        shelf_labels = self._merge_overlapping_shelf_clusters(bboxes, shelf_labels)
        num_shelves = int(np.max(shelf_labels)) + 1 if len(shelf_labels) > 0 else 0

        g = nx.DiGraph()
        shelf_clusters: Dict[int, List[GraphNode]] = {}
        gap_nodes: Dict[int, List[GraphNode]] = {}

        for s_idx in range(num_shelves):
            member_indices = [i for i, lbl in enumerate(shelf_labels) if lbl == s_idx]
            # Sort left-to-right by horizontal centroid xc
            member_indices.sort(key=lambda idx: (bboxes[idx][0] + bboxes[idx][2]) * 0.5)

            nodes_on_shelf: List[GraphNode] = []
            gaps_on_shelf: List[GraphNode] = []

            for slot_pos, det_idx in enumerate(member_indices):
                det = product_dets[det_idx]
                xmin, ymin, xmax, ymax = bboxes[det_idx]
                w = xmax - xmin
                h = ymax - ymin
                xc = (xmin + xmax) * 0.5
                yc = (ymin + ymax) * 0.5
                node_id = f"S{s_idx}_N{slot_pos}"
                sku = det.sku_id or "UNKNOWN_SKU"
                node = GraphNode(
                    node_id=node_id,
                    shelf_index=s_idx,
                    slot_index=slot_pos,
                    sku_id=sku,
                    bbox=det.bbox,
                    xc=xc,
                    yc=yc,
                    width=w,
                    height=h,
                    ymax=ymax,
                    confidence=det.confidence,
                )
                nodes_on_shelf.append(node)
                g.add_node(
                    node_id,
                    shelf_index=s_idx,
                    slot_index=slot_pos,
                    sku_id=sku,
                    bbox=det.bbox,
                    xc=xc,
                    yc=yc,
                    width=w,
                    height=h,
                    is_gap=False,
                )

            # Intra-shelf edges & intrinsic scale-calibrated width-normalized gap inference
            med_det_w = (
                float(np.median([n.width for n in nodes_on_shelf]))
                if nodes_on_shelf
                else median_w
            )
            if expected_widths:
                med_ref_w = float(np.median(list(expected_widths.values())))
                scale_alpha = float(np.clip(med_det_w / max(1.0, med_ref_w), 0.15, 6.0))
            else:
                med_ref_w = med_det_w
                scale_alpha = 1.0

            for j in range(len(nodes_on_shelf) - 1):
                left_node = nodes_on_shelf[j]
                right_node = nodes_on_shelf[j + 1]
                g.add_edge(left_node.node_id, right_node.node_id, relation="IMMEDIATE_RIGHT_OF")
                g.add_edge(right_node.node_id, left_node.node_id, relation="IMMEDIATE_LEFT_OF")

                delta_x = right_node.bbox[0] - left_node.bbox[2]
                if expected_widths:
                    calibrated_ref_w = expected_widths.get(left_node.sku_id, med_ref_w) * scale_alpha
                else:
                    calibrated_ref_w = med_det_w
                tau_gap = self.gap_width_multiplier * calibrated_ref_w

                if delta_x > tau_gap:
                    missing_facings = max(1, int(round(delta_x / max(calibrated_ref_w, 1.0))))
                    gap_bbox = (
                        left_node.bbox[2],
                        min(left_node.bbox[1], right_node.bbox[1]),
                        right_node.bbox[0],
                        max(left_node.bbox[3], right_node.bbox[3]),
                    )
                    gap_id = f"S{s_idx}_GAP{j}"
                    gap_node = GraphNode(
                        node_id=gap_id,
                        shelf_index=s_idx,
                        slot_index=j,
                        sku_id="GAP",
                        bbox=gap_bbox,
                        xc=(gap_bbox[0] + gap_bbox[2]) * 0.5,
                        yc=(gap_bbox[1] + gap_bbox[3]) * 0.5,
                        width=delta_x,
                        height=gap_bbox[3] - gap_bbox[1],
                        ymax=gap_bbox[3],
                        confidence=1.0,
                        is_gap=True,
                        estimated_missing_facings=missing_facings,
                    )
                    gaps_on_shelf.append(gap_node)
                    g.add_node(
                        gap_id,
                        shelf_index=s_idx,
                        sku_id="GAP",
                        bbox=gap_bbox,
                        is_gap=True,
                        estimated_missing_facings=missing_facings,
                    )
                    g.add_edge(
                        left_node.node_id,
                        right_node.node_id,
                        relation="GAP_BETWEEN",
                        gap_px=float(delta_x),
                        estimated_missing_facings=missing_facings,
                    )

            shelf_clusters[s_idx] = nodes_on_shelf
            gap_nodes[s_idx] = gaps_on_shelf

        # Inter-shelf vertical edges (ABOVE / BELOW)
        for s_idx in range(num_shelves - 1):
            upper_nodes = shelf_clusters.get(s_idx, [])
            lower_nodes = shelf_clusters.get(s_idx + 1, [])
            for u_node in upper_nodes:
                for l_node in lower_nodes:
                    overlap_x = max(
                        0.0,
                        min(u_node.bbox[2], l_node.bbox[2])
                        - max(u_node.bbox[0], l_node.bbox[0]),
                    )
                    min_w = max(1.0, min(u_node.width, l_node.width))
                    if overlap_x / min_w >= self.vertical_overlap_min_ratio:
                        g.add_edge(
                            u_node.node_id,
                            l_node.node_id,
                            relation="ABOVE",
                            overlap_px=float(overlap_x),
                        )
                        g.add_edge(
                            l_node.node_id,
                            u_node.node_id,
                            relation="BELOW",
                            overlap_px=float(overlap_x),
                        )

        g.graph["shelf_clusters"] = shelf_clusters
        g.graph["gap_nodes"] = gap_nodes
        return SpatialShelfGraph(
            shelf_clusters=shelf_clusters,
            gap_nodes=gap_nodes,
            graph=g,
            estimated_tilt_deg=tilt_deg,
        )

    def export_as_planogram_ref(
        self,
        spatial_graph: SpatialShelfGraph,
        planogram_id: str = "PLG_CALIBRATED_BAY",
        smooth_anomalies: bool = True,
    ) -> PlanogramRef:
        """Converts a SpatialShelfGraph into a run-length encoded PlanogramRef blueprint
        with inferred expected widths, anomaly smoothing, and cross-shelf vertical rules."""
        clusters: Dict[int, List[GraphNode]] = getattr(spatial_graph, "shelf_clusters", None) or (
            spatial_graph.graph.graph.get("shelf_clusters", {})
            if getattr(spatial_graph, "graph", None) is not None
            else {}
        )
        all_gaps: Dict[int, List[GraphNode]] = getattr(spatial_graph, "gap_nodes", None) or (
            spatial_graph.graph.graph.get("gap_nodes", {})
            if getattr(spatial_graph, "graph", None) is not None
            else {}
        )
        ref_shelves: List[ReferenceShelf] = []
        for s_idx in sorted(clusters.keys()):
            nodes = [
                n for n in clusters[s_idx] if not n.is_gap and n.sku_id not in ("GAP", "")
            ]
            smoothed_skus = [n.sku_id for n in nodes]
            if smooth_anomalies and len(nodes) >= 3:
                # (a) Sandwich Intruder Replacement ([A, B, A] or [A, B, C, A])
                for idx in range(1, len(smoothed_skus) - 1):
                    if smoothed_skus[idx - 1] == smoothed_skus[idx + 1]:
                        smoothed_skus[idx] = smoothed_skus[idx - 1]
                for idx in range(1, len(smoothed_skus) - 2):
                    if smoothed_skus[idx - 1] == smoothed_skus[idx + 2]:
                        smoothed_skus[idx] = smoothed_skus[idx + 1] = smoothed_skus[idx - 1]

                # (b) Isolated Geometric/Visual Outlier Replacement
                sku_counts = Counter(smoothed_skus)
                for idx, node in enumerate(nodes):
                    if sku_counts[smoothed_skus[idx]] != 1:
                        continue
                    for adj_idx in (idx - 1, idx + 1):
                        if 0 <= adj_idx < len(nodes) and sku_counts[smoothed_skus[adj_idx]] >= 2:
                            adj_sku = smoothed_skus[adj_idx]
                            blk = [nodes[k] for k, s in enumerate(smoothed_skus) if s == adj_sku]
                            wb = float(np.median([max(1.0, b.width) for b in blk]))
                            hb = float(np.median([max(1.0, b.height) for b in blk]))
                            wi, hi = max(1.0, float(node.width)), max(1.0, float(node.height))
                            arb, ari = wb / hb, wi / hi
                            if (
                                abs(wi - wb) / max(wi, wb) > 0.30
                                or abs(hi - hb) / max(hi, hb) > 0.30
                                or abs(ari - arb) / max(ari, arb, 1e-4) > 0.30
                            ):
                                smoothed_skus[idx] = adj_sku
                                break

                # (c) Singleton Edge Intruder on Homogeneous Shelves (len >= 4)
                if len(nodes) >= 4:
                    sku_counts = Counter(smoothed_skus)
                    for idx in range(len(nodes)):
                        if sku_counts[smoothed_skus[idx]] == 1:
                            for adj_idx in (idx - 1, idx + 1):
                                if 0 <= adj_idx < len(nodes) and sku_counts[smoothed_skus[adj_idx]] >= 2:
                                    smoothed_skus[idx] = smoothed_skus[adj_idx]
                                    break

            expected_seq: List[ExpectedSlot] = []
            run_ranges: List[Tuple[int, int]] = []
            slot_num = 1
            i = 0
            while i < len(nodes):
                curr_sku = smoothed_skus[i]
                run_widths = [nodes[i].bbox[2] - nodes[i].bbox[0]]
                genuine_widths = (
                    [nodes[i].bbox[2] - nodes[i].bbox[0]]
                    if nodes[i].sku_id == curr_sku
                    else []
                )
                j = i + 1
                while j < len(nodes) and smoothed_skus[j] == curr_sku:
                    w_j = nodes[j].bbox[2] - nodes[j].bbox[0]
                    run_widths.append(w_j)
                    if nodes[j].sku_id == curr_sku:
                        genuine_widths.append(w_j)
                    j += 1
                use_widths = genuine_widths if genuine_widths else run_widths
                expected_seq.append(
                    ExpectedSlot(
                        slot=slot_num,
                        sku_id=curr_sku,
                        name=curr_sku,
                        facings=len(run_widths),
                        expected_width_px=round(float(np.mean(use_widths)), 1),
                    )
                )
                run_ranges.append((i, j - 1))
                slot_num += 1
                i = j

            # (d) Physical Gap Back-Filling
            if smooth_anomalies and expected_seq:
                for g in all_gaps.get(s_idx, []):
                    add_facings = max(1, int(g.estimated_missing_facings))
                    matched_run = False
                    for r_idx, (start_i, end_i) in enumerate(run_ranges):
                        if start_i <= g.slot_index <= end_i:
                            expected_seq[r_idx].facings += add_facings
                            matched_run = True
                            break
                    if not matched_run:
                        expected_seq[-1].facings += add_facings

            ref_shelves.append(
                ReferenceShelf(
                    shelf_index=s_idx,
                    shelf_name=f"Shelf {s_idx + 1}",
                    expected_sequence=expected_seq,
                )
            )

        vertical_rules: List[VerticalRule] = []
        nx_graph = getattr(spatial_graph, "graph", None)
        if nx_graph is not None:
            seen_pairs = set()
            for u, v, data in nx_graph.edges(data=True):
                if data.get("relation") == "ABOVE":
                    u_sku = nx_graph.nodes[u].get("sku_id")
                    v_sku = nx_graph.nodes[v].get("sku_id")
                    if (
                        u_sku
                        and v_sku
                        and "GAP" not in (u_sku, v_sku)
                        and (u_sku, v_sku) not in seen_pairs
                    ):
                        seen_pairs.add((u_sku, v_sku))
                        vertical_rules.append(
                            VerticalRule(upper_sku=u_sku, lower_sku=v_sku, rule_type="ABOVE")
                        )

        return PlanogramRef(
            planogram_id=planogram_id,
            bay_number=1,
            total_shelves=max(1, len(ref_shelves)),
            shelves=ref_shelves,
            vertical_rules=vertical_rules,
        )
