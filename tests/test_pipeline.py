"""Comprehensive Unit & Integration Tests for PlangoGraph (Milestone 1-4 Verification)."""

import json
import pathlib
import sys
import unittest
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.schemas import BoundingBoxDetection, PlanogramRef
from core.classifier import MetricSKUClassifier
from core.compliance_engine import PlanogramComplianceEngine
from core.detector import RetailObjectDetector
from core.graph_builder import SpatialGraphBuilder
from core.preprocessing import ShelfPreprocessor, rectify_quadratic_shelf_curve
from core.visualizer import PlanogramVisualizer


class TestPlangoGraphPipeline(unittest.TestCase):
    def setUp(self) -> None:
        plg_path = ROOT / "data" / "planograms" / "planogram_ref.json"
        self.planogram = PlanogramRef.model_validate_json(plg_path.read_text())
        self.graph_builder = SpatialGraphBuilder()
        self.compliance_engine = PlanogramComplianceEngine()
        self.preprocessor = ShelfPreprocessor()
        self.visualizer = PlanogramVisualizer()

    def test_100_percent_compliant_shelf(self) -> None:
        """Verifies that an exact physical layout yields 100.0% compliance and 0.0 GED."""
        dets = []
        y_shelves = [(80, 260), (310, 490), (540, 720)]
        for s_idx, shelf in enumerate(self.planogram.shelves):
            y1, y2 = y_shelves[s_idx]
            x_cursor = 60.0
            for slot in shelf.expected_sequence:
                w = slot.expected_width_px or 110.0
                for _ in range(slot.facings):
                    dets.append(
                        BoundingBoxDetection(
                            bbox=(x_cursor, float(y1), x_cursor + w, float(y2)),
                            confidence=0.96,
                            sku_id=slot.sku_id,
                        )
                    )
                    x_cursor += w + 12.0

        s_graph = self.graph_builder.build_graph(
            dets, expected_widths=self.planogram.expected_widths_map()
        )
        report = self.compliance_engine.evaluate(s_graph, self.planogram)
        self.assertEqual(report.overall_compliance_score, 100.0)
        self.assertEqual(report.summary.missing_count, 0)
        self.assertEqual(report.summary.misplaced_count, 0)
        self.assertEqual(report.summary.extra_count, 0)
        self.assertEqual(report.summary.graph_edit_distance, 0.0)
        self.assertEqual(report.summary.vertical_rule_compliance, 100.0)

    def test_missing_extra_and_misplaced_anomalies(self) -> None:
        """Verifies OOS deletion, extra insertion, and substitution detection via Needleman-Wunsch."""
        # Top shelf expects: [SKU_001, SKU_001, SKU_002, SKU_002, SKU_002]
        # Provide: [SKU_001, SKU_001, SKU_002, SKU_002] -> 1 missing SKU_002
        # Middle shelf expects: [SKU_003, SKU_003, SKU_005, SKU_005, SKU_005]
        # Provide: [SKU_003, SKU_999, SKU_005, SKU_005, SKU_005] -> 1 misplaced (SKU_999 instead of SKU_003)
        # Bottom shelf expects: [SKU_006, SKU_006, SKU_006, SKU_007, SKU_007]
        # Provide: [SKU_006, SKU_006, SKU_006, SKU_007, SKU_007, SKU_888] -> 1 extra item
        dets = [
            # Top shelf (y: 80..260) with a physical gap at the 5th slot
            BoundingBoxDetection(bbox=(60, 80, 170, 260), confidence=0.95, sku_id="SKU_001"),
            BoundingBoxDetection(bbox=(182, 80, 292, 260), confidence=0.94, sku_id="SKU_001"),
            BoundingBoxDetection(bbox=(304, 80, 414, 260), confidence=0.92, sku_id="SKU_002"),
            BoundingBoxDetection(bbox=(426, 80, 536, 260), confidence=0.91, sku_id="SKU_002"),
            # Middle shelf (y: 310..490)
            BoundingBoxDetection(bbox=(60, 310, 170, 490), confidence=0.95, sku_id="SKU_003"),
            BoundingBoxDetection(bbox=(182, 310, 292, 490), confidence=0.89, sku_id="SKU_999"),
            BoundingBoxDetection(bbox=(304, 310, 414, 490), confidence=0.93, sku_id="SKU_005"),
            BoundingBoxDetection(bbox=(426, 310, 536, 490), confidence=0.94, sku_id="SKU_005"),
            BoundingBoxDetection(bbox=(548, 310, 658, 490), confidence=0.92, sku_id="SKU_005"),
            # Bottom shelf (y: 540..720)
            BoundingBoxDetection(bbox=(60, 540, 205, 720), confidence=0.95, sku_id="SKU_006"),
            BoundingBoxDetection(bbox=(217, 540, 362, 720), confidence=0.95, sku_id="SKU_006"),
            BoundingBoxDetection(bbox=(374, 540, 519, 720), confidence=0.95, sku_id="SKU_006"),
            BoundingBoxDetection(bbox=(531, 540, 646, 720), confidence=0.93, sku_id="SKU_007"),
            BoundingBoxDetection(bbox=(658, 540, 773, 720), confidence=0.91, sku_id="SKU_007"),
            BoundingBoxDetection(bbox=(785, 540, 895, 720), confidence=0.88, sku_id="SKU_888"),
        ]
        s_graph = self.graph_builder.build_graph(
            dets, expected_widths=self.planogram.expected_widths_map()
        )
        report = self.compliance_engine.evaluate(s_graph, self.planogram)
        self.assertEqual(report.summary.missing_count, 1)
        self.assertEqual(report.summary.misplaced_count, 1)
        self.assertEqual(report.summary.extra_count, 1)
        self.assertEqual(report.summary.compliant_count, 13)
        self.assertAlmostEqual(report.overall_compliance_score, round(13 / 16 * 100.0, 2))

    def test_scale_invariance_1m_vs_3m(self) -> None:
        """Verifies 100% identical compliance metrics across 0.45x (3m away), 1.0x (1.5m), and 2.25x (0.8m close-up)."""
        base_dets = [
            BoundingBoxDetection(bbox=(60, 80, 170, 260), confidence=0.95, sku_id="SKU_001"),
            BoundingBoxDetection(bbox=(182, 80, 292, 260), confidence=0.94, sku_id="SKU_001"),
            BoundingBoxDetection(bbox=(304, 80, 414, 260), confidence=0.92, sku_id="SKU_002"),
            BoundingBoxDetection(bbox=(548, 80, 658, 260), confidence=0.91, sku_id="SKU_002"),
            BoundingBoxDetection(bbox=(60, 310, 170, 490), confidence=0.95, sku_id="SKU_003"),
            BoundingBoxDetection(bbox=(182, 310, 292, 490), confidence=0.89, sku_id="SKU_999"),
            BoundingBoxDetection(bbox=(304, 310, 414, 490), confidence=0.93, sku_id="SKU_005"),
            BoundingBoxDetection(bbox=(426, 310, 536, 490), confidence=0.94, sku_id="SKU_005"),
            BoundingBoxDetection(bbox=(548, 310, 658, 490), confidence=0.92, sku_id="SKU_005"),
            BoundingBoxDetection(bbox=(60, 540, 205, 720), confidence=0.95, sku_id="SKU_006"),
            BoundingBoxDetection(bbox=(217, 540, 362, 720), confidence=0.95, sku_id="SKU_006"),
            BoundingBoxDetection(bbox=(374, 540, 519, 720), confidence=0.95, sku_id="SKU_006"),
            BoundingBoxDetection(bbox=(531, 540, 646, 720), confidence=0.93, sku_id="SKU_007"),
            BoundingBoxDetection(bbox=(658, 540, 773, 720), confidence=0.91, sku_id="SKU_007"),
            BoundingBoxDetection(bbox=(785, 540, 895, 720), confidence=0.88, sku_id="SKU_888"),
        ]
        exp_widths = self.planogram.expected_widths_map()
        results = {}
        for scale in (0.45, 1.0, 2.25):
            scaled_dets = [
                BoundingBoxDetection(
                    bbox=tuple(c * scale for c in d.bbox),
                    confidence=d.confidence,
                    sku_id=d.sku_id,
                )
                for d in base_dets
            ]
            s_graph = self.graph_builder.build_graph(scaled_dets, expected_widths=exp_widths)
            rep = self.compliance_engine.evaluate(s_graph, self.planogram)
            results[scale] = (
                rep.overall_compliance_score,
                rep.summary.missing_count,
                rep.summary.misplaced_count,
                rep.summary.extra_count,
            )
            self.assertEqual(len(s_graph.gap_nodes[0]), 1)
            self.assertEqual(s_graph.gap_nodes[0][0].estimated_missing_facings, 1)

        self.assertEqual(results[0.45], results[1.0])
        self.assertEqual(results[2.25], results[1.0])

    def test_ransac_tilted_shelf_recovery(self) -> None:
        """Verifies that RANSAC tilt estimation + deskew recovers 100% shelf clustering under 9 deg camera tilt."""
        base_boxes = []
        skus = []
        for s_idx, shelf in enumerate(self.planogram.shelves):
            y1 = 90.0 + s_idx * 230.0
            y2 = y1 + 170.0
            x_cur = 70.0
            for slot in shelf.expected_sequence:
                w = slot.expected_width_px or 110.0
                for _ in range(slot.facings):
                    base_boxes.append((x_cur, y1, x_cur + w, y2))
                    skus.append(slot.sku_id)
                    x_cur += w + 15.0

        # Apply a +9.0 degree camera tilt (so deskew by +9.0 means rotating by +9.0)
        tilted_boxes = ShelfPreprocessor.deskew_bboxes(base_boxes, -9.0)
        dets = [
            BoundingBoxDetection(bbox=b, confidence=0.95, sku_id=s)
            for b, s in zip(tilted_boxes, skus)
        ]
        s_graph = self.graph_builder.build_graph(
            dets, expected_widths=self.planogram.expected_widths_map()
        )
        report = self.compliance_engine.evaluate(s_graph, self.planogram)
        self.assertAlmostEqual(s_graph.estimated_tilt_deg, 9.0, delta=1.2)
        self.assertEqual(report.overall_compliance_score, 100.0)

    def test_metric_classifier_and_nms(self) -> None:
        """Verifies pure-NumPy NMS and MetricSKUClassifier cosine similarity indexing."""
        clf = MetricSKUClassifier(embedding_dim=128, similarity_threshold=0.5)
        v1 = np.ones(128, dtype=np.float32)
        v2 = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
        clf.register_sku("SKU_001", "Organic Honey", reference_vector=v1)
        clf.register_sku("SKU_002", "Assam Tea", reference_vector=v2)

        matched_id, sim = clf.match_embedding(v1)
        self.assertEqual(matched_id, "SKU_001")
        self.assertGreater(sim, 0.99)

        keep = RetailObjectDetector.non_max_suppression(
            [(10, 10, 110, 210), (12, 12, 112, 212), (250, 10, 350, 210)],
            [0.95, 0.80, 0.92],
            iou_threshold=0.5,
        )
        self.assertEqual(keep, [0, 2])

    def test_quadratic_shelf_curve_rectification(self) -> None:
        """Verifies wide-angle parabolic shelf curvature rectification (a * xc_norm^2)."""
        xc_vals = [100.0, 300.0, 500.0, 700.0, 900.0]
        mean_xc = float(np.mean(xc_vals))
        a_true = 35.0
        curved_bboxes = []
        for x in xc_vals:
            xc_norm = (x - mean_xc) / 500.0
            sag = a_true * (xc_norm ** 2)
            curved_bboxes.append((x - 50.0, 80.0 + sag, x + 50.0, 260.0 + sag))
        rectified = rectify_quadratic_shelf_curve(curved_bboxes, [0] * len(xc_vals))
        for _, ymin, _, ymax in rectified:
            self.assertAlmostEqual(ymin, 80.0, places=5)
            self.assertAlmostEqual(ymax, 260.0, places=5)

    def test_auto_register_catalog_io_and_planogram_export(self) -> None:
        """Verifies auto_register_from_detections, save_catalog/load_catalog, and export_as_planogram_ref."""
        import tempfile

        img = np.full((600, 800, 3), 240, dtype=np.uint8)
        # Shelf 0: 2 red boxes (same SKU_AUTO_01), 1 blue box (SKU_AUTO_02)
        cv2.rectangle(img, (50, 80), (150, 240), (20, 30, 220), -1)
        cv2.rectangle(img, (170, 80), (270, 240), (20, 30, 220), -1)
        cv2.rectangle(img, (290, 80), (390, 240), (220, 120, 20), -1)
        # Shelf 1: 2 green boxes (SKU_AUTO_03) directly below
        cv2.rectangle(img, (50, 320), (150, 480), (30, 190, 40), -1)
        cv2.rectangle(img, (170, 320), (270, 480), (30, 190, 40), -1)

        raw_dets = [
            BoundingBoxDetection(bbox=(50.0, 80.0, 150.0, 240.0), confidence=0.95),
            BoundingBoxDetection(bbox=(170.0, 80.0, 270.0, 240.0), confidence=0.94),
            BoundingBoxDetection(bbox=(290.0, 80.0, 390.0, 240.0), confidence=0.93),
            BoundingBoxDetection(bbox=(50.0, 320.0, 150.0, 480.0), confidence=0.96),
            BoundingBoxDetection(bbox=(170.0, 320.0, 270.0, 480.0), confidence=0.95),
        ]

        clf = MetricSKUClassifier()
        cal_dets = clf.auto_register_from_detections(img, raw_dets, cluster_sim_thresh=0.82)
        self.assertEqual(len(cal_dets), 5)
        self.assertEqual(cal_dets[0].sku_id, "SKU_AUTO_01")
        self.assertEqual(cal_dets[1].sku_id, "SKU_AUTO_01")
        self.assertEqual(cal_dets[2].sku_id, "SKU_AUTO_02")
        self.assertEqual(cal_dets[3].sku_id, "SKU_AUTO_03")
        self.assertEqual(cal_dets[4].sku_id, "SKU_AUTO_03")
        self.assertEqual(len(clf._sku_catalog), 3)
        self.assertIs(clf._sku_Catalog, clf._sku_catalog)

        with tempfile.TemporaryDirectory() as tmpdir:
            cat_path = pathlib.Path(tmpdir) / "sku_catalog.npz"
            saved_count = clf.save_catalog(cat_path)
            self.assertEqual(saved_count, 3)

            clf2 = MetricSKUClassifier()
            loaded_count = clf2.load_catalog(cat_path)
            self.assertEqual(loaded_count, 3)
            reclassified = clf2.classify_detections(img, raw_dets)
            self.assertEqual([d.sku_id for d in reclassified], [d.sku_id for d in cal_dets])

        s_graph = self.graph_builder.build_graph(cal_dets)
        plg_ref = self.graph_builder.export_as_planogram_ref(s_graph)
        self.assertEqual(plg_ref.planogram_id, "PLG_CALIBRATED_BAY")
        self.assertEqual(len(plg_ref.shelves), 2)
        self.assertEqual(plg_ref.shelves[0].expected_sequence[0].sku_id, "SKU_AUTO_01")
        self.assertEqual(plg_ref.shelves[0].expected_sequence[0].facings, 2)
        self.assertEqual(plg_ref.shelves[0].expected_sequence[1].sku_id, "SKU_AUTO_02")
        self.assertEqual(plg_ref.shelves[0].expected_sequence[1].facings, 1)
        self.assertTrue(
            any(
                r.upper_sku == "SKU_AUTO_01" and r.lower_sku == "SKU_AUTO_03"
                for r in plg_ref.vertical_rules
            )
        )
        rep = self.compliance_engine.evaluate(s_graph, plg_ref)
        self.assertEqual(rep.overall_compliance_score, 100.0)

    def test_onnx_detector_and_classifier_bridge(self) -> None:
        """Verifies YOLOv8 1024x1024 letterbox + (1,5,21504)->(21504,5) transpose and DINOv2 224x224 384-D ONNX bridge."""
        class _DummyInput:
            name = "images"

        class _DummyYoloSession:
            def get_inputs(self):
                return [_DummyInput()]

            def run(self, _, feed):
                blob = feed["images"]
                assert blob.shape == (1, 3, 1024, 1024)
                out = np.zeros((1, 5, 21504), dtype=np.float32)
                # In a 512x1024 image (h=512, w=1024), r=1.0, dw=0, dh=256.
                # Box at original [100, 50, 200, 150] -> xc=150, yc=100+256=356, w=100, h=100
                out[0, :, 0] = [150.0, 356.0, 100.0, 100.0, 0.92]
                return [out]

        det = RetailObjectDetector(conf_threshold=0.35, iou_threshold=0.55)
        det._onnx_session = _DummyYoloSession()
        self.assertEqual(det.backend_name, "YOLOv8-ONNX")
        dummy_img = np.zeros((512, 1024, 3), dtype=np.uint8)
        boxes = det.detect(dummy_img)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].bbox[0], 100.0, places=3)
        self.assertAlmostEqual(boxes[0].bbox[1], 50.0, places=3)
        self.assertAlmostEqual(boxes[0].bbox[2], 200.0, places=3)
        self.assertAlmostEqual(boxes[0].bbox[3], 150.0, places=3)

        class _DummyDinoSession:
            def get_inputs(self):
                return [_DummyInput()]

            def run(self, _, feed):
                blob = feed["images"]
                assert blob.shape == (1, 3, 224, 224)
                vec = np.arange(1, 385, dtype=np.float32).reshape(1, 384)
                return [vec]

        clf = MetricSKUClassifier()
        clf._onnx_session = _DummyDinoSession()
        clf.embedding_dim = 384
        self.assertEqual(clf.backend_name, "DINOv2 ONNX/GPU")
        emb = clf.compute_embedding(dummy_img[50:150, 100:200])
        self.assertEqual(emb.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)

    def test_high_res_multipack_shelf_detection(self) -> None:
        """Verifies that a 2400x1800 high-resolution shelf photo with large multipack boxes (650x500 px)
        yields >= 4 detections and extracts >= 2 shelves and >= 2 SKUs via export_as_planogram_ref()."""
        high_res_img = np.full((1800, 2400, 3), 235, dtype=np.uint8)
        # Top shelf (y: 180..680): 3 large tissue multipacks (650x500 px each)
        cv2.rectangle(high_res_img, (80, 180), (730, 680), (30, 50, 210), -1)
        cv2.rectangle(high_res_img, (80, 180), (730, 680), (15, 20, 90), 8)
        cv2.rectangle(high_res_img, (820, 180), (1470, 680), (30, 50, 210), -1)
        cv2.rectangle(high_res_img, (820, 180), (1470, 680), (15, 20, 90), 8)
        cv2.rectangle(high_res_img, (1560, 180), (2210, 680), (210, 110, 30), -1)
        cv2.rectangle(high_res_img, (1560, 180), (2210, 680), (90, 40, 15), 8)

        # Bottom shelf (y: 1050..1550): 2 large paper towel multipacks (650x500 px each)
        cv2.rectangle(high_res_img, (80, 1050), (730, 1550), (40, 180, 50), -1)
        cv2.rectangle(high_res_img, (80, 1050), (730, 1550), (15, 80, 20), 8)
        cv2.rectangle(high_res_img, (820, 1050), (1470, 1550), (40, 180, 50), -1)
        cv2.rectangle(high_res_img, (820, 1050), (1470, 1550), (15, 80, 20), 8)

        detector = RetailObjectDetector()
        detector._onnx_session = None
        detector._model = None
        self.assertEqual(detector.backend_name, "MultiScale-Contour-Fallback")

        dets = detector.detect(high_res_img)
        self.assertGreaterEqual(len(dets), 4)

        clf = MetricSKUClassifier()
        cal_dets = clf.auto_register_from_detections(high_res_img, dets)
        s_graph = self.graph_builder.build_graph(cal_dets)
        plg_ref = self.graph_builder.export_as_planogram_ref(s_graph)
        self.assertGreaterEqual(len(plg_ref.shelves), 2)
        self.assertGreaterEqual(len(clf._sku_catalog), 2)

    def test_wild_shelf_misplaced_intruder_detection(self) -> None:
        """Verifies single-photo anomaly smoothing (sandwich intruder, edge singleton intruder,
        and physical OOS gap back-filling) plus double-stacked shelf cluster merging."""
        dets = [
            # Shelf 0: [SKU_A, SKU_A, SKU_INTRUDER, SKU_A, SKU_B, (GAP), SKU_B, SKU_INTRUDER_2]
            BoundingBoxDetection(bbox=(50.0, 100.0, 150.0, 260.0), confidence=0.95, sku_id="SKU_A"),
            # Slightly raised double-stacked box on the same physical shelf (ymax=210 overlaps [100, 260] by 110/160 = 68.75% >= 45%)
            BoundingBoxDetection(bbox=(162.0, 100.0, 262.0, 210.0), confidence=0.94, sku_id="SKU_A"),
            BoundingBoxDetection(bbox=(274.0, 100.0, 374.0, 260.0), confidence=0.92, sku_id="SKU_INTRUDER"),
            BoundingBoxDetection(bbox=(386.0, 100.0, 486.0, 260.0), confidence=0.95, sku_id="SKU_A"),
            BoundingBoxDetection(bbox=(498.0, 100.0, 598.0, 260.0), confidence=0.95, sku_id="SKU_B"),
            # Physical OOS gap of ~110px between x=598 and x=720
            BoundingBoxDetection(bbox=(720.0, 100.0, 820.0, 260.0), confidence=0.93, sku_id="SKU_B"),
            BoundingBoxDetection(bbox=(832.0, 100.0, 932.0, 260.0), confidence=0.90, sku_id="SKU_INTRUDER_2"),
        ]
        s_graph = self.graph_builder.build_graph(dets)
        # Double-stacked box at ymax=210 should merge into the single dominant shelf 0
        self.assertEqual(len(s_graph.shelf_clusters), 1)
        self.assertEqual(len(s_graph.gap_nodes[0]), 1)

        plg_ref = self.graph_builder.export_as_planogram_ref(s_graph, smooth_anomalies=True)
        cal_graph = self.graph_builder.build_graph(
            dets, expected_widths=plg_ref.expected_widths_map()
        )
        report = self.compliance_engine.evaluate(cal_graph, plg_ref)
        self.assertGreaterEqual(report.summary.misplaced_count, 2)
        self.assertGreaterEqual(report.summary.missing_count, 1)
        self.assertLess(report.overall_compliance_score, 100.0)


if __name__ == "__main__":
    unittest.main()
