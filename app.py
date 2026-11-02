"""Streamlit Interactive Visual Audit Dashboard for PlangoGraph."""

from __future__ import annotations
import json
import pathlib
from typing import List, Optional
import cv2
import numpy as np
import streamlit as st
from api.schemas import AuditReport, BoundingBoxDetection, PlanogramRef
from core.classifier import MetricSKUClassifier
from core.compliance_engine import PlanogramComplianceEngine
from core.detector import RetailObjectDetector
from core.graph_builder import SpatialGraphBuilder, SpatialShelfGraph
from core.preprocessing import ShelfPreprocessor
from core.visualizer import PlanogramVisualizer


class PlangoGraphPipeline:
    """Stateful CV + Spatial Graph Compliance Pipeline container."""

    def __init__(self) -> None:
        self.preprocessor = ShelfPreprocessor()
        self.detector = RetailObjectDetector()
        self.classifier = MetricSKUClassifier()
        self.classifier.load_catalog()
        self.graph_builder = SpatialGraphBuilder()
        self.compliance_engine = PlanogramComplianceEngine()
        self.visualizer = PlanogramVisualizer()


st.set_page_config(
    page_title="PlangoGraph Retail Auditor",
    page_icon="🛒",
    layout="wide",
)

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = PlangoGraphPipeline()
pipeline: PlangoGraphPipeline = st.session_state["pipeline"]

default_plg_path = pathlib.Path("data/planograms/planogram_ref.json")
if "planogram_json" not in st.session_state:
    st.session_state["planogram_json"] = (
        default_plg_path.read_text() if default_plg_path.exists() else "{}"
    )

st.title("🛒 PlangoGraph: Retail Shelf Planogram Compliance")
st.markdown(
    "Automated retail shelf compliance powered by **2D Spatial Layout Graphs**, "
    "**Parabolic & RANSAC Perspective Rectification**, **1D DBSCAN Shelf Clustering**, "
    "and **Needleman-Wunsch Alignment**."
)

# Sidebar: Engine & Threshold Tuning + Runtime Telemetry + Planogram Blueprint
st.sidebar.header("🎛️ Control Center")
with st.sidebar.expander("⚙️ Engine & Threshold Tuning", expanded=True):
    conf_threshold = st.slider(
        "Detector Confidence (conf_threshold)",
        min_value=0.15,
        max_value=0.85,
        value=float(pipeline.detector.conf_threshold),
        step=0.05,
        key="slider_conf_thresh",
    )
    similarity_threshold = st.slider(
        "SKU Similarity Threshold (similarity_threshold)",
        min_value=0.30,
        max_value=0.90,
        value=float(pipeline.classifier.similarity_threshold),
        step=0.05,
        key="slider_sim_thresh",
    )
    gap_width_multiplier = st.slider(
        "Gap Width Multiplier (gap_width_multiplier)",
        min_value=0.50,
        max_value=1.20,
        value=float(pipeline.graph_builder.gap_width_multiplier),
        step=0.05,
        key="slider_gap_mult",
    )
    tau_y_ratio = st.slider(
        "DBSCAN Shelf τ_y Ratio (tau_y_ratio)",
        min_value=0.15,
        max_value=0.50,
        value=float(pipeline.graph_builder.tau_y_ratio),
        step=0.02,
        key="slider_tau_y",
    )

pipeline.detector.conf_threshold = conf_threshold
pipeline.classifier.similarity_threshold = similarity_threshold
pipeline.graph_builder.gap_width_multiplier = gap_width_multiplier
pipeline.graph_builder.tau_y_ratio = tau_y_ratio


def _render_audit_results(
    annotated_bgr: np.ndarray,
    report: AuditReport,
    key_prefix: str,
) -> None:
    """Renders 2-column Annotated Image + KPI Metrics, Shelf-by-Shelf Sequence Ribbon, and Downloads."""
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Annotated Spatial Scene Graph & Compliance Overlay")
        st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(
            "🟢 Green = Compliant | 🟡 Yellow = Misplaced/Extra | 🔴 Red Dashed = Out-of-Stock (OOS) Gap"
        )
    with col2:
        st.subheader("Compliance Audit Scorecard")
        m1, m2 = st.columns(2)
        m1.metric("Overall Compliance %", f"{report.overall_compliance_score:.1f}%")
        m2.metric("2D GED", f"{report.summary.graph_edit_distance:.1f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Compliant", report.summary.compliant_count)
        c2.metric("Missing (OOS)", report.summary.missing_count)
        c3.metric("Misplaced", report.summary.misplaced_count)
        c4.metric("Extra", report.summary.extra_count)

        t1, t2 = st.columns(2)
        t1.metric("Estimated Camera Tilt °", f"{report.estimated_tilt_deg:.2f}°")
        t2.metric("Vertical Rule Compliance", f"{report.summary.vertical_rule_compliance:.1f}%")

        dl_col1, dl_col2 = st.columns(2)
        report_json_bytes = json.dumps(
            report.model_dump(exclude={"annotated_image_base64"}), indent=2
        ).encode("utf-8")
        dl_col1.download_button(
            label="📥 Download audit_report.json",
            data=report_json_bytes,
            file_name=f"{report.audit_id.lower()}_report.json",
            mime="application/json",
            key=f"dl_json_{key_prefix}",
            use_container_width=True,
        )
        ok_png, png_buf = cv2.imencode(".png", annotated_bgr)
        if ok_png:
            dl_col2.download_button(
                label="🖼️ Download Annotated PNG",
                data=png_buf.tobytes(),
                file_name=f"{report.audit_id.lower()}_annotated.png",
                mime="image/png",
                key=f"dl_png_{key_prefix}",
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("📊 Shelf-by-Shelf Alignment Breakdown")
    for shelf in report.shelf_reports:
        shelf_title = shelf.shelf_name or f"Shelf {shelf.shelf_index + 1}"
        st.markdown(
            f"**{shelf_title}** — Compliance: **{shelf.compliance_score:.1f}%** "
            f"(NW Score: `{shelf.needleman_wunsch_score:.1f}`)"
        )
        st.progress(min(max(shelf.compliance_score / 100.0, 0.0), 1.0))

        rows = []
        for pos_idx, (exp_sku, det_sku) in enumerate(
            zip(shelf.aligned_expected, shelf.aligned_detected)
        ):
            if exp_sku == "-" and det_sku == "-":
                continue
            if exp_sku == det_sku and exp_sku != "-":
                badge = "✅ OK"
            elif det_sku == "-":
                badge = "❌ OOS GAP"
            elif exp_sku == "-":
                badge = "➕ EXTRA"
            else:
                badge = "⚠️ MISPLACED"
            rows.append(
                {
                    "Aligned Pos": pos_idx,
                    "Expected SKU": exp_sku,
                    "Detected SKU": det_sku,
                    "Status Badge": badge,
                }
            )
        if rows:
            st.table(rows)
        else:
            st.caption("No slots aligned on this shelf.")

    with st.expander("🔎 Raw Audit Report JSON"):
        st.json(report.model_dump(exclude={"annotated_image_base64"}))


def _calibrate_from_image(
    img_bgr: np.ndarray,
    key_prefix: str,
    precomputed_dets: Optional[List[BoundingBoxDetection]] = None,
    smooth_anomalies: bool = True,
) -> None:
    """Auto-registers SKUs from a wild shelf photo and extracts the active PlanogramRef blueprint."""
    rectified_bgr, _ = pipeline.preprocessor.preprocess(img_bgr)
    raw_dets = (
        pipeline.detector.detect(rectified_bgr, precomputed_boxes=precomputed_dets)
        if precomputed_dets is not None
        else pipeline.detector.detect(rectified_bgr)
    )
    if len(raw_dets) == 0:
        st.warning(
            f"0 detections at confidence threshold {pipeline.detector.conf_threshold:.2f} "
            f"(active backend: `{pipeline.detector.backend_name}`). Retrying with conf_threshold = 0.12..."
        )
        prev_conf = pipeline.detector.conf_threshold
        pipeline.detector.conf_threshold = 0.12
        try:
            raw_dets = (
                pipeline.detector.detect(rectified_bgr, precomputed_boxes=precomputed_dets)
                if precomputed_dets is not None
                else pipeline.detector.detect(rectified_bgr)
            )
        finally:
            pipeline.detector.conf_threshold = prev_conf

    cal_dets = pipeline.classifier.auto_register_from_detections(rectified_bgr, raw_dets)
    spatial_graph: SpatialShelfGraph = pipeline.graph_builder.build_graph(cal_dets)
    planogram_ref: PlanogramRef = pipeline.graph_builder.export_as_planogram_ref(
        spatial_graph, smooth_anomalies=smooth_anomalies
    )

    st.session_state["planogram_json"] = json.dumps(planogram_ref.model_dump(), indent=2)
    pipeline.classifier.save_catalog()

    n_shelves = len(planogram_ref.shelves)
    n_skus = len(pipeline.classifier._sku_catalog)
    st.toast(f"Calibrated {n_shelves} shelves & {n_skus} SKUs!", icon="🎯")
    st.success(
        f"Calibrated! Extracted {n_shelves} shelves and {n_skus} SKUs as the active blueprint. "
        "Now upload or capture a modified shelf image to test compliance."
    )

    cal_graph = pipeline.graph_builder.build_graph(
        cal_dets, expected_widths=planogram_ref.expected_widths_map()
    )
    report = pipeline.compliance_engine.evaluate(cal_graph, planogram_ref)
    annotated = pipeline.visualizer.render_audit(rectified_bgr, cal_graph, report)
    _render_audit_results(annotated, report, key_prefix=f"{key_prefix}_cal")


def run_audit(
    image_bgr: np.ndarray,
    planogram: PlanogramRef,
    key_prefix: str = "audit",
    precomputed_dets: Optional[List[BoundingBoxDetection]] = None,
) -> None:
    """Runs the 4-stage compliance pipeline against the active PlanogramRef blueprint."""
    rectified_bgr, _ = pipeline.preprocessor.preprocess(image_bgr)
    dets = (
        pipeline.detector.detect(rectified_bgr, precomputed_boxes=precomputed_dets)
        if precomputed_dets is not None
        else pipeline.detector.detect(rectified_bgr)
    )
    if len(dets) == 0:
        st.warning(
            f"0 detections at confidence threshold {pipeline.detector.conf_threshold:.2f} "
            f"(active backend: `{pipeline.detector.backend_name}`). Retrying with conf_threshold = 0.12..."
        )
        prev_conf = pipeline.detector.conf_threshold
        pipeline.detector.conf_threshold = 0.12
        try:
            dets = (
                pipeline.detector.detect(rectified_bgr, precomputed_boxes=precomputed_dets)
                if precomputed_dets is not None
                else pipeline.detector.detect(rectified_bgr)
            )
        finally:
            pipeline.detector.conf_threshold = prev_conf
    classified_dets = pipeline.classifier.classify_detections(rectified_bgr, dets)
    s_graph = pipeline.graph_builder.build_graph(
        classified_dets, expected_widths=planogram.expected_widths_map()
    )
    report = pipeline.compliance_engine.evaluate(s_graph, planogram)
    annotated = pipeline.visualizer.render_audit(rectified_bgr, s_graph, report)
    _render_audit_results(annotated, report, key_prefix=key_prefix)


# Main tabs: Demo, Live Camera, Upload
tab_demo, tab_camera, tab_upload = st.tabs(
    ["🧪 Sample Shelf Demo", "📷 Live Camera Audit", "📁 Upload Image / Detections"]
)

with tab_demo:
    st.subheader("Synthetic & Benchmark Retail Shelf")
    st.markdown("Run the compliance engine on the pre-configured sample retail shelf.")
    if st.button("🔍 Run Demo Audit", key="btn_demo"):
        sample_img_path = pathlib.Path("data/sample_images/shelf_audit_sample.png")
        sample_dets_path = pathlib.Path("data/sample_images/shelf_detections_sample.json")
        if sample_img_path.exists() and sample_dets_path.exists():
            planogram = PlanogramRef.model_validate_json(st.session_state["planogram_json"])
            img_bgr = cv2.imread(str(sample_img_path))
            raw_dets = [
                BoundingBoxDetection.model_validate(d)
                for d in json.loads(sample_dets_path.read_text())
            ]
            run_audit(img_bgr, planogram, key_prefix="demo", precomputed_dets=raw_dets)
        else:
            st.error("Sample shelf data files not found in data/sample_images/")

with tab_camera:
    st.subheader("Live Camera Capture & One-Click Calibration")
    st.markdown(
        "Use your device's webcam or phone camera to snap a retail shelf photo. "
        "**Calibrate** on a clean shelf first, or **Audit** against the active blueprint."
    )
    cam_file = st.camera_input("Take a photo of the retail shelf", key="cam_input")
    if cam_file is not None:
        file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            cam_smooth = st.checkbox(
                "🧠 Smart Anomaly Smoothing (Detect Misplaced Intruders & Gaps in Single Wild Photo)",
                value=True,
                key="chk_cam_smooth",
            )
            b_col1, b_col2 = st.columns(2)
            cal_clicked = b_col1.button(
                "🎯 Calibrate from This Image (Auto-Register SKUs & Extract Reference Blueprint)",
                key="btn_cam_calibrate",
                use_container_width=True,
            )
            aud_clicked = b_col2.button(
                "🔍 Run Compliance Audit Against Active Blueprint",
                key="btn_cam_audit",
                use_container_width=True,
            )
            if cal_clicked:
                with st.spinner("Calibrating SKUs and extracting reference planogram..."):
                    _calibrate_from_image(img_bgr, key_prefix="cam", smooth_anomalies=cam_smooth)
            elif aud_clicked:
                planogram = PlanogramRef.model_validate_json(st.session_state["planogram_json"])
                with st.spinner("Analyzing live shelf image..."):
                    run_audit(img_bgr, planogram, key_prefix="cam_aud")
        else:
            st.error("Could not decode captured camera frame.")

with tab_upload:
    st.subheader("Upload Custom Shelf Image")
    uploaded_img = st.file_uploader(
        "Upload shelf image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="up_img"
    )
    uploaded_json = st.file_uploader(
        "Optional: Upload precomputed detections JSON", type=["json"], key="up_json"
    )
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        custom_dets = None
        if uploaded_json is not None:
            try:
                custom_dets = [
                    BoundingBoxDetection.model_validate(d)
                    for d in json.loads(uploaded_json.read().decode("utf-8"))
                ]
            except Exception as e:
                st.warning(f"Could not parse detections JSON: {e}")

        up_smooth = st.checkbox(
            "🧠 Smart Anomaly Smoothing (Detect Misplaced Intruders & Gaps in Single Wild Photo)",
            value=True,
            key="chk_upload_smooth",
        )
        u_col1, u_col2 = st.columns(2)
        up_cal_clicked = u_col1.button(
            "🎯 Calibrate from This Image (Auto-Register SKUs & Extract Reference Blueprint)",
            key="btn_upload_calibrate",
            use_container_width=True,
        )
        up_aud_clicked = u_col2.button(
            "🔍 Run Compliance Audit Against Active Blueprint",
            key="btn_upload_audit",
            use_container_width=True,
        )
        if up_cal_clicked:
            with st.spinner("Calibrating SKUs and extracting reference planogram..."):
                _calibrate_from_image(
                    img_bgr,
                    key_prefix="upload",
                    precomputed_dets=custom_dets,
                    smooth_anomalies=up_smooth,
                )
        elif up_aud_clicked:
            planogram = PlanogramRef.model_validate_json(st.session_state["planogram_json"])
            with st.spinner("Running compliance audit on uploaded image..."):
                run_audit(img_bgr, planogram, key_prefix="upload_aud", precomputed_dets=custom_dets)

# Render Runtime Telemetry & Editable Blueprint in Sidebar AFTER tab callbacks so calibration updates reflect immediately
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Runtime Telemetry")
st.sidebar.markdown(
    f"- **Detector Backend**: `{pipeline.detector.backend_name}`\n"
    f"- **Classifier Backend**: `{pipeline.classifier.backend_name}`\n"
    f"- **Registered SKUs**: `{len(pipeline.classifier._sku_catalog)}`"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Planogram Configuration")
edited_planogram_json = st.sidebar.text_area(
    "Reference Planogram Blueprint (JSON)",
    value=st.session_state["planogram_json"],
    height=280,
)
if edited_planogram_json != st.session_state["planogram_json"]:
    st.session_state["planogram_json"] = edited_planogram_json
