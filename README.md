# PlangoGraph (v2.2): Retail Planogram Compliance Auditing via Spatial Layout Graphs

**Runtime Stack:** Python 3.10 – 3.13 • ONNX Runtime (CPU Edge) • Ultralytics YOLOv8m & DINOv2 (Colab L4 GPU) • FastAPI • Streamlit  
**License:** [Apache License 2.0](LICENSE)

---

## 1. Project Overview

**PlangoGraph** is a four-stage computer vision and mathematical graph alignment system that audits physical retail shelves against digital planogram blueprints. The system was deliberately built around a class-agnostic product localiser (`yolov8m_sku110k.onnx`) paired with an external metric feature index (`dinov2_vits14.onnx`), rather than a monolithic end-to-end multi-class detector, so that new retail Stock Keeping Units (SKUs) can be registered from a single reference crop without retraining neural weights.

On the local edge workstation, the entire pipeline executes through `onnxruntime`, OpenCV, and NetworkX in **39.5 milliseconds per image (25.3 frames per second)** inside a 180 megabyte virtual environment with zero CUDA dependencies. Heavy neural training on the 1.73 million dense bounding boxes of the SKU-110K dataset is isolated to a Google Colab NVIDIA L4 GPU workflow.

---

## 2. Four-Stage System Architecture

This section outlines the execution flow of a shelf photograph from raw image upload to the final compliance scorecard. **Table 1** maps each stage of the pipeline to its core module, mathematical mechanism, and empirical output.

| Stage | Module Path | Mathematical Mechanism | Operational Output |
| :--- | :--- | :--- | :--- |
| **Stage 1: Preprocessing** | `core/preprocessing.py` | CIE-LAB $L$-channel CLAHE ($\beta=2.0$, $8\times 8$ grid) and second-order polynomial residual curve fit | Glare-normalised BGR image with wide-angle smartphone barrel curvature removed |
| **Stage 2A: Localisation** | `core/detector.py` | $1024\times 1024$ YOLOv8m ONNX detector (`nc=1`) with letterbox unscaling, NMS ($\tau_{\text{iou}}=0.55$), and Sobel/Canny multi-scale fallback | Class-agnostic product bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ ($93.7\%$ $\text{mAP}@50$) |
| **Stage 2B: SKU Indexing** | `core/classifier.py` | L2-normalised 384-D DINOv2 ViT-S/14 ONNX descriptor fused with 24-bin HSV histogram and spatial adjacency bonus ($+0.08$) | SKU-labelled detections matched via inner-product cosine similarity ($\tau_{\text{sim}}=0.66$) |
| **Stage 3: Scene Graph** | `core/graph_builder.py` | Multi-shelf pairwise bottom-edge slope RANSAC, 1D DBSCAN ($\tau_y=0.26\tilde{h}$), $45\%$ vertical-span merge, intrinsic scale $\alpha_k$, and Markov smoothing | Directed `networkx.DiGraph` with `IMMEDIATE_RIGHT_OF`, `ABOVE`/`BELOW`, and `__PHYSICAL_GAP__` nodes |
| **Stage 4: Alignment** | `core/compliance_engine.py` | Spatially-augmented Needleman-Wunsch dynamic programming ($\text{match}=+2.0$, $\text{gap}=-0.65$, $\text{subst}=-1.60$) and 2D Graph Edit Distance | JSON `AuditReport` and colour-coded OpenCV visual overlay (`core/visualizer.py`) |

### Key Algorithmic Design Decisions

- **CIE-LAB Luminance Equalisation:** Equalising luminance solely on the $L$ channel in CIE-LAB space, rather than equalising RGB channels separately, removes overhead store glare while preserving the label hue and saturation required by Stage 2B.
- **Pairwise Bottom-Edge Slope RANSAC:** Estimating camera roll from pairwise slopes across bounding-box bases $(x_c, y_{\max})$, rather than from global Canny-Hough lines, keeps shelf row recovery at **100.0%** up to $15^\circ$ camera tilt even when hanging price tags occlude physical shelf rails.
- **Intrinsic Width Scale Normalisation ($\alpha_k$):** Scaling expected product widths dynamically by the per-shelf median box-width ratio $\alpha_k = \text{median}(w_{\text{det},k})/\text{median}(\hat{w}_{\text{ref},k})$, rather than relying on fixed pixel thresholds or physical ArUco stickers, makes out-of-stock gap counting invariant across camera standoff distances from $0.8\text{ m}$ to $3.0\text{ m}$.
- **Markov Singleton Smoothing (`[A, B, A] → [A, A, A]`):** Smoothing isolated singleton intruders and back-filling physical gaps when calibrating from a wild photograph, rather than copying raw detections verbatim into the blueprint, allows the engine to flag misplaced items in amber and empty gaps in red from a single uncalibrated shelf image.

---

## 3. Repository Structure

The codebase contains 17 core tracked modules indexed into 110 Abstract Syntax Tree (AST) nodes and 1,772 call-graph edges:

```text
plangograph/
├── api/
│   ├── main.py                      # FastAPI service entrypoint (port 8000)
│   ├── routes.py                    # REST endpoints (/api/v1/planograms, /audit/analyze, /skus)
│   └── schemas.py                   # Pydantic v2 input and output data contracts
├── core/
│   ├── preprocessing.py             # Stage 1: CIE-LAB CLAHE, slope RANSAC & quadratic curve fit
│   ├── detector.py                  # Stage 2A: YOLOv8m ONNX Runtime bridge & multi-scale contour fallback
│   ├── classifier.py                # Stage 2B: DINOv2 384-D ONNX bridge, HSV fusion & auto-cataloging
│   ├── graph_builder.py             # Stage 3: y_max DBSCAN, vertical-span merge, scale alpha & Markov smoothing
│   ├── compliance_engine.py         # Stage 4: Spatially-augmented Needleman-Wunsch & 2D GED engine
│   └── visualizer.py                # OpenCV bounding-box, dashed OOS gap & scene graph overlay renderer
├── data/
│   ├── planograms/                  # Reference digital blueprints (planogram_ref.json)
│   ├── sample_images/               # Test shelf captures & precomputed detection fixtures
│   ├── weights/                     # Local ONNX models (yolov8m_sku110k.onnx, dinov2_vits14.onnx) & sku_catalog.npz
│   └── outputs/                     # Annotated audit PNGs, JSON scorecards & benchmark logs
├── docs/
│   ├── plangograph_evaluation_12slides.pptx # 12-slide evaluation presentation deck
│   ├── plangograph_ieee_paper.tex           # IEEE conference paper manuscript (LaTeX)
│   └── plangograph_springer_paper.tex       # Springer LNCS paper manuscript (LaTeX)
├── notebooks/
│   ├── 01_yolo_training.ipynb       # Google Colab L4 GPU training on SKU-110K & ONNX export
│   └── 02_graph_experiments.ipynb   # DINOv2 + FAISS retrieval & Monte Carlo graph ablations
├── scripts/
│   ├── build_12slide_deck.py        # Automated generator for the 12-slide evaluation deck
│   └── run_benchmarks.py            # Reproducible 500-scene tilt & 200-bay anomaly benchmark suite
├── tests/
│   └── test_pipeline.py             # 10 unit and integration tests (0.22s execution time)
├── LICENSE                          # Apache License, Version 2.0
├── README.md                        # Project architecture, benchmarks & operational guide
├── app.py                           # Three-tab Streamlit auditing workstation
├── start.sh                         # Automated virtual environment & multi-modal service launcher
├── requirements.txt                 # Lightweight local CPU manifest (<200 MB, zero CUDA bloat)
└── requirements-gpu.txt             # Cloud GPU training manifest for Google Colab
```

---

## 4. Quickstart and Operational Commands

The `start.sh` orchestrator automatically creates `.venv`, synchronises `requirements.txt`, and manages background processes. **Table 2** lists the supported command-line modes.

| Command | Execution Mode | Description |
| :--- | :--- | :--- |
| `./start.sh --both` | **Full Local Stack** | Launches both the FastAPI REST server (`http://localhost:8000`) and the Streamlit UI (`http://localhost:8501`) |
| `./start.sh --ui` | **Streamlit Workstation** | Starts the interactive three-tab auditing dashboard (Sample Shelf Demo, Live Camera Audit, and Upload Image) |
| `./start.sh --api` | **FastAPI Service Only** | Starts the Uvicorn REST server with interactive OpenAPI docs at `http://localhost:8000/docs` |
| `./start.sh --test` | **Automated Test Suite** | Runs all 10 unit and integration tests in `tests/test_pipeline.py` (tilt recovery, scale invariance, ONNX bridge, and intruder detection) |
| `./start.sh --benchmark` | **Empirical Ablations** | Executes `scripts/run_benchmarks.py` across 500 tilt scenes and 200 multi-anomaly bays, updating `data/outputs/` |

### Using the Streamlit Auditing Workstation (`app.py`)

1. **Tab 1 — Sample Shelf Demo:** Click **Run Demo Audit** to evaluate the reference bay (`PLG_SAKHI_AISLE_04`), which achieves an **86.7% compliance score** with 13 compliant facings (green), 1 missing `SKU_002` facing in slot 3 of the top shelf (red dashed box), and 1 misplaced `SKU_999` item on the middle shelf (amber box).
2. **Tab 2 — Live Camera Audit / Tab 3 — Upload Image & Detections:** Upload any real-world shelf photograph and click **Calibrate from This Image** with **Smart Anomaly Smoothing** enabled to extract the shelf graph, group matching product facings, smooth out singleton intruders in the expected blueprint, and flag misplaced items and physical gaps automatically.
3. **Interactive Sidebar Controls:** Tune `conf_threshold`, `similarity_threshold`, `gap_width_multiplier`, and `tau_y_ratio` live without restarting the application.

---

## 5. Empirical Benchmark Summary

### 5.1 Stage 2A Dense Product Localisation on SKU-110K (NVIDIA L4 GPU)

**Table 3** summarises the training progression of `yolov8m.pt` (`nc=1`, `imgsz=1024`, `batch=16`, `max_det=600`) across 8,216 training images and 588 validation images (90,968 validation instances) on Google Colab Pro.

| Epoch | Box Loss (CIoU) | Cls Loss (BCE) | DFL Loss | Precision (%) | Recall (%) | mAP@50 (%) | mAP@50-95 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | 1.432 | 0.8016 | 1.140 | 87.0 | 81.5 | 88.1 | 52.6 |
| **Epoch 5** | 1.334 | 0.6135 | 1.091 | 89.9 | 86.6 | 92.4 | 57.4 |
| **Epoch 10** | 1.293 | 0.5687 | 1.078 | 90.7 | 87.5 | 93.3 | 58.6 |
| **Epoch 13 (Best)** | **1.280** | **0.5535** | **1.074** | **90.3** | **88.0** | **93.7** | **59.2** |
| *CVPR'19 Baseline* | — | — | — | — | — | — | *49.2* |

### 5.2 Stage 3 Camera Roll Ablation ($N = 500$ Multi-Shelf Scenes)

**Table 4** compares shelf clustering compliance accuracy across camera roll angles $\theta \in [0^\circ, 15^\circ]$ with product height variance $h \sim \mathcal{U}(80, 180)\text{ px}$.

| Shelf Grouping Method | $0^\circ$ Roll | $3^\circ$ Roll | $6^\circ$ Roll | $9^\circ$ Roll | $12^\circ$ Roll | $15^\circ$ Roll |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Centroid Clustering ($y_c$)** | 97.56% | 80.00% | 35.10% | 16.03% | 11.55% | 8.99% |
| **Bottom-Edge $y_{\max}$ DBSCAN (No Deskew)** | 100.00% | 100.00% | 95.58% | 40.56% | 11.01% | 6.99% |
| **PlangoGraph (Pairwise Slope RANSAC + $y_{\max}$ DBSCAN)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

### 5.3 Stage 4 Anomaly Classification Performance ($N = 200$ Perturbed Bays, 756 Anomalies)

**Table 5** contrasts classical one-dimensional string Needleman-Wunsch against PlangoGraph's spatially-augmented Needleman-Wunsch incorporating Stage 3 `__PHYSICAL_GAP__` tokens.

| Alignment Variant | Anomaly Category | True Positives | False Positives | False Negatives | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical 1D String NW** | Missing Item (OOS) | 329 | 0 | 33 | 100.00 | 90.88 | 95.22 |
| **Classical 1D String NW** | Misplaced Substitution | 291 | 33 | 0 | 89.81 | 100.00 | 94.63 |
| **Classical 1D String NW** | Extra Insertion | 69 | 0 | 33 | 100.00 | 67.65 | 80.70 |
| **PlangoGraph Spatial NW** | **Missing Item (OOS)** | **359** | **5** | **3** | **98.63** | **99.17** | **98.90** |
| **PlangoGraph Spatial NW** | **Misplaced Substitution** | **286** | **3** | **5** | **98.96** | **98.28** | **98.62** |
| **PlangoGraph Spatial NW** | **Extra Insertion** | **99** | **5** | **3** | **95.19** | **97.06** | **96.12** |

---

## 6. Scope Boundaries and Future Roadmap

Because PlangoGraph operates on single monocular RGB images, it audits front-facing slot compliance rather than counting recessed inventory depth hidden behind the front facing; that is a deliberate boundary of monocular auditing rather than an oversight. Three extensions are planned for future work:

1. **Aisle-Wide Panorama Stitching:** Stitching multi-frame video sweeps across long supermarket aisles using sequential homography tracking.
2. **Monocular Volumetric Depth Estimation:** Estimating volumetric back-of-shelf stock depth by pairing 2D bounding boxes with monocular depth maps from Depth-Anything-V2.
3. **Price-Rail OCR Verification:** Attaching optical character recognition (OCR) on shelf price rails to verify promotional price tags against the planogram database.

---

## 7. License

Copyright 2026 PlangoGraph Contributors. Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for full terms and conditions.
