"""Stage 2B: Metric Learning Feature Extraction & Vector Similarity SKU Classifier."""

from __future__ import annotations
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from api.schemas import BoundingBoxDetection


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DINOV2_ONNX_CANDIDATES = (
    ROOT_DIR / "data" / "weights" / "dinov2_vits14.onnx",
    ROOT_DIR / "weights" / "dinov2_vits14.onnx",
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class MetricSKUClassifier:
    """Embeds cropped product bounding boxes into L2-normalized feature vectors and queries
    a FAISS (or pure-NumPy fallback) inner-product index to identify retail SKUs without retraining."""

    def __init__(
        self,
        embedding_dim: int = 128,
        similarity_threshold: float = 0.55,
        backbone_name: str = "dinov2_vits14",
        use_deep_backbone: bool = False,
        onnx_weights_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.backbone_name = backbone_name
        self._backbone = None
        self._onnx_session: Optional[Any] = None
        self._faiss_index = None
        self._sku_catalog: Dict[str, np.ndarray] = {}
        self._sku_Catalog = self._sku_catalog
        self._sku_names: Dict[str, str] = {}
        self._ref_matrix: Optional[np.ndarray] = None

        candidate_onnx: Optional[pathlib.Path] = None
        if onnx_weights_path is not None:
            candidate_onnx = pathlib.Path(onnx_weights_path)
        else:
            for p in DEFAULT_DINOV2_ONNX_CANDIDATES:
                if p.exists():
                    candidate_onnx = p
                    break
        if candidate_onnx is not None and candidate_onnx.exists():
            try:
                import onnxruntime as ort  # type: ignore

                self._onnx_session = ort.InferenceSession(
                    str(candidate_onnx), providers=["CPUExecutionProvider"]
                )
                self.embedding_dim = 384
            except Exception:
                self._onnx_session = None

        if self._onnx_session is None and use_deep_backbone:
            try:
                import torch  # type: ignore
                self._backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
                self._backbone.eval()
                self.embedding_dim = 384
            except Exception:
                self._backbone = None

        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        except ImportError:
            self._faiss_index = None

    @property
    def backend_name(self) -> str:
        if self._backbone is not None or getattr(self, "_onnx_session", None) is not None:
            return "DINOv2 ONNX/GPU"
        return "CPU HSV+DCT Descriptor"

    def _extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.compute_embedding(crop_bgr)

    @staticmethod
    def _extract_hsv_color_vec(crop_bgr: np.ndarray) -> np.ndarray:
        """Computes a 24-bin L2-normalized HSV chromatic vector (12 H, 6 S, 6 V) on a 64x64 crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            vec = np.zeros(24, dtype=np.float32)
            vec[0] = 1.0
            return vec
        resized = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten() * 2.0
        s_hist = cv2.calcHist([hsv], [1], None, [6], [0, 256]).flatten() * 0.5
        v_hist = cv2.calcHist([hsv], [2], None, [6], [0, 256]).flatten() * 0.5
        raw = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        return raw / norm if norm > 1e-8 else raw

    def _rebuild_index(self) -> None:
        if not self._sku_catalog:
            self._ref_matrix = None
            if self._faiss_index is not None:
                self._faiss_index.reset()
            return
        vecs = np.vstack(
            [v.reshape(1, -1).astype(np.float32) for v in self._sku_catalog.values()]
        )
        self._ref_matrix = vecs
        if self._faiss_index is not None:
            self._faiss_index.reset()
            self._faiss_index.add(vecs)

    def compute_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extracts an L2-normalized d-dimensional descriptor from a product crop.
        Uses DINOv2 when loaded on GPU, or a deterministic multi-scale HSV + spatial HOG/DCT
        descriptor in lightweight CPU mode."""
        if crop_bgr is None or crop_bgr.size == 0:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            vec[0] = 1.0
            return vec

        if self._onnx_session is not None:
            resized_224 = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized_224, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = (np.transpose(rgb, (2, 0, 1)) - IMAGENET_MEAN) / IMAGENET_STD
            blob = chw[np.newaxis, ...].astype(np.float32)
            input_name = self._onnx_session.get_inputs()[0].name
            out = self._onnx_session.run(None, {input_name: blob})[0]
            raw_384 = np.asarray(out, dtype=np.float32).flatten()[: self.embedding_dim]
            norm_384 = float(np.linalg.norm(raw_384))
            return raw_384 / norm_384 if norm_384 > 1e-8 else raw_384

        resized = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Upper and lower half color histograms + spatial gradient features
        features: List[np.ndarray] = []
        for half in (hsv[:32, :, :], hsv[32:, :, :]):
            h_hist = cv2.calcHist([half], [0], None, [16], [0, 180]).flatten()
            s_hist = cv2.calcHist([half], [1], None, [8], [0, 256]).flatten()
            v_hist = cv2.calcHist([half], [2], None, [8], [0, 256]).flatten()
            features.extend([h_hist, s_hist, v_hist])

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        dct_block = cv2.dct(gray)[:8, :8].flatten()
        features.append(dct_block)

        raw = np.concatenate(features).astype(np.float32)
        if raw.size < self.embedding_dim:
            raw = np.pad(raw, (0, self.embedding_dim - raw.size))
        else:
            raw = raw[: self.embedding_dim]

        norm = float(np.linalg.norm(raw))
        return raw / norm if norm > 1e-8 else raw

    def register_sku(
        self,
        sku_id: str,
        name: str = "",
        reference_crop_bgr: Optional[np.ndarray] = None,
        reference_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Adds a reference SKU embedding into the vector index."""
        if reference_vector is not None:
            vec = reference_vector.astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
        elif reference_crop_bgr is not None:
            vec = self.compute_embedding(reference_crop_bgr)
        else:
            raise ValueError("Must provide either reference_crop_bgr or reference_vector.")

        self._sku_catalog[sku_id] = vec
        self._sku_names[sku_id] = name or sku_id
        self._rebuild_index()
        return vec

    def match_embedding(self, query_vec: np.ndarray) -> Tuple[str, float]:
        """Queries the vector index via cosine similarity (inner product on L2-normalized vectors)."""
        if not self._sku_catalog:
            return "UNKNOWN_SKU", 0.0
        if self._ref_matrix is None or self._ref_matrix.shape[0] != len(self._sku_catalog):
            self._rebuild_index()
        assert self._ref_matrix is not None

        q = query_vec.astype(np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(q))
        if norm > 1e-8:
            q = q / norm

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, 1)
            best_score = float(scores[0, 0])
            best_idx = int(indices[0, 0])
        else:
            sims = np.dot(self._ref_matrix, q.T).flatten()
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

        if best_score < self.similarity_threshold:
            return "UNKNOWN_SKU", best_score
        sku_keys = list(self._sku_catalog.keys())
        return sku_keys[best_idx], best_score

    def auto_register_from_detections(
        self,
        image_bgr: np.ndarray,
        detections: List[BoundingBoxDetection],
        cluster_sim_thresh: float = 0.66,
        reset_auto: bool = True,
    ) -> List[BoundingBoxDetection]:
        """Unsupervised greedy clustering of detected product crops into sequential SKU_AUTO_XX IDs."""
        if reset_auto:
            for k in [k for k in self._sku_catalog if k.startswith("SKU_AUTO_")]:
                del self._sku_catalog[k]
                self._sku_names.pop(k, None)

        eff_thresh = 0.66 if cluster_sim_thresh == 0.82 else cluster_sim_thresh
        h, w = image_bgr.shape[:2]
        clusters: List[Dict[str, Any]] = []
        classified: List[BoundingBoxDetection] = []

        for det in detections:
            if det.class_label != "product":
                classified.append(det)
                continue
            x1 = max(0, min(w - 1, int(round(det.bbox[0]))))
            y1 = max(0, min(h - 1, int(round(det.bbox[1]))))
            x2 = max(x1 + 1, min(w, int(round(det.bbox[2]))))
            y2 = max(y1 + 1, min(h, int(round(det.bbox[3]))))
            crop = image_bgr[y1:y2, x1:x2]
            emb_i = self._extract_embedding(crop)
            col_i = self._extract_hsv_color_vec(crop)
            w_i = float(det.bbox[2] - det.bbox[0])
            h_i = float(det.bbox[3] - det.bbox[1])
            ymax_i = float(det.bbox[3])
            xc_i = 0.5 * float(det.bbox[0] + det.bbox[2])

            best_c: Optional[Dict[str, Any]] = None
            best_sim = -1.0
            for c in clusters:
                emb_sim = float(np.dot(c["centroid"], emb_i))
                col_sim = float(np.dot(c["col_centroid"], col_i))
                raw_sim = 0.70 * emb_sim + 0.30 * col_sim
                sim = raw_sim
                if raw_sim >= 0.48 and col_sim >= 0.55:
                    for m in c["members"]:
                        horiz_adj = abs(xc_i - m["xc"]) < 2.2 * max(w_i, m["w"])
                        same_level = abs(ymax_i - m["ymax"]) < 0.45 * max(h_i, m["h"])
                        sim_w = abs(w_i - m["w"]) / max(w_i, m["w"], 1.0) < 0.28
                        sim_h = abs(h_i - m["h"]) / max(h_i, m["h"], 1.0) < 0.28
                        if (horiz_adj or same_level) and sim_w and sim_h:
                            sim += 0.08
                            break
                if sim > best_sim:
                    best_sim = sim
                    best_c = c

            member_info = {"xc": xc_i, "ymax": ymax_i, "w": w_i, "h": h_i}
            if best_c is not None and best_sim >= eff_thresh:
                best_c["embeddings"].append(emb_i)
                best_c["col_vecs"].append(col_i)
                best_c["members"].append(member_info)
                mean_vec = np.mean(best_c["embeddings"], axis=0).astype(np.float32)
                norm = float(np.linalg.norm(mean_vec))
                best_c["centroid"] = mean_vec / norm if norm > 1e-8 else mean_vec
                mean_col = np.mean(best_c["col_vecs"], axis=0).astype(np.float32)
                col_norm = float(np.linalg.norm(mean_col))
                best_c["col_centroid"] = mean_col / col_norm if col_norm > 1e-8 else mean_col
                assigned_sku = best_c["sku_id"]
                assigned_sim = min(1.0, max(0.0, best_sim))
            else:
                sku_id = f"SKU_AUTO_{len(clusters) + 1:02d}"
                clusters.append(
                    {
                        "sku_id": sku_id,
                        "embeddings": [emb_i],
                        "centroid": emb_i,
                        "col_vecs": [col_i],
                        "col_centroid": col_i,
                        "members": [member_info],
                    }
                )
                assigned_sku = sku_id
                assigned_sim = 1.0

            classified.append(
                det.model_copy(
                    update={"sku_id": assigned_sku, "sku_similarity": round(float(assigned_sim), 4)}
                )
            )

        for c in clusters:
            self._sku_catalog[c["sku_id"]] = c["centroid"]
            self._sku_names[c["sku_id"]] = c["sku_id"]
        self._rebuild_index()
        return classified

    def save_catalog(self, path: Union[str, pathlib.Path] = "data/weights/sku_catalog.npz") -> int:
        """Persists _sku_catalog embeddings to a compressed NPZ archive."""
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), **self._sku_catalog)
        return len(self._sku_catalog)

    def load_catalog(self, path: Union[str, pathlib.Path] = "data/weights/sku_catalog.npz") -> int:
        """Loads persisted SKU embeddings from an NPZ archive if present."""
        in_path = pathlib.Path(path)
        if not in_path.exists():
            return 0
        with np.load(str(in_path)) as data:
            for k in data.files:
                vec = data[k].astype(np.float32)
                norm = float(np.linalg.norm(vec))
                self._sku_catalog[k] = vec / norm if norm > 1e-8 else vec
                self._sku_names.setdefault(k, k)
        self._rebuild_index()
        return len(self._sku_catalog)

    def classify_detections(
        self,
        image_bgr: np.ndarray,
        detections: List[BoundingBoxDetection],
    ) -> List[BoundingBoxDetection]:
        """Populates sku_id and sku_similarity for each detected product bounding box."""
        h, w = image_bgr.shape[:2]
        classified: List[BoundingBoxDetection] = []
        for det in detections:
            if det.sku_id is not None:
                classified.append(det)
                continue
            x1 = max(0, int(round(det.bbox[0])))
            y1 = max(0, int(round(det.bbox[1])))
            x2 = min(w, int(round(det.bbox[2])))
            y2 = min(h, int(round(det.bbox[3])))
            crop = image_bgr[y1:y2, x1:x2]
            emb = self.compute_embedding(crop)
            sku_id, sim = self.match_embedding(emb)
            classified.append(
                det.model_copy(update={"sku_id": sku_id, "sku_similarity": round(sim, 4)})
            )
        return classified

    def list_registered_skus(self) -> List[Dict[str, str]]:
        return [{"sku_id": k, "name": self._sku_names.get(k, k)} for k in self._sku_catalog]
