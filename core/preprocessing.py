"""Stage 1: Vision Preprocessing, CLAHE Normalization & RANSAC/Hough Perspective Correction."""

from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np


class ShelfPreprocessor:
    """Normalizes retail shelf images via CLAHE and corrects camera tilt/perspective."""

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        max_tilt_deg: float = 15.0,
    ) -> None:
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.max_tilt_deg = max_tilt_deg

    def apply_clahe(self, image_bgr: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization on the L-channel in LAB space."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size
        )
        l_norm = clahe.apply(l_chan)
        merged = cv2.merge((l_norm, a_chan, b_chan))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def estimate_hough_tilt_angle(self, image_bgr: np.ndarray) -> float:
        """Estimates global horizontal shelf tilt angle (degrees) via Canny + HoughLinesP."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=80,
            minLineLength=max(40, image_bgr.shape[1] // 6),
            maxLineGap=20,
        )
        if lines is None or len(lines) == 0:
            return 0.0
        angles: List[float] = []
        for line in lines:
            coords = np.asarray(line).reshape(-1)
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = coords[:4]
            if x2 == x1:
                continue
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= self.max_tilt_deg:
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0

    @staticmethod
    def estimate_ransac_bbox_tilt(
        bboxes: List[Tuple[float, float, float, float]],
        min_samples: int = 3,
        residual_threshold: float = 1.2,
        iterations: int = 100,
        rng_seed: int = 42,
    ) -> float:
        """Multi-shelf pairwise slope RANSAC consensus on bounding-box bottom centers (xc, ymax).
        Because intra-shelf pairs across all shelves share the same tilt slope tan(theta) regardless
        of vertical shelf intercept, consensus in slope space pools statistical power across the entire bay."""
        if len(bboxes) < min_samples:
            return 0.0
        pts = np.array(
            [[(b[0] + b[2]) * 0.5, b[3]] for b in bboxes], dtype=np.float64
        )
        heights = np.array([b[3] - b[1] for b in bboxes], dtype=np.float64)
        med_h = float(np.median(heights)) if len(heights) > 0 else 120.0

        pair_angles: List[float] = []
        pair_weights: List[float] = []
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[j, 0] - pts[i, 0]
                dy = pts[j, 1] - pts[i, 1]
                if abs(dx) < 50.0:
                    continue
                # Filter out cross-shelf pairs using vertical distance bound
                if abs(dy) > max(med_h * 0.85, abs(dx) * np.tan(np.radians(16.5))):
                    continue
                ang = float(np.degrees(np.arctan2(dy, dx)))
                if abs(ang) <= 16.0:
                    pair_angles.append(ang)
                    pair_weights.append(abs(dx))

        if not pair_angles:
            return 0.0
        angles_arr = np.array(pair_angles, dtype=np.float64)
        weights_arr = np.array(pair_weights, dtype=np.float64)

        best_score = -1.0
        best_angle = float(np.median(angles_arr))
        for cand in angles_arr:
            mask = np.abs(angles_arr - cand) <= residual_threshold
            score = float(np.sum(weights_arr[mask]))
            if score > best_score:
                best_score = score
                best_angle = float(np.average(angles_arr[mask], weights=weights_arr[mask]))
        return best_angle

    @staticmethod
    def deskew_bboxes(
        bboxes: List[Tuple[float, float, float, float]],
        tilt_angle_deg: float,
        image_center: Tuple[float, float] = (600.0, 400.0),
    ) -> List[Tuple[float, float, float, float]]:
        """Rotates bounding box coordinates by -tilt_angle_deg around image_center to restore
        fronto-parallel horizontal alignment without pixel interpolation artifacts."""
        if abs(tilt_angle_deg) < 0.15:
            return list(bboxes)
        rad = -np.radians(tilt_angle_deg)
        cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
        cx0, cy0 = image_center
        corrected: List[Tuple[float, float, float, float]] = []
        for xmin, ymin, xmax, ymax in bboxes:
            w, h = xmax - xmin, ymax - ymin
            xc, yc = (xmin + xmax) * 0.5 - cx0, (ymin + ymax) * 0.5 - cy0
            xc_new = xc * cos_a - yc * sin_a + cx0
            yc_new = xc * sin_a + yc * cos_a + cy0
            corrected.append(
                (xc_new - w * 0.5, yc_new - h * 0.5, xc_new + w * 0.5, yc_new + h * 0.5)
            )
        return corrected

    @staticmethod
    def rectify_quadratic_shelf_curve(
        bboxes: List[Tuple[float, float, float, float]],
        shelf_labels: np.ndarray | List[int],
    ) -> List[Tuple[float, float, float, float]]:
        """Fits shared 2nd-order parabolic shelf curvature a * xc_norm^2 across shelves with >= 3
        boxes and subtracts vertical sag from each box's ymin and ymax."""
        return rectify_quadratic_shelf_curve(bboxes, shelf_labels)

    def preprocess_image(
        self, image_bgr: np.ndarray, apply_warp: bool = True
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Runs CLAHE normalization and optional affine deskew warp.
        Returns (normalized_image, estimated_tilt_deg, affine_matrix_2x3)."""
        norm_img = self.apply_clahe(image_bgr)
        tilt_deg = self.estimate_hough_tilt_angle(norm_img)
        h, w = norm_img.shape[:2]
        center = (w * 0.5, h * 0.5)
        rot_mat = cv2.getRotationMatrix2D(center, tilt_deg, 1.0)
        if apply_warp and abs(tilt_deg) > 0.25:
            warped = cv2.warpAffine(
                norm_img,
                rot_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return warped, tilt_deg, rot_mat
        return norm_img, tilt_deg, np.eye(2, 3, dtype=np.float64)

    def preprocess(
        self, image_bgr: np.ndarray, apply_warp: bool = False
    ) -> Tuple[np.ndarray, float]:
        """Runs CLAHE normalization and returns (normalized_image, estimated_tilt_deg)."""
        norm_img, tilt_deg, _ = self.preprocess_image(image_bgr, apply_warp=apply_warp)
        return norm_img, tilt_deg


def rectify_quadratic_shelf_curve(
    bboxes: List[Tuple[float, float, float, float]],
    shelf_labels: np.ndarray | List[int],
) -> List[Tuple[float, float, float, float]]:
    """Rectifies wide-angle parabolic shelf curvature (y_max = a * xc_norm^2 + b * xc + c_k).
    Centers xc around the midpoint xc_norm = (xc - mean_xc) / 500.0, fits shared 2nd-order
    curvature term a * xc_norm^2 across intra-shelf residuals for shelves with >= 3 boxes,
    and subtracts the vertical sag from each box's ymin and ymax."""
    if len(bboxes) < 3:
        return list(bboxes)
    labels = np.asarray(shelf_labels, dtype=int)
    xc = np.array([(b[0] + b[2]) * 0.5 for b in bboxes], dtype=np.float64)
    y_max = np.array([b[3] for b in bboxes], dtype=np.float64)
    xc_norm = (xc - float(np.mean(xc))) / 500.0
    x2 = xc_norm ** 2

    res_x2: List[np.ndarray] = []
    res_y: List[np.ndarray] = []
    for s_id in np.unique(labels):
        mask = labels == s_id
        if int(np.sum(mask)) >= 3:
            res_x2.append(x2[mask] - np.mean(x2[mask]))
            res_y.append(y_max[mask] - np.mean(y_max[mask]))

    if not res_x2:
        return list(bboxes)

    rx2_all = np.concatenate(res_x2)
    ry_all = np.concatenate(res_y)
    denom = float(np.dot(rx2_all, rx2_all))
    if denom < 1e-9:
        return list(bboxes)
    a = float(np.dot(rx2_all, ry_all)) / denom

    return [
        (xmin, ymin - float(a * x2[i]), xmax, ymax - float(a * x2[i]))
        for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes)
    ]
